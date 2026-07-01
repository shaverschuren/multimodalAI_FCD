"""
Train output-level MRI+EEG multimodal fusion with a minimal residual MGU.

Default trainability:
- MRI model: frozen
- EEG model: frozen
- MGU fusion module: trainable
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.multimodal import (
    MultimodalMRIEEGPatchDataset,
    multimodal_mri_eeg_collate,
)
from datasets.eeg import load_split
from models.eeg import SpikeMILModel
from models.mri import (
    ResEncUNet_3D,
    expand_mismatched_conv3d_weights_to_match_model,
    prefix_mri_checkpoint_state_dict_for_wrapper,
)
from models.multimodal import MGU3D, MGUOutputFusionModel
from util import EarlyStopping, emit_run_fingerprint


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.softmax(logits, dim=1)
    if logits.shape[1] == 2:
        probs = probs[:, 1]
    else:
        probs = probs.max(dim=1)[0]

    targets_float = targets.float()
    inter = (probs * targets_float).sum()
    union = probs.sum() + targets_float.sum()
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice


def _as_fg_logit(logits_or_fg):
    """
    Convert either [B, 1, D, H, W] foreground logits or [B, 2, D, H, W]
    two-class logits to [B, 1, D, H, W] foreground logits.
    """
    if logits_or_fg.ndim != 5:
        raise ValueError(f"Expected 5D logits, got shape {tuple(logits_or_fg.shape)}")
    if logits_or_fg.shape[1] == 1:
        return logits_or_fg
    if logits_or_fg.shape[1] == 2:
        return logits_or_fg[:, 1:2]
    raise ValueError(f"Unsupported logits channel count: {logits_or_fg.shape[1]}")


def _target_fg_5d(target):
    """
    Convert target [B, D, H, W] or [B, 1, D, H, W] to float [B, 1, D, H, W].
    """
    if target.ndim == 4:
        return (target.long() == 1).float().unsqueeze(1)
    if target.ndim == 5 and target.shape[1] == 1:
        return (target.long() == 1).float()
    raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")


def _safe_odd_kernel(k):
    k = int(k)
    if k < 1:
        raise ValueError(f"pool_kernel_size must be >= 1, got {k}")
    if k % 2 == 0:
        k += 1
    return k


def compute_local_gate_target(
    mri_logit,
    eeg_logit,
    target,
    pool_kernel_size=15,
    temperature=0.05,
):
    """
    Computes a soft oracle gate target from local MRI-vs-EEG errors.

    Returns:
        gate_target: [B, 1, D, H, W], values in [0, 1]
        err_mri_s:   [B, 1, D, H, W]
        err_eeg_s:   [B, 1, D, H, W]
    """
    k = _safe_odd_kernel(pool_kernel_size)
    pad = k // 2
    tau = max(float(temperature), 1e-6)

    with torch.no_grad():
        mri_fg = _as_fg_logit(mri_logit)
        eeg_fg = _as_fg_logit(eeg_logit)
        y = _target_fg_5d(target).to(dtype=mri_fg.dtype, device=mri_fg.device)

        p_mri = torch.sigmoid(mri_fg)
        p_eeg = torch.sigmoid(eeg_fg)

        err_mri = torch.abs(p_mri - y)
        err_eeg = torch.abs(p_eeg - y)

        err_mri_s = F.avg_pool3d(err_mri, kernel_size=k, stride=1, padding=pad)
        err_eeg_s = F.avg_pool3d(err_eeg, kernel_size=k, stride=1, padding=pad)

        # If EEG error is larger than MRI error, target should favor MRI.
        gate_target = torch.sigmoid((err_eeg_s - err_mri_s) / tau)

    return gate_target, err_mri_s, err_eeg_s


def compute_disagreement_weight(
    mri_logit,
    eeg_logit,
    disagreement_weight=2.0,
    detach=True,
):
    """
    Returns a voxel weight map [B, 1, D, H, W] that upweights locations where
    MRI and EEG probabilities disagree.
    """
    mri_fg = _as_fg_logit(mri_logit)
    eeg_fg = _as_fg_logit(eeg_logit)

    p_mri = torch.sigmoid(mri_fg)
    p_eeg = torch.sigmoid(eeg_fg)

    disagreement = torch.abs(p_mri - p_eeg)
    if detach:
        disagreement = disagreement.detach()

    return 1.0 + float(disagreement_weight) * disagreement


def weighted_bce_with_logits_fg(fg_logit, target, weight=None):
    y = _target_fg_5d(target).to(dtype=fg_logit.dtype, device=fg_logit.device)
    loss = F.binary_cross_entropy_with_logits(fg_logit, y, reduction="none")
    if weight is not None:
        loss = loss * weight.to(dtype=loss.dtype, device=loss.device)
    return loss.mean()


def weighted_dice_loss_fg(fg_logit, target, weight=None, smooth=1.0):
    y = _target_fg_5d(target).to(dtype=fg_logit.dtype, device=fg_logit.device)
    p = torch.sigmoid(fg_logit)

    if weight is None:
        weight = torch.ones_like(p)
    else:
        weight = weight.to(dtype=p.dtype, device=p.device)

    dims = tuple(range(1, p.ndim))
    inter = (weight * p * y).sum(dim=dims)
    denom = (weight * p).sum(dim=dims) + (weight * y).sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def weighted_bce_dice_loss_fg(
    fg_logit,
    target,
    weight=None,
    bce_weight=0.7,
    dice_weight=0.3,
):
    bce = weighted_bce_with_logits_fg(fg_logit, target, weight=weight)
    dl = weighted_dice_loss_fg(fg_logit, target, weight=weight)
    return float(bce_weight) * bce + float(dice_weight) * dl, bce, dl


def compute_gate_supervision_loss(
    gate,
    gate_target,
    loss_type="bce",
):
    """
    gate may be [B, hidden_channels, D, H, W].
    gate_target is [B, 1, D, H, W].

    For now, average gate channels to get one scalar reliability gate per voxel.
    """
    if gate.ndim != 5:
        raise ValueError(f"Expected 5D gate, got shape {tuple(gate.shape)}")

    gate_scalar = gate.mean(dim=1, keepdim=True)

    if gate_target.shape != gate_scalar.shape:
        if gate_target.shape[1] == 1 and gate_target.shape[2:] == gate_scalar.shape[2:]:
            pass
        else:
            raise ValueError(
                f"Gate target shape {tuple(gate_target.shape)} incompatible with "
                f"gate scalar shape {tuple(gate_scalar.shape)}"
            )

    if loss_type == "bce":
        # Use binary_cross_entropy_with_logits (safe for autocast) instead of binary_cross_entropy
        return F.binary_cross_entropy_with_logits(gate_scalar, gate_target), gate_scalar

    if loss_type == "mse":
        return F.mse_loss(gate_scalar, gate_target), gate_scalar

    raise ValueError(f"Unknown gate supervision loss type: {loss_type}")


def _foreground_probability(logits):
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    raise ValueError(f"Unsupported logits shape for probability conversion: {tuple(logits.shape)}")


def _dice_from_logits(logits, targets, smooth=1.0):
    pred_fg = _foreground_probability(logits)
    tgt_fg = (targets.long() == 1).float()
    inter = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
    union = pred_fg.flatten(1).sum(dim=1) + tgt_fg.flatten(1).sum(dim=1)
    return ((2.0 * inter + smooth) / (union + smooth)).mean()


def _precision_recall_from_logits(logits, targets, smooth=1.0):
    if logits.shape[1] == 2:
        pred_fg = (torch.argmax(logits, dim=1) == 1).float()
    elif logits.shape[1] == 1:
        pred_fg = (torch.sigmoid(logits[:, 0]) > 0.5).float()
    else:
        raise ValueError(f"Unsupported logits channels for precision/recall: {logits.shape[1]}")

    tgt_fg = (targets.long() == 1).float()

    tp = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
    pred_pos = pred_fg.flatten(1).sum(dim=1)
    tgt_pos = tgt_fg.flatten(1).sum(dim=1)

    precision = (tp + smooth) / (pred_pos + smooth)
    recall = (tp + smooth) / (tgt_pos + smooth)
    return precision.mean(), recall.mean()


def _build_cases(subject_ids, mri_data_root):
    cases = []
    for sid in subject_ids:
        npz_path = os.path.join(mri_data_root, sid, f"{sid}_preproc.npz")
        if os.path.exists(npz_path):
            cases.append({"id": sid, "npy": npz_path})
    if not cases:
        raise RuntimeError(f"No valid MRI cases found under {mri_data_root}")
    return cases


def _resolve_split_ids(args):
    train_ids, val_ids = load_split(args.splits_json, args.fold)
    train_ids = [str(v) for v in train_ids]
    val_ids = [str(v) for v in val_ids]
    test_ids = []

    if args.infer_test_set:
        try:
            with open(args.splits_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fold_key = f"fold_{args.fold}"
            test_ids = [str(v) for v in payload.get("folds", {}).get(fold_key, {}).get("test_ids", [])]
        except Exception as e:
            warnings.warn(f"Failed to load test_ids from splits JSON: {e}")
            test_ids = []

    print(f"Loaded fold {args.fold} from splits JSON.")

    if args.test_mode:
        train_ids = train_ids[: min(len(train_ids), 8)]
        val_ids = val_ids[: min(len(val_ids), 8)]
        test_ids = test_ids[: min(len(test_ids), 8)]
        print("Test mode active: truncated train/val subject lists to at most 8 each.")

    print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}, Test subjects: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(output, dict):
        for key in ("logits", "out", "pred", "prediction"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Could not extract logits from model output type {type(output)!r}")


def compute_gaussian_importance_map(
    patch_size,
    sigma_scale=1 / 8,
    value_scaling_factor=10.0,
    dtype=torch.float32,
):
    patch_size = tuple(int(v) for v in patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"Expected 3D patch_size, got {patch_size!r}")

    center = tuple(v // 2 for v in patch_size)
    coords = [torch.arange(v, dtype=torch.float32) for v in patch_size]
    zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    sigmas = [max(float(v) * float(sigma_scale), 1e-6) for v in patch_size]
    gaussian = torch.exp(
        -(
            ((zz - float(center[0])) ** 2) / (2.0 * sigmas[0] ** 2)
            + ((yy - float(center[1])) ** 2) / (2.0 * sigmas[1] ** 2)
            + ((xx - float(center[2])) ** 2) / (2.0 * sigmas[2] ** 2)
        )
    )

    max_val = float(torch.max(gaussian).item())
    if max_val <= 0.0:
        raise RuntimeError("Gaussian importance map is all zeros.")

    gaussian = gaussian / max_val
    gaussian = gaussian * float(value_scaling_factor)

    nonzero = gaussian[gaussian > 0]
    if nonzero.numel() == 0:
        raise RuntimeError("Gaussian importance map contains no positive values.")
    min_nonzero = float(nonzero.min().item())
    gaussian = torch.where(gaussian == 0, torch.tensor(min_nonzero, dtype=gaussian.dtype), gaussian)

    return gaussian.to(dtype=dtype).unsqueeze(0)


def _load_inference_eeg_input(eeg_root, subject_id, window_size=128, max_spikes=32):
    eeg_path = os.path.join(eeg_root, f"{subject_id}_spikes_1-70Hz.npy")
    spikes = np.load(eeg_path)
    if spikes.shape[0] > max_spikes:
        idx = np.random.choice(spikes.shape[0], max_spikes, replace=False)
        spikes = spikes[idx]

    l = spikes.shape[-1]
    s0 = l // 2 - window_size // 2
    s1 = s0 + window_size
    spikes = spikes[:, :, s0:s1]

    spikes_t = torch.from_numpy(spikes.astype(np.float32)).unsqueeze(0)
    mask_t = torch.ones((1, spikes_t.shape[1]), dtype=torch.float32)
    return {"spikes": spikes_t, "mask": mask_t}


def sliding_window_inference_mgu(
    model,
    mri_image,
    eeg_input,
    patch_size,
    stride,
    device,
    amp_enabled=True,
    use_gaussian_weighting=True,
    gaussian_sigma_scale=1 / 8,
    gaussian_value_scaling_factor=10.0,
    accumulate_on_cpu=True,
):
    model.eval()
    if isinstance(mri_image, np.ndarray):
        mri_image = torch.from_numpy(mri_image).float()
    elif torch.is_tensor(mri_image):
        mri_image = mri_image.float()
    else:
        raise TypeError(f"Expected mri_image to be numpy.ndarray or torch.Tensor, got {type(mri_image)!r}")

    _, d, h, w = mri_image.shape
    original_shape = (d, h, w)
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        mri_image = F.pad(mri_image, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)

    _, d_pad, h_pad, w_pad = mri_image.shape

    def starts(dim, p, s):
        if dim <= p:
            return [0]
        out = list(range(0, dim - p + 1, s))
        if out[-1] != (dim - p):
            out.append(dim - p)
        return out

    d_starts = starts(d_pad, pd, stride[0])
    h_starts = starts(h_pad, ph, stride[1])
    w_starts = starts(w_pad, pw, stride[2])
    n_tiles = len(d_starts) * len(h_starts) * len(w_starts)

    overlap = tuple(1.0 - (float(stride[i]) / float(patch_size[i])) for i in range(3))
    print(
        "[inference] sliding-window config | "
        f"patch_size={patch_size}, stride={tuple(stride)}, overlap={overlap}, "
        f"tiles={n_tiles}, gaussian_weighting={bool(use_gaussian_weighting)}, "
        f"accumulate_on_cpu={bool(accumulate_on_cpu)}"
    )

    accumulator_device = torch.device("cpu") if accumulate_on_cpu else device

    gaussian = None
    if use_gaussian_weighting:
        gaussian = compute_gaussian_importance_map(
            patch_size=patch_size,
            sigma_scale=gaussian_sigma_scale,
            value_scaling_factor=gaussian_value_scaling_factor,
            dtype=torch.float32,
        ).to(accumulator_device)

    with torch.no_grad():
        logit_accum = None
        weight_accum = torch.zeros((d_pad, h_pad, w_pad), dtype=torch.float32, device=accumulator_device)
        volume_shape_t = torch.tensor([[d_pad, h_pad, w_pad]], dtype=torch.int32, device=device)

        for ds in d_starts:
            for hs in h_starts:
                for ws in w_starts:
                    de = ds + pd
                    he = hs + ph
                    we = ws + pw

                    patch = mri_image[:, ds:de, hs:he, ws:we]

                    center = np.array([(ds + de - 1) * 0.5, (hs + he - 1) * 0.5, (ws + we - 1) * 0.5], dtype=np.float32)
                    center_norm = 2.0 * center / np.maximum(np.array([d_pad, h_pad, w_pad], dtype=np.float32) - 1.0, 1.0) - 1.0
                    center_t = torch.from_numpy(center_norm).view(1, 3).to(device)

                    patch_t = patch.unsqueeze(0).to(device)
                    eeg_batch = {
                        "spikes": eeg_input["spikes"].to(device),
                        "mask": eeg_input["mask"].to(device),
                    }
                    patch_bbox_t = torch.tensor([[ds, hs, ws, de, he, we]], dtype=torch.int32, device=device)

                    with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                        patch_fg_logit = model(
                            mri_patch=patch_t,
                            eeg_input=eeg_batch,
                            patch_center=center_t,
                            patch_bbox=patch_bbox_t,
                            volume_shape=volume_shape_t,
                            return_aux=False,
                        )
                        patch_fg_logit = _extract_logits(patch_fg_logit)

                    if patch_fg_logit.ndim != 5 or patch_fg_logit.shape[0] != 1 or patch_fg_logit.shape[1] != 1:
                        raise RuntimeError(
                            "Expected patch foreground logits shape (1, 1, pd, ph, pw), got "
                            f"{tuple(patch_fg_logit.shape)}"
                        )

                    patch_logits = patch_fg_logit[0, 0].detach().to(accumulator_device, dtype=torch.float32)

                    if logit_accum is None:
                        logit_accum = torch.zeros((d_pad, h_pad, w_pad), dtype=torch.float32, device=accumulator_device)

                    if tuple(patch_logits.shape) != (pd, ph, pw):
                        raise RuntimeError(
                            f"Unexpected patch logit spatial shape {tuple(patch_logits.shape)}; expected {(pd, ph, pw)}"
                        )

                    if use_gaussian_weighting:
                        weighted_logits = patch_logits * gaussian[0]
                        patch_weight = gaussian[0]
                    else:
                        weighted_logits = patch_logits
                        patch_weight = torch.ones((pd, ph, pw), dtype=torch.float32, device=accumulator_device)

                    logit_accum[ds:de, hs:he, ws:we] += weighted_logits
                    weight_accum[ds:de, hs:he, ws:we] += patch_weight

    assert logit_accum is not None, "No inference tiles were processed."
    assert tuple(logit_accum.shape) == (d_pad, h_pad, w_pad), (
        f"Final logit shape mismatch: got {tuple(logit_accum.shape)}, expected {(d_pad, h_pad, w_pad)}"
    )

    weight_min = float(weight_accum.min().item())
    assert weight_min > 0.0, f"weight_accum.min() must be > 0, got {weight_min}"

    final_logits = logit_accum / torch.clamp_min(weight_accum, 1e-8)

    if not torch.isfinite(final_logits).all():
        warnings.warn("Non-finite values detected in final stitched logits.", RuntimeWarning)

    pred_prob = torch.sigmoid(final_logits)
    pred_prob = pred_prob[: original_shape[0], : original_shape[1], : original_shape[2]]

    if not torch.isfinite(pred_prob).all():
        warnings.warn("Non-finite values detected in final stitched probability map.", RuntimeWarning)

    return pred_prob.detach().cpu().numpy().astype(np.float32)


def _load_mri_checkpoint_strict(mri_model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "network_weights" in ckpt:
        state_dict = ckpt["network_weights"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported MRI checkpoint format: {checkpoint_path}")

    state_dict = prefix_mri_checkpoint_state_dict_for_wrapper(state_dict, verbose=True)

    state_dict = expand_mismatched_conv3d_weights_to_match_model(
        state_dict=state_dict,
        model=mri_model,
        verbose=True,
    )
    missing, unexpected = mri_model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:raise RuntimeError(
            "MRI checkpoint load had key mismatches. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def _build_eeg_model_from_checkpoint(eeg_checkpoint_path, device):
    ckpt = torch.load(eeg_checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model_kwargs = dict(
        in_channels=int(saved_args.get("in_channels", 21)),
        emb_dim=int(saved_args.get("emb_dim", 128)),
        hidden=int(saved_args.get("hidden", 128)),
        dropout=float(saved_args.get("dropout", 0.3)),
        encoder_type=str(saved_args.get("encoder_type", "t_s_cnn")),
        pooling=str(saved_args.get("pooling", "mean")),
        use_coord_head=False,
        use_hemi_head=False,
        use_lobe_head=False,
        spatial_head=str(saved_args.get("spatial_head", "deconv")),
        num_gaussians=int(saved_args.get("num_gaussians", 3)),
        gaussian_coord_dim=int(saved_args.get("gaussian_coord_dim", 3)),
        gaussian_sigma_min=saved_args.get("gaussian_sigma_min", None),
        gaussian_sigma_max=saved_args.get("gaussian_sigma_max", None),
        gaussian_isotropic=bool(saved_args.get("gaussian_isotropic", True)),
        gaussian_output_space=str(saved_args.get("gaussian_output_space", "normalized")),
        gaussian_make_heatmap=bool(saved_args.get("gaussian_make_heatmap", False)),
        gaussian_heatmap_shape=saved_args.get("gaussian_heatmap_shape", None),
        deconv_output_shape=tuple(saved_args.get("deconv_output_shape", (32, 40, 32))),
        deconv_latent_shape=tuple(saved_args.get("deconv_latent_shape", (4, 5, 4))),
        deconv_base_channels=int(saved_args.get("deconv_base_channels", 128)),
        deconv_dropout=float(saved_args.get("deconv_dropout", 0.0)),
    )

    eeg_model = SpikeMILModel(**model_kwargs).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        missing, unexpected = eeg_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        missing, unexpected = eeg_model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    else:
        raise KeyError("EEG checkpoint must contain 'model_state_dict' or 'encoder_state_dict'.")

    print(f"EEG checkpoint loaded. missing={missing}, unexpected={unexpected}")
    return eeg_model


def _set_trainability(model: MGUOutputFusionModel, args):
    train_mri = bool(args.train_mri)
    train_eeg = bool(args.train_eeg)

    if args.train_mgu_only:
        train_mri = False
        train_eeg = False

    if args.freeze_mri:
        train_mri = False
    if args.freeze_eeg:
        train_eeg = False

    for p in model.mri_model.parameters():
        p.requires_grad = bool(train_mri)

    for p in model.eeg_model.parameters():
        p.requires_grad = bool(train_eeg)

    for p in model.mgu.parameters():
        p.requires_grad = True

    return {
        "train_mri": train_mri,
        "train_eeg": train_eeg,
        "train_mgu": True,
    }


def _count_params(params):
    return int(sum(p.numel() for p in params))


def _compose_two_class_logits_from_fg(fg_logit: torch.Tensor, reference_two_class: torch.Tensor | None = None):
    if reference_two_class is not None and reference_two_class.ndim == 5 and reference_two_class.shape[1] == 2:
        bg_logit = reference_two_class[:, :1]
        return torch.cat([bg_logit, fg_logit], dim=1)
    return torch.cat([-fg_logit, fg_logit], dim=1)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _collect_mgu_stats(aux, fused_fg):
    gate = aux["mgu_gate"]
    mri_fg = aux["mri_logit"]
    alpha = aux.get("mgu_alpha", torch.tensor(0.0, device=fused_fg.device, dtype=fused_fg.dtype))

    prob_delta = torch.abs(torch.sigmoid(fused_fg) - torch.sigmoid(mri_fg))
    logit_delta = torch.abs(fused_fg - mri_fg)

    return {
        "mgu/gate_mean": float(gate.detach().mean().cpu()),
        "mgu/gate_std": float(gate.detach().std(unbiased=False).cpu()),
        "mgu/gate_min": float(gate.detach().min().cpu()),
        "mgu/gate_max": float(gate.detach().max().cpu()),
        "mgu/alpha": float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha),
        "mgu/mean_abs_logit_delta": float(logit_delta.detach().mean().cpu()),
        "mgu/mean_abs_prob_delta": float(prob_delta.detach().mean().cpu()),
    }


def _mean_of_dict(list_of_dicts):
    if not list_of_dicts:
        return {}
    keys = sorted({k for d in list_of_dicts for k in d.keys()})
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in list_of_dicts if k in d and np.isfinite(float(d[k]))]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def run_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    train,
    amp_enabled,
    ce_weight,
    dice_weight,
    correction_loss_weight,
    gate_supervision_weight,
    gate_supervision_pool_kernel,
    gate_supervision_temperature,
    gate_supervision_loss_type,
    disagreement_loss_weight,
    use_reliability_supervision,
):
    if train:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "corr_loss": 0.0,
        "gate_loss": 0.0,
        "weighted_bce": 0.0,
        "weighted_dice": 0.0,
        "gate_target_mean": 0.0,
        "gate_target_std": 0.0,
        "disagreement_weight_mean": 0.0,
        "dice_fused": 0.0,
        "precision_fused": 0.0,
        "recall_fused": 0.0,
        "dice_mri": 0.0,
        "precision_mri": 0.0,
        "recall_mri": 0.0,
        "dice_eeg": 0.0,
        "precision_eeg": 0.0,
        "recall_eeg": 0.0,
    }
    mgu_stats_list = []
    total = 0

    for batch in dataloader:
        mri = batch["mri"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        patch_center = batch["patch_center"].to(device, non_blocking=True)
        patch_bbox = batch["patch_bbox"].to(device, non_blocking=True) if "patch_bbox" in batch else None
        volume_shape = batch["volume_shape"].to(device, non_blocking=True) if "volume_shape" in batch else None
        eeg_input = {
            "spikes": batch["eeg_input"]["spikes"].to(device, non_blocking=True),
            "mask": batch["eeg_input"]["mask"].to(device, non_blocking=True),
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                fused_fg_logit, aux = model(
                    mri_patch=mri,
                    eeg_input=eeg_input,
                    patch_center=patch_center,
                    patch_bbox=patch_bbox,
                    volume_shape=volume_shape,
                    return_aux=True,
                )

                mri_logits_full = aux.get("mri_logits_full", None)
                fused_logits = _compose_two_class_logits_from_fg(fg_logit=fused_fg_logit, reference_two_class=mri_logits_full)
                mri_logits = mri_logits_full if (mri_logits_full is not None and mri_logits_full.shape[1] == 2) else _compose_two_class_logits_from_fg(aux["mri_logit"]) 
                eeg_logits = _compose_two_class_logits_from_fg(aux["eeg_logit"], reference_two_class=mri_logits_full)

                correction_loss = torch.mean(torch.abs(fused_fg_logit - aux["mri_logit"].detach()))

                gate_loss = torch.zeros((), device=device, dtype=fused_fg_logit.dtype)
                weighted_bce = torch.zeros((), device=device, dtype=fused_fg_logit.dtype)
                weighted_dice = torch.zeros((), device=device, dtype=fused_fg_logit.dtype)
                gate_target = None
                voxel_weight = None

                if use_reliability_supervision:
                    gate_target, _err_mri_s, _err_eeg_s = compute_local_gate_target(
                        mri_logit=aux["mri_logit"].detach(),
                        eeg_logit=aux["eeg_logit"].detach(),
                        target=target,
                        pool_kernel_size=gate_supervision_pool_kernel,
                        temperature=gate_supervision_temperature,
                    )

                    gate_loss, _gate_scalar = compute_gate_supervision_loss(
                        gate=aux["mgu_gate"],
                        gate_target=gate_target,
                        loss_type=gate_supervision_loss_type,
                    )

                    voxel_weight = compute_disagreement_weight(
                        mri_logit=aux["mri_logit"].detach(),
                        eeg_logit=aux["eeg_logit"].detach(),
                        disagreement_weight=disagreement_loss_weight,
                        detach=True,
                    )

                    seg_loss, weighted_bce, weighted_dice = weighted_bce_dice_loss_fg(
                        fg_logit=fused_fg_logit,
                        target=target,
                        weight=voxel_weight,
                        bce_weight=ce_weight,
                        dice_weight=dice_weight,
                    )
                else:
                    ce = criterion(fused_logits, target.long())
                    dl = dice_loss(fused_logits, target)
                    seg_loss = ce_weight * ce + dice_weight * dl

                loss = (
                    seg_loss
                    + float(gate_supervision_weight) * gate_loss
                    + float(correction_loss_weight) * correction_loss
                )

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        b = mri.shape[0]
        total += b

        totals["loss"] += float(loss.detach().cpu()) * b
        totals["seg_loss"] += float(seg_loss.detach().cpu()) * b
        totals["corr_loss"] += float(correction_loss.detach().cpu()) * b
        totals["gate_loss"] += float(gate_loss.detach().cpu()) * b
        totals["weighted_bce"] += float(weighted_bce.detach().cpu()) * b
        totals["weighted_dice"] += float(weighted_dice.detach().cpu()) * b

        if use_reliability_supervision and gate_target is not None and voxel_weight is not None:
            totals["gate_target_mean"] += float(gate_target.detach().mean().cpu()) * b
            totals["gate_target_std"] += float(gate_target.detach().std(unbiased=False).cpu()) * b
            totals["disagreement_weight_mean"] += float(voxel_weight.detach().mean().cpu()) * b

        dice_fused = _dice_from_logits(fused_logits.detach(), target.detach())
        prec_fused, rec_fused = _precision_recall_from_logits(fused_logits.detach(), target.detach())
        dice_mri = _dice_from_logits(mri_logits.detach(), target.detach())
        prec_mri, rec_mri = _precision_recall_from_logits(mri_logits.detach(), target.detach())
        dice_eeg = _dice_from_logits(eeg_logits.detach(), target.detach())
        prec_eeg, rec_eeg = _precision_recall_from_logits(eeg_logits.detach(), target.detach())

        totals["dice_fused"] += float(dice_fused.detach().cpu()) * b
        totals["precision_fused"] += float(prec_fused.detach().cpu()) * b
        totals["recall_fused"] += float(rec_fused.detach().cpu()) * b
        totals["dice_mri"] += float(dice_mri.detach().cpu()) * b
        totals["precision_mri"] += float(prec_mri.detach().cpu()) * b
        totals["recall_mri"] += float(rec_mri.detach().cpu()) * b
        totals["dice_eeg"] += float(dice_eeg.detach().cpu()) * b
        totals["precision_eeg"] += float(prec_eeg.detach().cpu()) * b
        totals["recall_eeg"] += float(rec_eeg.detach().cpu()) * b

        mgu_stats_list.append(_collect_mgu_stats(aux=aux, fused_fg=fused_fg_logit))

    denom = max(total, 1)
    result = {
        "loss": totals["loss"] / denom,
        "seg_loss": totals["seg_loss"] / denom,
        "correction_loss": totals["corr_loss"] / denom,
        "gate_loss": totals["gate_loss"] / denom,
        "weighted_bce": totals["weighted_bce"] / denom,
        "weighted_dice": totals["weighted_dice"] / denom,
        "gate_target_mean": totals["gate_target_mean"] / denom,
        "gate_target_std": totals["gate_target_std"] / denom,
        "disagreement_weight_mean": totals["disagreement_weight_mean"] / denom,
        "dice_fused": totals["dice_fused"] / denom,
        "precision_fused": totals["precision_fused"] / denom,
        "recall_fused": totals["recall_fused"] / denom,
        "dice_mri": totals["dice_mri"] / denom,
        "precision_mri": totals["precision_mri"] / denom,
        "recall_mri": totals["recall_mri"] / denom,
        "dice_eeg": totals["dice_eeg"] / denom,
        "precision_eeg": totals["precision_eeg"] / denom,
        "recall_eeg": totals["recall_eeg"] / denom,
        "mgu_stats": _mean_of_dict(mgu_stats_list),
    }
    return result


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_val_loss,
    best_smoothed_val_loss,
    args,
    scheduler=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": float(best_val_loss),
            "best_smoothed_val_loss": float(best_smoothed_val_loss),
            "args": vars(args),
        },
        path,
    )


def save_mgu_only(path, model, args):
    model_obj = _unwrap_model(model)
    torch.save(
        {
            "mgu_state_dict": model_obj.mgu.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ids, val_ids, test_ids = _resolve_split_ids(args)
    train_cases = _build_cases(train_ids, args.mri_data_root)
    val_cases = _build_cases(val_ids, args.mri_data_root)
    test_cases = []
    if args.infer_test_set and len(test_ids) > 0:
        test_cases = [
            {"id": sid, "npy": os.path.join(args.mri_data_root, sid, f"{sid}_preproc.npz")}
            for sid in test_ids
            if os.path.exists(os.path.join(args.mri_data_root, sid, f"{sid}_preproc.npz"))
        ]
        if len(test_cases) == 0:
            warnings.warn("infer_test_set enabled but no valid test MRI npz files were found.")

    patch_size = tuple(args.patch_size)
    train_aug_params = {
        "p_mri_heavy_noise": args.p_mri_heavy_noise,
    }

    train_dataset = MultimodalMRIEEGPatchDataset(
        cases=train_cases,
        eeg_root=args.eeg_npz_root,
        force_load_into_memory=args.force_dataset_in_memory,
        patch_size=patch_size,
        enable_patch_sampling=True,
        patch_center_mode="random",
        enable_augmentation=True,
        augmentation_params=train_aug_params,
        enable_eeg_augmentation=True,
        disable_lr_flip=args.disable_lr_flip,
        disable_strong_spatial_aug=args.disable_strong_spatial_aug,
        eeg_training=True,
        eeg_max_offset=args.eeg_max_offset,
        eeg_training_drop_ratio=args.eeg_training_drop_ratio,
    )
    val_dataset = MultimodalMRIEEGPatchDataset(
        cases=val_cases,
        eeg_root=args.eeg_npz_root,
        force_load_into_memory=args.force_dataset_in_memory,
        patch_size=patch_size,
        enable_patch_sampling=True,
        patch_center_mode="random",
        enable_augmentation=False,
        disable_lr_flip=True,
        disable_strong_spatial_aug=True,
        eeg_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=multimodal_mri_eeg_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=multimodal_mri_eeg_collate,
    )

    mri_model = ResEncUNet_3D(input_channels=2, num_classes=2).to(device)
    _load_mri_checkpoint_strict(mri_model, args.mri_unet_checkpoint, device=device)

    eeg_model = _build_eeg_model_from_checkpoint(args.eeg_checkpoint, device=device)

    mgu = MGU3D(
        in_channels_per_modality=1,
        hidden_channels=args.mgu_hidden_channels,
        kernel_size=args.mgu_kernel_size,
        use_residual_mri=args.mgu_residual,
        init_residual_alpha=args.mgu_init_alpha,
        gate_bias_init=args.mgu_gate_bias_init,
    ).to(device)

    model = MGUOutputFusionModel(mri_model=mri_model, eeg_model=eeg_model, mgu=mgu).to(device)

    trainability = _set_trainability(model, args)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.log_root, exist_ok=True)
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"mgu_output_fusion_fold{args.fold}_{datestr}"
    log_dir = os.path.join(args.log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    if args.output_dir and os.path.normpath(args.output_dir) != os.path.normpath(log_dir):
        warnings.warn(
            f"--output_dir ({args.output_dir}) differs from auto-computed log_dir ({log_dir}). "
            "Checkpoints/logs will go to log_dir."
        )

    run_fingerprint_payload = emit_run_fingerprint(
        script_name="mgu_training",
        train_config=vars(args),
        model_kwargs={
            "mgu_hidden_channels": args.mgu_hidden_channels,
            "mgu_kernel_size": args.mgu_kernel_size,
            "mgu_residual": args.mgu_residual,
            "mgu_init_alpha": args.mgu_init_alpha,
            "mgu_gate_bias_init": args.mgu_gate_bias_init,
            "correction_loss_weight": args.correction_loss_weight,
        },
        effective_model_config={
            "model_class": model.__class__.__name__,
            "mri_model_class": mri_model.__class__.__name__,
            "eeg_model_class": eeg_model.__class__.__name__,
            "train_mri": trainability["train_mri"],
            "train_eeg": trainability["train_eeg"],
            "train_mgu": trainability["train_mgu"],
            "use_reliability_supervision": args.use_reliability_supervision,
            "gate_supervision_weight": args.gate_supervision_weight,
            "gate_supervision_pool_kernel": args.gate_supervision_pool_kernel,
            "gate_supervision_temperature": args.gate_supervision_temperature,
            "gate_supervision_loss_type": args.gate_supervision_loss_type,
            "disagreement_loss_weight": args.disagreement_loss_weight,
        },
        extra={
            "device": str(device),
            "mri_checkpoint": args.mri_unet_checkpoint,
            "eeg_checkpoint": args.eeg_checkpoint,
            "fold": args.fold,
            "train_subject_ids": train_ids,
            "val_subject_ids": val_ids,
            "test_subject_ids": test_ids,
            "seed": args.seed,
        },
    )

    run_fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(run_fingerprint_path, "w") as f:
        json.dump(run_fingerprint_payload, f, indent=2)
    print(f"Saved run fingerprint to: {run_fingerprint_path}")

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config/args", json.dumps(vars(args), indent=2))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    writer.add_scalar("params/trainable_total", _count_params(trainable_params), 0)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_last = os.path.join(ckpt_dir, "checkpoint_last.pt")
    ckpt_best_raw = os.path.join(ckpt_dir, "checkpoint_best_raw_val_loss.pt")
    ckpt_best_smoothed = os.path.join(ckpt_dir, "checkpoint_best_smoothed_val_loss.pt")
    mgu_only_path = os.path.join(ckpt_dir, "mgu_only.pt")

    start_epoch = 0
    best_val_loss = float("inf")
    best_smoothed_val_loss = float("inf")
    es = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        warmup=args.early_stopping_warmup,
        smoothing_window=args.early_stopping_smoothing_window,
        enabled=args.early_stopping,
    )

    if args.resume is not None and os.path.exists(args.resume):
        restored = torch.load(args.resume, map_location=device)
        model.load_state_dict(restored["model_state_dict"], strict=False)
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        if restored.get("scaler_state_dict") is not None:
            scaler.load_state_dict(restored["scaler_state_dict"])
        start_epoch = int(restored.get("epoch", 0)) + 1
        best_val_loss = float(restored.get("best_val_loss", best_val_loss))
        best_smoothed_val_loss = float(restored.get("best_smoothed_val_loss", best_smoothed_val_loss))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=True,
            amp_enabled=True,
            ce_weight=args.ce_loss_weight,
            dice_weight=args.dice_loss_weight,
            correction_loss_weight=args.correction_loss_weight,
            gate_supervision_weight=args.gate_supervision_weight,
            gate_supervision_pool_kernel=args.gate_supervision_pool_kernel,
            gate_supervision_temperature=args.gate_supervision_temperature,
            gate_supervision_loss_type=args.gate_supervision_loss_type,
            disagreement_loss_weight=args.disagreement_loss_weight,
            use_reliability_supervision=args.use_reliability_supervision,
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=False,
            amp_enabled=True,
            ce_weight=args.ce_loss_weight,
            dice_weight=args.dice_loss_weight,
            correction_loss_weight=args.correction_loss_weight,
            gate_supervision_weight=args.gate_supervision_weight,
            gate_supervision_pool_kernel=args.gate_supervision_pool_kernel,
            gate_supervision_temperature=args.gate_supervision_temperature,
            gate_supervision_loss_type=args.gate_supervision_loss_type,
            disagreement_loss_weight=args.disagreement_loss_weight,
            use_reliability_supervision=args.use_reliability_supervision,
        )

        es_info = es.update(epoch=epoch, val_loss=float(val_metrics["loss"]))
        smoothed_val_loss = float(es_info["smoothed_val_loss"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"train_seg={train_metrics['seg_loss']:.4f}, val_seg={val_metrics['seg_loss']:.4f}, "
            f"train_corr={train_metrics['correction_loss']:.4f}, val_corr={val_metrics['correction_loss']:.4f}, "
            f"train_gate={train_metrics['gate_loss']:.4f}, val_gate={val_metrics['gate_loss']:.4f}, "
            f"val_gate_target_mean={val_metrics['gate_target_mean']:.3f}, "
            f"val_dice_fused={val_metrics['dice_fused']:.4f}, val_dice_mri={val_metrics['dice_mri']:.4f}, "
            f"val_dice_eeg={val_metrics['dice_eeg']:.4f}, time={elapsed:.1f}s"
        )

        writer.add_scalars("loss/total", {"Train": train_metrics["loss"], "Val": val_metrics["loss"]}, epoch)
        writer.add_scalars("loss/seg", {"Train": train_metrics["seg_loss"], "Val": val_metrics["seg_loss"]}, epoch)
        writer.add_scalars("loss/correction", {"Train": train_metrics["correction_loss"], "Val": val_metrics["correction_loss"]}, epoch)
        writer.add_scalars("loss/gate_supervision", {"Train": train_metrics["gate_loss"], "Val": val_metrics["gate_loss"]}, epoch)
        writer.add_scalars("loss/weighted_bce", {"Train": train_metrics["weighted_bce"], "Val": val_metrics["weighted_bce"]}, epoch)
        writer.add_scalars("loss/weighted_dice", {"Train": train_metrics["weighted_dice"], "Val": val_metrics["weighted_dice"]}, epoch)

        writer.add_scalars("reliability/gate_target_mean", {"Train": train_metrics["gate_target_mean"], "Val": val_metrics["gate_target_mean"]}, epoch)
        writer.add_scalars("reliability/gate_target_std", {"Train": train_metrics["gate_target_std"], "Val": val_metrics["gate_target_std"]}, epoch)
        writer.add_scalars("reliability/disagreement_weight_mean", {"Train": train_metrics["disagreement_weight_mean"], "Val": val_metrics["disagreement_weight_mean"]}, epoch)

        writer.add_scalars("val/dice_fused", {"Val": val_metrics["dice_fused"]}, epoch)
        writer.add_scalars("val/dice_mri", {"Val": val_metrics["dice_mri"]}, epoch)
        writer.add_scalars("val/dice_eeg", {"Val": val_metrics["dice_eeg"]}, epoch)

        writer.add_scalars("val/precision_fused", {"Val": val_metrics["precision_fused"]}, epoch)
        writer.add_scalars("val/precision_mri", {"Val": val_metrics["precision_mri"]}, epoch)
        writer.add_scalars("val/precision_eeg", {"Val": val_metrics["precision_eeg"]}, epoch)

        writer.add_scalars("val/recall_fused", {"Val": val_metrics["recall_fused"]}, epoch)
        writer.add_scalars("val/recall_mri", {"Val": val_metrics["recall_mri"]}, epoch)
        writer.add_scalars("val/recall_eeg", {"Val": val_metrics["recall_eeg"]}, epoch)

        for tag, value in train_metrics["mgu_stats"].items():
            writer.add_scalar(f"{tag}/train", float(value), epoch)
        for tag, value in val_metrics["mgu_stats"].items():
            writer.add_scalar(f"{tag}/val", float(value), epoch)

        writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
        writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("val/loss_raw", float(val_metrics["loss"]), epoch)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss, epoch)

        save_checkpoint(
            path=ckpt_last,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=min(best_val_loss, val_metrics["loss"]),
            best_smoothed_val_loss=min(best_smoothed_val_loss, smoothed_val_loss),
            args=args,
            scheduler=None,
        )
        save_mgu_only(path=mgu_only_path, model=model, args=args)

        if es_info["raw_improved"]:
            best_val_loss = float(es_info["best_raw_val_loss"])
            save_checkpoint(
                path=ckpt_best_raw,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_smoothed_val_loss=min(best_smoothed_val_loss, smoothed_val_loss),
                args=args,
                scheduler=None,
            )
            print(f"New best raw validation loss: {best_val_loss:.6f}")

        if es_info["improved"]:
            best_smoothed_val_loss = float(es_info["best_smoothed_val_loss"])
            save_checkpoint(
                path=ckpt_best_smoothed,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=min(best_val_loss, val_metrics["loss"]),
                best_smoothed_val_loss=best_smoothed_val_loss,
                args=args,
                scheduler=None,
            )
            save_mgu_only(path=mgu_only_path, model=model, args=args)
            print(f"New best smoothed validation loss: {best_smoothed_val_loss:.6f}")

        if es_info["should_stop"]:
            best_epoch_1based = (es_info["best_epoch"] + 1) if es_info["best_epoch"] is not None else None
            print(
                "Early stopping triggered "
                f"at epoch {epoch + 1}. Best smoothed val loss: {es_info['best_smoothed_val_loss']:.6f} "
                f"(epoch {best_epoch_1based})."
            )
            break

    print("Training finished.")
    print("Best raw-loss checkpoint:", ckpt_best_raw)
    print("Best smoothed-loss checkpoint:", ckpt_best_smoothed)
    print("MGU-only checkpoint:", mgu_only_path)

    if args.restore_best_checkpoint and os.path.exists(ckpt_best_smoothed):
        restored = torch.load(ckpt_best_smoothed, map_location=device)
        model.load_state_dict(restored["model_state_dict"], strict=False)
        print(f"Restored best smoothed-loss checkpoint from: {ckpt_best_smoothed}")

    if args.run_full_volume_inference:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required for NIfTI export during inference.") from exc

        print("Running full-volume sliding-window inference on train/val splits...")
        split_to_cases = {
            "train": train_cases,
            "val": val_cases,
        }
        if args.infer_test_set and len(test_cases) > 0:
            split_to_cases["test"] = test_cases

        patch_size = tuple(int(v) for v in args.patch_size)
        stride = tuple(max(1, int(round(v * args.inference_stride_factor))) for v in patch_size)

        for split_name, cases in split_to_cases.items():
            pred_dir = os.path.join(log_dir, "pred_niftis", split_name)
            os.makedirs(pred_dir, exist_ok=True)

            for case in cases:
                sid = case["id"]
                try:
                    npz = np.load(case["npy"], allow_pickle=True)
                    img = np.asarray(npz["image"], dtype=np.float32)
                    affine = np.asarray(npz["affine"], dtype=np.float32)
                    npz.close()

                    eeg_input = _load_inference_eeg_input(
                        eeg_root=args.eeg_npz_root,
                        subject_id=sid,
                        window_size=args.inference_eeg_window_size,
                        max_spikes=args.inference_max_spikes_per_bag,
                    )

                    pred = sliding_window_inference_mgu(
                        model=model,
                        mri_image=img,
                        eeg_input=eeg_input,
                        patch_size=patch_size,
                        stride=stride,
                        device=device,
                        amp_enabled=True,
                        use_gaussian_weighting=args.use_gaussian_weighting,
                        gaussian_sigma_scale=args.gaussian_sigma_scale,
                        gaussian_value_scaling_factor=args.gaussian_value_scaling_factor,
                        accumulate_on_cpu=args.accumulate_on_cpu,
                    )

                    out_path = os.path.join(pred_dir, f"{sid}_pred_prob.nii.gz")
                    nib.save(nib.Nifti1Image(pred, affine), out_path)
                except Exception as e:
                    print(f"Warning: skipped {sid} due to error during {split_name} inference: {e}")

            print(f"Saved full-volume inference NIfTIs to: {pred_dir}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train residual MGU output-fusion model")

    parser.add_argument("--mri_unet_checkpoint", "--mri_checkpoint", dest="mri_unet_checkpoint", type=str, required=True)
    parser.add_argument("--eeg_checkpoint", type=str, required=True)
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mri_data_root", type=str, required=True)
    parser.add_argument("--eeg_npz_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_root", type=str, default="./runs")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ce_loss_weight", type=float, default=0.7)
    parser.add_argument("--dice_loss_weight", type=float, default=0.3)
    parser.add_argument("--correction_loss_weight", type=float, default=0.01)
    parser.add_argument("--use_reliability_supervision", action="store_true", default=False)
    parser.add_argument("--gate_supervision_weight", type=float, default=0.1)
    parser.add_argument("--gate_supervision_pool_kernel", type=int, default=15)
    parser.add_argument("--gate_supervision_temperature", type=float, default=0.05)
    parser.add_argument("--gate_supervision_loss_type", type=str, default="bce", choices=["bce", "mse"])
    parser.add_argument("--disagreement_loss_weight", type=float, default=2.0)

    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])

    parser.add_argument("--mgu_hidden_channels", type=int, default=8)
    parser.add_argument("--mgu_kernel_size", type=int, default=3)
    parser.add_argument("--mgu_residual", action="store_true", default=True)
    parser.add_argument("--no_mgu_residual", action="store_false", dest="mgu_residual")
    parser.add_argument("--mgu_init_alpha", type=float, default=0.0)
    parser.add_argument("--mgu_gate_bias_init", type=float, default=1.0)

    parser.add_argument("--freeze_mri", action="store_true", default=True)
    parser.add_argument("--no_freeze_mri", action="store_false", dest="freeze_mri")
    parser.add_argument("--freeze_eeg", action="store_true", default=True)
    parser.add_argument("--no_freeze_eeg", action="store_false", dest="freeze_eeg")
    parser.add_argument("--train_mgu_only", action="store_true", default=True)
    parser.add_argument("--no_train_mgu_only", action="store_false", dest="train_mgu_only")
    parser.add_argument("--train_mri", action="store_true")
    parser.add_argument("--train_eeg", action="store_true")

    parser.add_argument("--disable_lr_flip", action="store_true")
    parser.add_argument("--disable_strong_spatial_aug", action="store_true")
    parser.add_argument("--p_mri_heavy_noise", type=float, default=0.0)
    parser.add_argument("--eeg_max_offset", type=int, default=0)
    parser.add_argument("--eeg_training_drop_ratio", type=float, default=0.0)
    parser.add_argument("--force_dataset_in_memory", action="store_true", default=True)
    parser.add_argument("--no_force_dataset_in_memory", action="store_false", dest="force_dataset_in_memory")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--infer_test_set", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stopping_warmup", type=int, default=10)
    parser.add_argument("--early_stopping_smoothing_window", type=int, default=5)
    parser.add_argument("--restore_best_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_restore_best_checkpoint", action="store_false", dest="restore_best_checkpoint")
    parser.add_argument("--run_full_volume_inference", action="store_true")
    parser.add_argument("--inference_stride_factor", type=float, default=0.25)
    parser.add_argument("--inference_eeg_window_size", type=int, default=128)
    parser.add_argument("--inference_max_spikes_per_bag", type=int, default=64)
    parser.add_argument("--use_gaussian_weighting", action="store_true", default=True)
    parser.add_argument("--no_gaussian_weighting", action="store_false", dest="use_gaussian_weighting")
    parser.add_argument("--gaussian_sigma_scale", type=float, default=1 / 8)
    parser.add_argument("--gaussian_value_scaling_factor", type=float, default=10.0)
    parser.add_argument("--accumulate_on_cpu", action="store_true", default=True)
    parser.add_argument("--no_accumulate_on_cpu", action="store_false", dest="accumulate_on_cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
