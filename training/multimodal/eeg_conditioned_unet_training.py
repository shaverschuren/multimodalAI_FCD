"""
Train MRI U-Net segmentation conditioned by online EEG MIL embeddings.

Default trainability:
- MRI encoder: frozen
- MRI decoder: trainable
- EEG model: frozen
- Fusion conditioner: trainable
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime

import nibabel as nib
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
    normalize_mri_checkpoint_state_dict,
)
from models.multimodal import EEGConditionedUNet
from util import EarlyStopping, emit_run_fingerprint

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:
    _scipy_gaussian_filter = None


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


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, smooth=1.0):
    if logits.shape[1] != 2:
        return torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()

    pred_fg = torch.softmax(logits, dim=1)[:, 1]
    tgt_fg = (targets.long() == 1).float()

    tp = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
    fp = (pred_fg * (1.0 - tgt_fg)).flatten(1).sum(dim=1)
    fn = ((1.0 - pred_fg) * tgt_fg).flatten(1).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def _dice_from_logits(logits, targets, smooth=1.0):
    if logits.shape[1] == 2:
        pred_fg = torch.softmax(logits, dim=1)[:, 1]
        tgt_fg = (targets.long() == 1).float()
        inter = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
        union = pred_fg.flatten(1).sum(dim=1) + tgt_fg.flatten(1).sum(dim=1)
        return ((2.0 * inter + smooth) / (union + smooth)).mean()
    return torch.zeros(1, device=logits.device)


def _precision_recall_from_logits(logits, targets, smooth=1.0):
    if logits.shape[1] == 2:
        pred_fg = (torch.argmax(logits, dim=1) == 1).float()
        tgt_fg = (targets.long() == 1).float()

        tp = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
        pred_pos = pred_fg.flatten(1).sum(dim=1)
        tgt_pos = tgt_fg.flatten(1).sum(dim=1)

        precision = (tp + smooth) / (pred_pos + smooth)
        recall = (tp + smooth) / (tgt_pos + smooth)
        return precision.mean(), recall.mean()

    zero = torch.zeros(1, device=logits.device)
    return zero, zero


def _build_cases(subject_ids, mri_data_root):
    cases = []
    for sid in subject_ids:
        npz_path = os.path.join(mri_data_root, sid, f"{sid}_preproc.npz")
        if os.path.exists(npz_path):
            cases.append({"id": sid, "npy": npz_path})
    if not cases:
        raise RuntimeError(f"No valid MRI cases found under {mri_data_root}")
    return cases


def _resolve_train_val_ids(args):
    train_ids, val_ids = load_split(args.splits_json, args.fold)
    train_ids = [str(v) for v in train_ids]
    val_ids = [str(v) for v in val_ids]
    print(f"Loaded fold {args.fold} from splits JSON.")

    if args.test_mode:
        train_ids = train_ids[: min(len(train_ids), 8)]
        val_ids = val_ids[: min(len(val_ids), 8)]
        print("Test mode active: truncated train/val subject lists to at most 8 each.")

    print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")
    return train_ids, val_ids


def _resolve_trainable(freeze_flag, train_flag, default_trainable):
    if train_flag:
        return True
    if freeze_flag:
        return False
    return bool(default_trainable)


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

    state_dict = normalize_mri_checkpoint_state_dict(state_dict, verbose=True)

    state_dict = expand_mismatched_conv3d_weights_to_match_model(
        state_dict=state_dict,
        model=mri_model.backbone,
        verbose=True,
    )
    missing, unexpected = mri_model.backbone.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(
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


def _count_params(params):
    return int(sum(p.numel() for p in params))


def _build_optimizer(model, args):
    groups = model.get_parameter_groups()
    optim_groups = []

    if groups["fusion"]:
        optim_groups.append({"params": groups["fusion"], "lr": args.fusion_lr, "name": "fusion"})
    if groups["decoder"]:
        optim_groups.append({"params": groups["decoder"], "lr": args.decoder_lr, "name": "decoder"})
    if groups["encoder"]:
        optim_groups.append({"params": groups["encoder"], "lr": args.encoder_lr, "name": "encoder"})
    if groups["eeg"]:
        optim_groups.append({"params": groups["eeg"], "lr": args.eeg_lr, "name": "eeg"})

    if not optim_groups:
        raise RuntimeError("No trainable parameters found after train/freeze settings.")

    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, groups


def _initialize_eeg_projector_if_needed(model, dataloader, device):
    """
    Materialize lazy EEG projector parameters before optimizer construction.

    EEGConditionedUNet may create `eeg_projector` on first forward when EEG embedding
    dim does not match `eeg_dim`. If optimizer is built earlier, projector params are
    missing from optimizer groups and won't be updated.
    """
    model_obj = _unwrap_model(model)
    if not hasattr(model_obj, "_forward_eeg") or not hasattr(model_obj, "_project_eeg_if_needed"):
        return

    if getattr(model_obj, "eeg_model", None) is None:
        return

    if getattr(model_obj, "eeg_projector", None) is not None:
        return

    if dataloader is None or len(dataloader) == 0:
        return

    try:
        probe_batch = next(iter(dataloader))
    except StopIteration:
        return

    eeg_input = {
        "spikes": probe_batch["eeg_input"]["spikes"].to(device, non_blocking=True),
        "mask": probe_batch["eeg_input"]["mask"].to(device, non_blocking=True),
    }
    batch_size = int(eeg_input["spikes"].shape[0])

    with torch.no_grad():
        eeg_embedding = model_obj._forward_eeg(eeg_input=eeg_input, batch_size=batch_size, use_no_grad=True)
        _ = model_obj._project_eeg_if_needed(eeg_embedding)


def _set_eeg_trunk_trainable(model, trainable=True):
    """Helper to enable training of EEG trunk layers for two-step training. Unused for now but kept just in case."""
    model_obj = _unwrap_model(model)
    eeg_model = getattr(model_obj, "eeg_model", None)
    if eeg_model is None:
        raise RuntimeError("EEG trunk configuration requires model.eeg_model, but none was found.")

    # Start from a fully frozen EEG model, then re-enable only trunk if requested.
    for p in eeg_model.parameters():
        p.requires_grad = False

    if not trainable:
        return 0

    # SpikeMILModel exposes its post-pooling trunk as `trunk`; require this explicitly.
    if not hasattr(eeg_model, "trunk") or not isinstance(eeg_model.trunk, nn.Module):
        raise RuntimeError(
            "Two-step training expects eeg_model.trunk (SpikeMILModel trunk), but it was not found."
        )

    n = 0
    for p in eeg_model.trunk.parameters():
        p.requires_grad = True
        n += p.numel()
    return n


def _configure_trainability_phase(model, args, phase):
    model_obj = _unwrap_model(model)
    if args.two_step_training:
        train_mri_encoder = False
        train_mri_decoder = bool(phase >= 2)
        train_eeg = False
        model_obj.set_trainability(
            train_mri_encoder=train_mri_encoder,
            train_mri_decoder=train_mri_decoder,
            train_eeg=train_eeg,
            train_fusion=True,
            train_segmentation_head=getattr(args, "train_segmentation_head", False),
        )
        eeg_trunk_params = 0
        return {
            "train_mri_encoder": train_mri_encoder,
            "train_mri_decoder": train_mri_decoder,
            "train_eeg": train_eeg,
            "eeg_trunk_params": eeg_trunk_params,
        }

    freeze_eeg_flag = getattr(args, "freeze_eeg_model", getattr(args, "freeze_eeg", False))
    train_eeg_flag = getattr(args, "train_eeg", False)
    train_segmentation_head = getattr(args, "train_segmentation_head", False)
    default_decoder_trainable = bool(args.fusion_mode != "bottleneck_film_skip_gate" and not train_segmentation_head)
    default_eeg_trainable = False

    train_mri_encoder = _resolve_trainable(args.freeze_mri_encoder, args.train_mri_encoder, default_trainable=False)
    train_mri_decoder = _resolve_trainable(args.freeze_mri_decoder, args.train_mri_decoder, default_trainable=default_decoder_trainable)
    train_eeg = _resolve_trainable(freeze_eeg_flag, train_eeg_flag, default_trainable=default_eeg_trainable)
    model_obj.set_trainability(
        train_mri_encoder=train_mri_encoder,
        train_mri_decoder=train_mri_decoder,
        train_eeg=train_eeg,
        train_fusion=True,
        train_segmentation_head=train_segmentation_head,
    )
    return {
        "train_mri_encoder": train_mri_encoder,
        "train_mri_decoder": train_mri_decoder,
        "train_eeg": train_eeg,
        "eeg_trunk_params": 0,
    }


def _print_first_batch_debug(batch, aux, logits, model):
    eeg_input = batch["eeg_input"]
    eeg_shapes = {k: tuple(v.shape) for k, v in eeg_input.items() if torch.is_tensor(v)}
    model_obj = _unwrap_model(model)
    print("[debug] mri_patch.shape:", tuple(batch["mri"].shape))
    print("[debug] target.shape:", tuple(batch["target"].shape))
    print("[debug] patch_center.shape:", tuple(batch["patch_center"].shape))
    print("[debug] eeg_input shapes:", eeg_shapes)
    print("[debug] eeg_embedding.shape:", tuple(aux["eeg_embedding"].shape))
    print("[debug] mri_bottleneck.shape:", tuple(aux["mri_bottleneck"].shape))
    print("[debug] output.shape:", tuple(logits.shape))
    print("[debug] alpha.item():", float(model_obj.alpha.detach().cpu().item()))


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_film_stats(model):
    model_obj = _unwrap_model(model)
    for attr in ("get_fusion_stats", "get_film_stats"):
        if hasattr(model_obj, attr):
            stats = getattr(model_obj, attr)()
            if isinstance(stats, dict):
                return stats
    return {}


def _log_scalar_dict(writer, scalar_dict, step):
    for tag, value in scalar_dict.items():
        if value is None:
            continue
        writer.add_scalar(tag, float(value), step)


def _prefix_scalar_dict(scalar_dict, prefix):
    return {f"{prefix}{k}": v for k, v in scalar_dict.items()}


def _accumulate_scalar_dict(sum_dict, count_dict, scalar_dict):
    for tag, value in scalar_dict.items():
        if value is None:
            continue
        value_f = float(value)
        if not np.isfinite(value_f):
            continue
        sum_dict[tag] = sum_dict.get(tag, 0.0) + value_f
        count_dict[tag] = count_dict.get(tag, 0) + 1


def _finalize_scalar_dict_mean(sum_dict, count_dict):
    out = {}
    for tag, total in sum_dict.items():
        n = int(count_dict.get(tag, 0))
        if n > 0:
            out[tag] = float(total / n)
    return out


def _foreground_probability(logits):
    probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
    if probs.shape[1] > 1:
        return probs[:, 1]
    return probs[:, 0]


def _compute_eeg_sensitivity_metrics(full_logits, zero_logits, target):
    full_prob = _foreground_probability(full_logits)
    zero_prob = _foreground_probability(zero_logits)
    delta_signed = full_prob - zero_prob
    delta = delta_signed.abs()

    eps = torch.finfo(delta.dtype).eps
    metrics = {
        "prediction/eeg_delta_mean": float(delta.mean().detach().cpu()),
        "prediction/eeg_delta_signed_mean": float(delta_signed.mean().detach().cpu()),
    }

    gt_mask = target > 0.5
    if gt_mask.any():
        metrics["prediction/eeg_delta_in_gt"] = float(delta[gt_mask].mean().detach().cpu())
        metrics["prediction/eeg_delta_signed_in_gt"] = float(delta_signed[gt_mask].mean().detach().cpu())
    else:
        metrics["prediction/eeg_delta_in_gt"] = float("nan")
        metrics["prediction/eeg_delta_signed_in_gt"] = float("nan")

    outside_mask = ~gt_mask
    if outside_mask.any():
        metrics["prediction/eeg_delta_outside_gt"] = float(delta[outside_mask].mean().detach().cpu())
        metrics["prediction/eeg_delta_signed_outside_gt"] = float(delta_signed[outside_mask].mean().detach().cpu())
    else:
        metrics["prediction/eeg_delta_outside_gt"] = float("nan")
        metrics["prediction/eeg_delta_signed_outside_gt"] = float("nan")

    zero_fg = zero_prob > 0.5
    if zero_fg.any():
        metrics["prediction/eeg_delta_in_mri_candidates"] = float(delta[zero_fg].mean().detach().cpu())
        metrics["prediction/eeg_delta_signed_in_mri_candidates"] = float(delta_signed[zero_fg].mean().detach().cpu())
    else:
        metrics["prediction/eeg_delta_in_mri_candidates"] = float("nan")
        metrics["prediction/eeg_delta_signed_in_mri_candidates"] = float("nan")

    return metrics


def run_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    train,
    amp_enabled,
    debug_shapes=False,
    seg_loss_mode="dice_bce",
    ce_weight=1.0,
    dice_weight=1.0,
    tversky_weight=0.2,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    skip_gate_reg_weight=0.0,
    writer=None,
    enable_film_logging=False,
    log_film_stats_every_n_steps=0,
    log_step_offset=0,
    log_eeg_sensitivity=False,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_ce_loss = torch.tensor(0.0, device=device)
    total_dice_loss = torch.tensor(0.0, device=device)
    total_tversky_loss = torch.tensor(0.0, device=device)
    total_dice = torch.tensor(0.0, device=device)
    total_precision = torch.tensor(0.0, device=device)
    total_recall = torch.tensor(0.0, device=device)
    total_emb_norm_mean = torch.tensor(0.0, device=device)
    total_emb_norm_std = torch.tensor(0.0, device=device)
    total = 0
    n_steps = 0
    val_film_stats_logged = False
    film_sum = {}
    film_count = {}
    sens_sum = {}
    sens_count = {}

    for step, batch in enumerate(dataloader):
        global_step = int(log_step_offset + step)
        mri = batch["mri"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        patch_bbox = batch["patch_bbox"].to(device, non_blocking=True) if "patch_bbox" in batch else None
        volume_shape = batch["volume_shape"].to(device, non_blocking=True) if "volume_shape" in batch else None
        patch_center = batch["patch_center"].to(device, non_blocking=True)
        eeg_input = {
            "spikes": batch["eeg_input"]["spikes"].to(device, non_blocking=True),
            "mask": batch["eeg_input"]["mask"].to(device, non_blocking=True),
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                logits, aux = model(
                    mri_patch=mri,
                    eeg_input=eeg_input,
                    patch_center=patch_center,
                    patch_bbox=patch_bbox,
                    volume_shape=volume_shape,
                    return_aux=True,
                )
                ce = criterion(logits, target.long())
                dl = dice_loss(logits, target)
                tvl = tversky_loss(
                    logits,
                    target,
                    alpha=float(tversky_alpha),
                    beta=float(tversky_beta),
                )
                gate_reg = getattr(_unwrap_model(model), "latest_skip_gate_reg_loss", torch.tensor(0.0, device=device))
                dice_bce_loss = ce_weight * ce + dice_weight * dl
                if seg_loss_mode == "dice_bce_plus_tversky":
                    loss = dice_bce_loss + float(tversky_weight) * tvl + float(skip_gate_reg_weight) * gate_reg
                elif seg_loss_mode == "dice_bce":
                    loss = dice_bce_loss + float(skip_gate_reg_weight) * gate_reg
                else:
                    raise ValueError(f"Unknown seg_loss_mode: {seg_loss_mode}")

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        film_stats_this_step = _get_film_stats(model) if enable_film_logging else {}
        sensitivity_stats_this_step = None
        if enable_film_logging and log_eeg_sensitivity and step == 0:
            with torch.no_grad():
                zero_eeg_embedding = torch.zeros(
                    (mri.shape[0], _unwrap_model(model).eeg_dim),
                    device=mri.device,
                    dtype=mri.dtype,
                )
                with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                    zero_logits = model(
                        mri_patch=mri,
                        patch_center=patch_center,
                        patch_bbox=patch_bbox,
                        volume_shape=volume_shape,
                        force_zero_eeg=True,
                    )
                    if isinstance(zero_logits, (list, tuple)):
                        zero_logits = zero_logits[0]
            sensitivity_stats_this_step = _compute_eeg_sensitivity_metrics(
                logits.detach(),
                zero_logits.detach(),
                target.detach(),
            )

        if enable_film_logging:
            _accumulate_scalar_dict(film_sum, film_count, film_stats_this_step)
            if sensitivity_stats_this_step is not None:
                _accumulate_scalar_dict(sens_sum, sens_count, sensitivity_stats_this_step)

        if writer is not None and enable_film_logging:
            should_log_film = False
            if train and log_film_stats_every_n_steps and log_film_stats_every_n_steps > 0:
                should_log_film = (step % int(log_film_stats_every_n_steps)) == 0
            elif (not train) and not val_film_stats_logged:
                should_log_film = True

            if should_log_film:
                split_prefix = "film_step/train/" if train else "film_step/val/"
                split_film_stats = _prefix_scalar_dict(film_stats_this_step, split_prefix)
                _log_scalar_dict(writer, split_film_stats, global_step)
                if not train:
                    val_film_stats_logged = True

        if debug_shapes and step == 0:
            _print_first_batch_debug(batch, aux, logits, model)

        b = mri.shape[0]
        total += b
        n_steps += 1
        total_loss += loss.detach() * b
        total_ce_loss += ce.detach() * b
        total_dice_loss += dl.detach() * b
        total_tversky_loss += tvl.detach() * b
        total_dice += _dice_from_logits(logits.detach(), target.detach()) * b
        precision, recall = _precision_recall_from_logits(logits.detach(), target.detach())
        total_precision += precision.detach() * b
        total_recall += recall.detach() * b
        total_emb_norm_mean += aux["eeg_embedding_norm_mean"].detach()
        total_emb_norm_std += aux["eeg_embedding_norm_std"].detach()

    result = {
        "loss": (total_loss / max(total, 1)).item(),
        "ce_loss": (total_ce_loss / max(total, 1)).item(),
        "dice_loss": (total_dice_loss / max(total, 1)).item(),
        "tversky_loss": (total_tversky_loss / max(total, 1)).item(),
        "dice": (total_dice / max(total, 1)).item(),
        "precision": (total_precision / max(total, 1)).item(),
        "recall": (total_recall / max(total, 1)).item(),
        "emb_norm_mean": (total_emb_norm_mean / max(n_steps, 1)).item(),
        "emb_norm_std": (total_emb_norm_std / max(n_steps, 1)).item(),
    }
    if enable_film_logging:
        fusion_epoch_avg = _finalize_scalar_dict_mean(film_sum, film_count)
        result["film_epoch_avg"] = fusion_epoch_avg
        result["fusion_epoch_avg"] = fusion_epoch_avg
        result["prediction_epoch_avg"] = _finalize_scalar_dict_mean(sens_sum, sens_count)
    return result


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    training_phase,
    best_val_loss,
    best_smoothed_val_loss,
    args,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "training_phase": int(training_phase),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": float(best_val_loss),
            "best_smoothed_val_loss": float(best_smoothed_val_loss),
            "args": vars(args),
        },
        path,
    )


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

    if _scipy_gaussian_filter is not None:
        gaussian = np.zeros(patch_size, dtype=np.float32)
        gaussian[center] = 1.0
        sigmas = [max(float(v) * float(sigma_scale), 1e-6) for v in patch_size]
        gaussian = _scipy_gaussian_filter(gaussian, sigma=sigmas, mode="constant", cval=0.0)
    else:
        # Torch-only fallback equivalent to filtering a centered impulse with a Gaussian kernel.
        coords = [torch.arange(v, dtype=torch.float32) for v in patch_size]
        zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        sigmas = [max(float(v) * float(sigma_scale), 1e-6) for v in patch_size]
        gaussian = torch.exp(
            -(
                ((zz - float(center[0])) ** 2) / (2.0 * sigmas[0] ** 2)
                + ((yy - float(center[1])) ** 2) / (2.0 * sigmas[1] ** 2)
                + ((xx - float(center[2])) ** 2) / (2.0 * sigmas[2] ** 2)
            )
        ).numpy()

    max_val = float(np.max(gaussian))
    if max_val <= 0.0:
        raise RuntimeError("Gaussian importance map is all zeros.")

    gaussian = gaussian / max_val
    gaussian = gaussian * float(value_scaling_factor)

    nonzero = gaussian[gaussian > 0]
    if nonzero.size == 0:
        raise RuntimeError("Gaussian importance map contains no positive values.")
    min_nonzero = float(nonzero.min())
    gaussian[gaussian == 0] = min_nonzero

    return torch.from_numpy(gaussian.astype(np.float32)).to(dtype=dtype).unsqueeze(0)


def _extract_logits(output):
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


def sliding_window_inference_conditioned(
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
        out_channels = None
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
                        output = model(
                            mri_patch=patch_t,
                            eeg_input=eeg_batch,
                            patch_center=center_t,
                            patch_bbox=patch_bbox_t,
                            volume_shape=volume_shape_t,
                        )
                        patch_logits = _extract_logits(output)

                    if patch_logits.ndim != 5 or patch_logits.shape[0] != 1:
                        raise RuntimeError(
                            f"Expected patch logits shape (1, C_out, pd, ph, pw), got {tuple(patch_logits.shape)}"
                        )

                    patch_logits = patch_logits[0].detach().to(accumulator_device, dtype=torch.float32)

                    if out_channels is None:
                        out_channels = int(patch_logits.shape[0])
                        logit_accum = torch.zeros(
                            (out_channels, d_pad, h_pad, w_pad), dtype=torch.float32, device=accumulator_device
                        )

                    if tuple(patch_logits.shape[1:]) != (pd, ph, pw):
                        raise RuntimeError(
                            f"Unexpected patch logit spatial shape {tuple(patch_logits.shape[1:])}; "
                            f"expected {(pd, ph, pw)}"
                        )

                    if use_gaussian_weighting:
                        weighted_logits = patch_logits * gaussian
                        patch_weight = gaussian[0]
                    else:
                        weighted_logits = patch_logits
                        patch_weight = torch.ones((pd, ph, pw), dtype=torch.float32, device=accumulator_device)

                    logit_accum[:, ds:de, hs:he, ws:we] += weighted_logits
                    weight_accum[ds:de, hs:he, ws:we] += patch_weight

    assert logit_accum is not None and out_channels is not None, "No inference tiles were processed."
    assert tuple(logit_accum.shape[1:]) == (d_pad, h_pad, w_pad), (
        f"Final logit shape mismatch: got {tuple(logit_accum.shape[1:])}, expected {(d_pad, h_pad, w_pad)}"
    )

    weight_min = float(weight_accum.min().item())
    assert weight_min > 0.0, f"weight_accum.min() must be > 0, got {weight_min}"

    final_logits = logit_accum / torch.clamp_min(weight_accum.unsqueeze(0), 1e-8)

    if not torch.isfinite(final_logits).all():
        warnings.warn("Non-finite values detected in final stitched logits.", RuntimeWarning)

    if out_channels == 1:
        full_probs = torch.sigmoid(final_logits)
        pred_prob = full_probs[0]
    else:
        full_probs = torch.softmax(final_logits, dim=0)
        pred_prob = full_probs[1 if out_channels > 1 else 0]

    pred_prob = pred_prob[: original_shape[0], : original_shape[1], : original_shape[2]]

    if not torch.isfinite(pred_prob).all():
        warnings.warn("Non-finite values detected in final stitched probability map.", RuntimeWarning)

    return pred_prob.detach().cpu().numpy().astype(np.float32)


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


def run_smoke_test(model, loader, optimizer, criterion, scaler, device, amp_enabled):
    model.train()
    batch = next(iter(loader))
    out = run_epoch(
        model=model,
        dataloader=[batch],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=scaler,
        train=True,
        amp_enabled=amp_enabled,
        debug_shapes=True,
    )
    print("Smoke test passed. Metrics:", out)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ids, val_ids = _resolve_train_val_ids(args)

    train_cases = _build_cases(train_ids, args.mri_data_root)
    val_cases = _build_cases(val_ids, args.mri_data_root)

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

    mri_backbone = ResEncUNet_3D(input_channels=2, num_classes=2).to(device)
    _load_mri_checkpoint_strict(mri_backbone, args.mri_unet_checkpoint, device=device)

    eeg_model = _build_eeg_model_from_checkpoint(args.eeg_checkpoint, device=device)

    model = EEGConditionedUNet(
        mri_backbone=mri_backbone,
        eeg_model=eeg_model,
        fusion_mode=args.fusion_mode,
        eeg_dim=args.eeg_dim,
        coord_dim=3,
        bottleneck_channels=320,
        conditioner_hidden_dim=args.conditioner_hidden_dim,
        conditioner_dropout=args.conditioner_dropout,
        eeg_dropout_p=args.eeg_dropout_p,
        eeg_null_strategy=args.eeg_null_strategy,
        alpha_init=args.alpha_init,
        alpha_max=args.alpha_max,
        film_init_delta=args.film_init_delta,
        skip_gate_hidden_dim=args.skip_gate_hidden_dim,
        skip_gate_min=args.skip_gate_min,
        skip_gate_max=args.skip_gate_max,
        skip_gate_reg_weight=args.skip_gate_reg_weight,
        debug_shapes=args.debug_shapes,
        enable_eeg_training=args.enable_eeg_training,
        verbose_fusion_debug=args.debug_shapes,
    ).to(device)

    phase = 1
    phase_cfg = _configure_trainability_phase(model=model, args=args, phase=phase)
    train_mri_encoder = phase_cfg["train_mri_encoder"]
    train_mri_decoder = phase_cfg["train_mri_decoder"]
    train_eeg = phase_cfg["train_eeg"]

    _initialize_eeg_projector_if_needed(model=model, dataloader=train_loader, device=device)

    optimizer, groups = _build_optimizer(model, args)
    if args.seg_loss_mode not in {"dice_bce", "dice_bce_plus_tversky"}:
        raise ValueError(f"Unsupported seg_loss_mode: {args.seg_loss_mode}")
    if args.ce_loss_weight < 0.0 or args.dice_loss_weight < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    if (args.ce_loss_weight + args.dice_loss_weight) <= 0.0:
        raise ValueError("At least one loss weight must be > 0.")
    if args.tversky_weight < 0.0:
        raise ValueError("tversky_weight must be non-negative.")
    if args.tversky_alpha < 0.0 or args.tversky_beta < 0.0:
        raise ValueError("tversky_alpha and tversky_beta must be non-negative.")
    if (args.tversky_alpha + args.tversky_beta) <= 0.0:
        raise ValueError("tversky_alpha + tversky_beta must be > 0.")

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    if args.enable_torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Model compiled with torch.compile.")

    print("Training configuration:")
    print("  fusion_mode:", args.fusion_mode)
    print("  phase:", phase)
    print("  two_step_training:", args.two_step_training)
    print("  train_mri_encoder:", train_mri_encoder)
    print("  train_mri_decoder:", train_mri_decoder)
    print("  train_eeg:", train_eeg)
    print("  train_segmentation_head:", args.train_segmentation_head)
    print("  freeze_eeg_model:", args.freeze_eeg_model)
    print("  enable_eeg_training:", args.enable_eeg_training)
    print("  eeg_dropout_p:", args.eeg_dropout_p)
    print("  skip_gate_reg_weight:", args.skip_gate_reg_weight)
    print("  eeg_trunk_params:", phase_cfg["eeg_trunk_params"])
    print("Trainable parameter counts:")
    print("  fusion:", _count_params(groups["fusion"]))
    print("  decoder:", _count_params(groups["decoder"]))
    print("  encoder:", _count_params(groups["encoder"]))
    print("  eeg:", _count_params(groups["eeg"]))
    print("  enable_torch_compile:", args.enable_torch_compile)
    print("  seg_loss_mode:", args.seg_loss_mode)
    print("  loss_weights:", {"ce": args.ce_loss_weight, "dice": args.dice_loss_weight})
    print(
        "  tversky:",
        {
            "weight": args.tversky_weight,
            "alpha": args.tversky_alpha,
            "beta": args.tversky_beta,
        },
    )
    print("  enable_film_logging:", args.enable_film_logging)
    print("  tb_log_every_n_steps:", args.tb_log_every_n_steps)
    print("  log_eeg_sensitivity:", args.log_eeg_sensitivity)

    os.makedirs(args.log_root, exist_ok=True)
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"eeg_conditioned_unet_fold{args.fold}_{datestr}"
    log_dir = os.path.join(args.log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    if args.output_dir and os.path.normpath(args.output_dir) != os.path.normpath(log_dir):
        warnings.warn(
            f"--output_dir ({args.output_dir}) differs from auto-computed log_dir ({log_dir}). "
            "Checkpoints/logs will go to log_dir."
        )

    run_fingerprint_payload = emit_run_fingerprint(
        script_name="eeg_conditioned_unet_training",
        train_config=vars(args),
        model_kwargs={
            "fusion_mode": args.fusion_mode,
            "eeg_dim": args.eeg_dim,
            "coord_dim": 3,
            "bottleneck_channels": 320,
            "conditioner_hidden_dim": args.conditioner_hidden_dim,
            "conditioner_dropout": args.conditioner_dropout,
            "eeg_dropout_p": args.eeg_dropout_p,
            "eeg_null_strategy": args.eeg_null_strategy,
            "alpha_init": args.alpha_init,
            "alpha_max": args.alpha_max,
            "film_init_delta": args.film_init_delta,
            "skip_gate_hidden_dim": args.skip_gate_hidden_dim,
            "skip_gate_min": args.skip_gate_min,
            "skip_gate_max": args.skip_gate_max,
            "skip_gate_reg_weight": args.skip_gate_reg_weight,
            "debug_shapes": args.debug_shapes,
            "enable_eeg_training": args.enable_eeg_training,
        },
        effective_model_config={
            "model_class": model.__class__.__name__,
            "train_mri_encoder": train_mri_encoder,
            "train_mri_decoder": train_mri_decoder,
            "train_eeg": train_eeg,
            "phase": phase,
            "two_step_training": args.two_step_training,
            "eeg_trunk_params": phase_cfg["eeg_trunk_params"],
        },
        extra={"device": str(device)},
    )

    run_fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(run_fingerprint_path, "w") as f:
        json.dump(run_fingerprint_payload, f, indent=2)
    print(f"Saved run fingerprint to: {run_fingerprint_path}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")

    writer.add_text("config/args", json.dumps(vars(args), indent=2))
    writer.add_text(
        "config/trainability",
        json.dumps(
            {
                "train_mri_encoder": train_mri_encoder,
                "train_mri_decoder": train_mri_decoder,
                "train_eeg": train_eeg,
            },
            indent=2,
        ),
    )

    param_counts = {
        "fusion": _count_params(groups["fusion"]),
        "decoder": _count_params(groups["decoder"]),
        "encoder": _count_params(groups["encoder"]),
        "eeg": _count_params(groups["eeg"]),
    }
    writer.add_text("config/param_counts", json.dumps(param_counts, indent=2))

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_last = os.path.join(ckpt_dir, "checkpoint_last.pt")
    ckpt_best_raw = os.path.join(ckpt_dir, "checkpoint_best_raw_val_loss.pt")
    ckpt_best_smoothed = os.path.join(ckpt_dir, "checkpoint_best_smoothed_val_loss.pt")

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
        phase = int(restored.get("training_phase", phase))
        phase_cfg = _configure_trainability_phase(model=model, args=args, phase=phase)
        train_mri_encoder = phase_cfg["train_mri_encoder"]
        train_mri_decoder = phase_cfg["train_mri_decoder"]
        train_eeg = phase_cfg["train_eeg"]
        best_val_loss = float(restored.get("best_val_loss", best_val_loss))
        best_smoothed_val_loss = float(restored.get("best_smoothed_val_loss", best_smoothed_val_loss))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (phase {phase}).")

    if args.smoke_test:
        run_smoke_test(model, train_loader, optimizer, criterion, scaler, device, amp_enabled=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        steps_per_epoch = max(len(train_loader), 1) + max(len(val_loader), 1) + 1
        epoch_train_step_offset = epoch * steps_per_epoch
        epoch_val_step_offset = epoch * steps_per_epoch + max(len(train_loader), 1)

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=True,
            amp_enabled=True,
            debug_shapes=(args.debug_shapes and epoch == start_epoch),
            seg_loss_mode=args.seg_loss_mode,
            ce_weight=args.ce_loss_weight,
            dice_weight=args.dice_loss_weight,
            tversky_weight=args.tversky_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            skip_gate_reg_weight=args.skip_gate_reg_weight,
            writer=writer,
            enable_film_logging=args.enable_film_logging,
            log_film_stats_every_n_steps=(args.tb_log_every_n_steps if args.enable_film_logging else 0),
            log_step_offset=epoch_train_step_offset,
            log_eeg_sensitivity=(args.log_eeg_sensitivity and args.enable_film_logging),
        )

        model_obj = _unwrap_model(model)
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            train=False,
            amp_enabled=True,
            debug_shapes=False,
            seg_loss_mode=args.seg_loss_mode,
            ce_weight=args.ce_loss_weight,
            dice_weight=args.dice_loss_weight,
            tversky_weight=args.tversky_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            skip_gate_reg_weight=args.skip_gate_reg_weight,
            writer=writer,
            enable_film_logging=args.enable_film_logging,
            log_film_stats_every_n_steps=0,
            log_step_offset=epoch_val_step_offset,
            log_eeg_sensitivity=(args.log_eeg_sensitivity and args.enable_film_logging),
        )
        es_info = es.update(epoch=epoch, val_loss=float(val_metrics["loss"]))
        smoothed_val_loss = float(es_info["smoothed_val_loss"])
        best_epoch_1based = (es_info["best_epoch"] + 1) if es_info["best_epoch"] is not None else None

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"phase={phase}, "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"train_ce={train_metrics['ce_loss']:.4f}, val_ce={val_metrics['ce_loss']:.4f}, "
            f"train_dice_loss={train_metrics['dice_loss']:.4f}, val_dice_loss={val_metrics['dice_loss']:.4f},\n"
            f"train_tversky_loss={train_metrics['tversky_loss']:.4f}, val_tversky_loss={val_metrics['tversky_loss']:.4f}, "
            f"smoothed_val_loss={smoothed_val_loss:.4f}, "
            f"train_dice={train_metrics['dice']:.4f}, val_dice={val_metrics['dice']:.4f}, "
            f"no_improve={es_info['epochs_without_improvement']}/{args.early_stopping_patience}, "
            f"alpha={float(model_obj.alpha.detach().cpu()):.6f}, "
            f"emb_norm_mean={train_metrics['emb_norm_mean']:.4f}, "
            f"emb_norm_std={train_metrics['emb_norm_std']:.4f}, "
            f"time={elapsed:.1f}s"
        )

        writer.add_scalars("loss/total", {"Train": train_metrics["loss"], "Val": val_metrics["loss"]}, epoch)
        writer.add_scalars(
            "loss/cross_entropy",
            {"Train": train_metrics["ce_loss"], "Val": val_metrics["ce_loss"]},
            epoch,
        )
        writer.add_scalars(
            "loss/dice",
            {"Train": train_metrics["dice_loss"], "Val": val_metrics["dice_loss"]},
            epoch,
        )
        writer.add_scalars(
            "loss/tversky",
            {"Train": train_metrics["tversky_loss"], "Val": val_metrics["tversky_loss"]},
            epoch,
        )
        writer.add_scalars("segmentation/dice", {"Train": train_metrics["dice"], "Val": val_metrics["dice"]}, epoch)
        writer.add_scalars(
            "segmentation/precision",
            {"Train": train_metrics["precision"], "Val": val_metrics["precision"]},
            epoch,
        )
        writer.add_scalars(
            "segmentation/recall",
            {"Train": train_metrics["recall"], "Val": val_metrics["recall"]},
            epoch,
        )
        writer.add_scalars(
            "eeg/embedding_norm_mean",
            {"Train": train_metrics["emb_norm_mean"], "Val": val_metrics["emb_norm_mean"]},
            epoch,
        )
        writer.add_scalars(
            "eeg/embedding_norm_std",
            {"Train": train_metrics["emb_norm_std"], "Val": val_metrics["emb_norm_std"]},
            epoch,
        )
        if args.enable_film_logging:
            writer.add_scalar("fusion/alpha_bottleneck", float(model_obj.alpha_bottleneck.detach().cpu()), epoch)
            writer.add_scalar("fusion/alpha_skip", float(model_obj.alpha_skip.detach().cpu()), epoch)
            writer.add_scalar("fusion/eeg_dropout_p", float(args.eeg_dropout_p), epoch)
            train_film_epoch = train_metrics.get("film_epoch_avg", {})
            val_film_epoch = val_metrics.get("film_epoch_avg", {})
            common_film_tags = sorted(set(train_film_epoch.keys()) & set(val_film_epoch.keys()))
            for tag in common_film_tags:
                tag_suffix = tag[5:] if tag.startswith("film/") else tag
                writer.add_scalars(
                    f"film_epoch/{tag_suffix}",
                    {
                        "train_epoch_avg": float(train_film_epoch[tag]),
                        "val_epoch_avg": float(val_film_epoch[tag]),
                    },
                    epoch,
                )
            train_pred_epoch = train_metrics.get("prediction_epoch_avg", {})
            val_pred_epoch = val_metrics.get("prediction_epoch_avg", {})
            common_pred_tags = sorted(set(train_pred_epoch.keys()) & set(val_pred_epoch.keys()))
            for tag in common_pred_tags:
                tag_suffix = tag[11:] if tag.startswith("prediction/") else tag
                writer.add_scalars(
                    f"prediction_epoch/{tag_suffix}",
                    {
                        "train_epoch_avg": float(train_pred_epoch[tag]),
                        "val_epoch_avg": float(val_pred_epoch[tag]),
                    },
                    epoch,
                )
        writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
        writer.add_scalar("optimizer/lr_group0", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("val/loss_raw", float(val_metrics["loss"]), epoch)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss, epoch)
        writer.add_scalar("early_stopping/best_smoothed_val_loss", es_info["best_smoothed_val_loss"], epoch)
        writer.add_scalar("early_stopping/epochs_without_improvement", es_info["epochs_without_improvement"], epoch)
        if best_epoch_1based is not None:
            writer.add_scalar("early_stopping/best_epoch", best_epoch_1based, epoch)

        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group{i}")
            writer.add_scalar(f"optimizer/lr_{name}", group["lr"], epoch)

        writer.add_scalar("params/trainable_fusion", param_counts["fusion"], epoch)
        writer.add_scalar("params/trainable_decoder", param_counts["decoder"], epoch)
        writer.add_scalar("params/trainable_encoder", param_counts["encoder"], epoch)
        writer.add_scalar("params/trainable_eeg", param_counts["eeg"], epoch)
        writer.add_scalar("training/phase", phase, epoch)

        save_checkpoint(
            path=ckpt_last,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            training_phase=phase,
            best_val_loss=min(best_val_loss, val_metrics["loss"]),
            best_smoothed_val_loss=min(best_smoothed_val_loss, smoothed_val_loss),
            args=args,
        )

        if es_info["raw_improved"]:
            best_val_loss = float(es_info["best_raw_val_loss"])
            save_checkpoint(
                path=ckpt_best_raw,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                training_phase=phase,
                best_val_loss=best_val_loss,
                best_smoothed_val_loss=min(best_smoothed_val_loss, smoothed_val_loss),
                args=args,
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
                training_phase=phase,
                best_val_loss=min(best_val_loss, val_metrics["loss"]),
                best_smoothed_val_loss=best_smoothed_val_loss,
                args=args,
            )
            print(f"New best smoothed validation loss: {best_smoothed_val_loss:.6f}")

        if es_info["should_stop"]:
            if args.two_step_training and phase == 1:
                if not os.path.exists(ckpt_best_smoothed):
                    raise FileNotFoundError(
                        "Two-step training requested, but no best smoothed checkpoint found to restore: "
                        f"{ckpt_best_smoothed}"
                    )

                print(
                    "Phase 1 early stopping triggered. Restoring best checkpoint and starting phase 2 "
                    "with decoder unfrozen."
                )
                restored = torch.load(ckpt_best_smoothed, map_location=device)
                model.load_state_dict(restored["model_state_dict"], strict=False)

                phase = 2
                phase_cfg = _configure_trainability_phase(model=model, args=args, phase=phase)
                train_mri_encoder = phase_cfg["train_mri_encoder"]
                train_mri_decoder = phase_cfg["train_mri_decoder"]
                train_eeg = phase_cfg["train_eeg"]
                optimizer, groups = _build_optimizer(model, args)
                es = EarlyStopping(
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_min_delta,
                    warmup=args.early_stopping_warmup,
                    smoothing_window=args.early_stopping_smoothing_window,
                    enabled=args.early_stopping,
                )
                best_val_loss = float("inf")
                best_smoothed_val_loss = float("inf")
                print(
                    f"Phase 2 started. decoder_unfrozen={train_mri_decoder}, "
                    f"eeg_trunk_params={phase_cfg['eeg_trunk_params']}"
                )
                continue

            print(
                "Early stopping triggered "
                f"at epoch {epoch + 1}. Best smoothed val loss: {es_info['best_smoothed_val_loss']:.6f} "
                f"(epoch {best_epoch_1based})."
            )
            break

    print("Training finished.")
    print("Best raw-loss checkpoint:", ckpt_best_raw)
    print("Best smoothed-loss checkpoint:", ckpt_best_smoothed)

    if args.restore_best_checkpoint and os.path.exists(ckpt_best_smoothed):
        restored = torch.load(ckpt_best_smoothed, map_location=device)
        model.load_state_dict(restored["model_state_dict"], strict=False)
        print(f"Restored best smoothed-loss checkpoint from: {ckpt_best_smoothed}")

    if args.run_full_volume_inference:
        print("Running full-volume sliding-window inference on validation cases...")
        if os.path.exists(ckpt_best_smoothed):
            restored = torch.load(ckpt_best_smoothed, map_location=device)
            model.load_state_dict(restored["model_state_dict"], strict=False)

        pred_dir = os.path.join(log_dir, "pred_niftis", "val")
        os.makedirs(pred_dir, exist_ok=True)

        stride = tuple(max(1, int(round(v * args.inference_stride_factor))) for v in patch_size)

        for case in val_cases:
            try:
                sid = case["id"]
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

                pred = sliding_window_inference_conditioned(
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
                print(f"Warning: skipped {sid} due to error during inference: {e}")

        print(f"Saved full-volume inference NIfTIs to: {pred_dir}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train EEG-conditioned MRI U-Net")

    parser.add_argument("--mri_unet_checkpoint", type=str, required=True)
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
    parser.add_argument("--fusion_lr", type=float, default=1e-3)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--eeg_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ce_loss_weight", type=float, default=0.7)
    parser.add_argument("--dice_loss_weight", type=float, default=0.3)
    parser.add_argument(
        "--seg_loss_mode",
        type=str,
        default="dice_bce",
        choices=["dice_bce", "dice_bce_plus_tversky"],
    )
    parser.add_argument("--tversky_weight", type=float, default=0.2)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)

    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])

    parser.add_argument("--fusion_mode", type=str, default="residual_film", choices=["residual_film", "bottleneck_film_skip_gate"])
    parser.add_argument("--eeg_dim", type=int, default=64)
    parser.add_argument("--conditioner_hidden_dim", type=int, default=256)
    parser.add_argument("--conditioner_dropout", type=float, default=0.1)
    parser.add_argument("--eeg_dropout_p", "--eeg_dropout", dest="eeg_dropout_p", type=float, default=0.3)
    parser.add_argument("--eeg_null_strategy", type=str, default="zero", choices=["zero", "learned"])
    parser.add_argument("--alpha_init", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=0.2)
    parser.add_argument("--film_init_delta", type=float, default=1e-3)
    parser.add_argument("--skip_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--skip_gate_min", type=float, default=0.75)
    parser.add_argument("--skip_gate_max", type=float, default=1.25)
    parser.add_argument("--skip_gate_reg_weight", type=float, default=0.0)
    parser.add_argument("--enable_eeg_training", action="store_true")
    parser.add_argument("--train_segmentation_head", action="store_true")

    parser.add_argument("--freeze_mri_encoder", action="store_true")
    parser.add_argument("--train_mri_encoder", action="store_true")
    parser.add_argument("--freeze_mri_decoder", action="store_true")
    parser.add_argument("--train_mri_decoder", action="store_true")
    parser.add_argument("--freeze_eeg_model", action="store_true")
    parser.add_argument("--freeze_eeg", action="store_true")
    parser.add_argument("--train_eeg", action="store_true")
    parser.add_argument("--two_step_training", action="store_true", default=True)
    parser.add_argument("--no_two_step_training", action="store_false", dest="two_step_training")

    parser.add_argument("--disable_lr_flip", action="store_true")
    parser.add_argument("--disable_strong_spatial_aug", action="store_true")
    parser.add_argument("--p_mri_heavy_noise", type=float, default=0.3)
    parser.add_argument("--eeg_max_offset", type=int, default=0)
    parser.add_argument("--eeg_training_drop_ratio", type=float, default=0.0)
    parser.add_argument("--force_dataset_in_memory", action="store_true", default=True)
    parser.add_argument("--no_force_dataset_in_memory", action="store_false", dest="force_dataset_in_memory")

    parser.add_argument("--debug_shapes", action="store_true")
    parser.add_argument("--enable_torch_compile", action="store_true")
    parser.add_argument("--enable_film_logging", action="store_true")
    parser.add_argument("--tb_log_every_n_steps", type=int, default=50)
    parser.add_argument("--log_eeg_sensitivity", action="store_true")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
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
    main(args)
