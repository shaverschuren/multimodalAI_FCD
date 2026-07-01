"""
training.multimodal.UNet_with_prior_training.py
Training script for a UNet model that incorporates EEG prior information
into its architecture for early multimodal fusion of MRI and EEG data.

The model combines:
- MRI backbone: 3D Residual Encoder U-Net (nnUNet-style ResEncUNet_3D)
- Input: T1 + FLAIR + EEG-derived Gaussian prior (3 channels)
- Output: segmentation logits (num_classes channels)

The script assumes:
1. A pre-trained MRI-only (2-channel) nnUNet checkpoint is available
2. An EEG model predictions JSON file with localization estimates
3. A 5-fold dataset split JSON is available

Author: Sjors Verschuren
Date: January 2026
"""

import os
import json
import argparse
import time
import warnings
from datetime import datetime
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from datasets.multimodal import UNetWithPriorDataset
from models.multimodal import ResEncUNet_3D_with_prior
from util import emit_run_fingerprint, EarlyStopping

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:
    _scipy_gaussian_filter = None


PRIOR_SUFFIX = "_prior.nii.gz"


def load_eeg_predictions(json_path):
    """
    Load EEG model predictions from JSON.
    
    Expected format:
    {
        "train": {
            "RESP0001": {"mu": [x, y, z], "sigma": float or [sx, sy, sz]},
            "RESP0002": {"mu": [x, y, z], "sigma": [sx, sy, sz]},
            ...
        },
        "val": {
            "RESP0101": {"mu": [x, y, z], "sigma": float or [sx, sy, sz]},
            ...
        }
    }
    
    Where mu are normalized MNI coordinates [-1, 1] and sigma is uncertainty.
    Prior generation will be handled in the dataset class.
    """
    with open(json_path, "r") as f:
        predictions = json.load(f)

    predictions_combined = {}
    for split in ["train", "val"]:
        if split in predictions:
            for subj_id, pred in predictions[split].items():
                predictions_combined[subj_id] = pred

    return predictions_combined

def load_fold_split(json_path, fold_index):
    """Load subject IDs for a given fold from k_fold_splits.json."""
    with open(json_path, "r") as f:
        payload = json.load(f)
    
    fold_payload = payload["folds"]
    fold_key = f"fold_{fold_index}"
    if fold_key not in fold_payload:
        raise ValueError(f"{fold_key} not found in fold split JSON.")
    
    train_ids = fold_payload[fold_key]["train_ids"]
    val_ids = fold_payload[fold_key]["val_ids"]
    
    return train_ids, val_ids

def build_cases(subject_ids, mri_data_dir):
    """
    Build case list from subject IDs and MRI data directory.
    Each case should have a .npz file: <id>_preproc.npz
    """
    cases = []
    for sid in subject_ids:
        npy_path = os.path.join(mri_data_dir, sid, f"{sid}_preproc.npz")
        if os.path.exists(npy_path):
            cases.append({"id": sid, "npy": npy_path})
        else:
            print(f"Warning: Missing preprocessed file for {sid} at {npy_path}")
    
    if not cases:
        raise ValueError(f"No valid cases found in {mri_data_dir}")
    
    print(f"Built {len(cases)} valid cases from {len(subject_ids)} subjects.")
    return cases


def _extract_pid(path, suffix):
    import re
    name = os.path.basename(path)
    # Extract patient ID matching pattern RESP**** from the filename
    match = re.search(r'(RESP\d{4})', name)
    if match:
        return match.group(1)
    return None


def _index_files(root_dir, suffix, must_contain_dir=None):
    """Recursively index files by patient ID, similar to plot_mil_mh_prior_niftis."""
    pattern = os.path.join(root_dir, "**", f"*{suffix}")
    matches = sorted(glob(pattern, recursive=True))
    mapping = {}
    for path in matches:
        if must_contain_dir is not None:
            path_parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
            if must_contain_dir.lower() not in path_parts:
                continue
        pid = _extract_pid(path, suffix)
        if pid is None:
            continue
        if pid not in mapping:
            mapping[pid] = path
    return mapping


def prepare_val_prior_cache_from_niftis(val_cases, val_prior_dir, out_cache_dir, split_dir_token="val"):
    """
    Build dataset-compatible .npy prior cache from recursive prior NIfTI outputs.

    The dataset loader expects files named <patient_id>_prior.npy in prior_cache_dir.
    We find <patient_id>_prior.nii.gz recursively (prefer paths under split_dir_token),
    validate shapes against each case npz image, and write cache .npy files.
    """
    if val_prior_dir is None:
        raise ValueError("val_prior_dir is required when using external validation priors.")

    val_prior_map = _index_files(val_prior_dir, PRIOR_SUFFIX, must_contain_dir=split_dir_token)
    if not val_prior_map:
        raise FileNotFoundError(
            f"No prior NIfTIs found under paths containing '{split_dir_token}'. "
            f"Searched recursively in: {val_prior_dir}"
        )
    prior_map = val_prior_map
    print(f"Using {len(prior_map)} prior NIfTIs from paths containing '{split_dir_token}'.")

    missing = [c["id"] for c in val_cases if c["id"] not in prior_map]
    if missing:
        missing_preview = ", ".join(missing[:10])
        print(
            "Warning: Missing validation prior NIfTIs for "
            f"{len(missing)} subjects (first 10: {missing_preview}). "
            "These subjects will be skipped."
        )

    valid_cases = [c for c in val_cases if c["id"] in prior_map]
    if not valid_cases:
        raise FileNotFoundError(
            "No cases remain after filtering for available validation prior NIfTIs. "
            f"Searched recursively in: {val_prior_dir}"
        )

    os.makedirs(out_cache_dir, exist_ok=True)
    for c in valid_cases:
        sid = c["id"]
        prior_path = prior_map[sid]

        prior = nib.load(prior_path).get_fdata(dtype="float32")
        npz = np.load(c["npy"], allow_pickle=True)
        img = npz["image"]
        npz.close()

        expected_shape = tuple(img.shape[1:])
        if prior.shape != expected_shape:
            raise ValueError(
                f"Prior shape mismatch for {sid}: prior={prior.shape}, expected={expected_shape} "
                f"(from {prior_path})"
            )

        prior = prior.clip(min=0.0, max=1.0).astype("float32")
        out_npy = os.path.join(out_cache_dir, f"{sid}_prior.npy")
        np.save(out_npy, prior)

    print(
        f"Prepared validation prior cache for {len(valid_cases)} subjects at: {out_cache_dir}"
    )
    return valid_cases

def dice_loss(logits, targets, smooth=1.0):
    """
    Dice loss for segmentation.
    logits: [B, C, D, H, W]
    targets: [B, D, H, W] (binary/class indices)
    """
    probs = torch.softmax(logits, dim=1)  # [B, C, D, H, W]
    
    # For binary segmentation (C=2), take foreground probability
    if logits.shape[1] == 2:
        probs = probs[:, 1]  # [B, D, H, W]
    else:
        # For multi-class, use max probability
        probs = probs.max(dim=1)[0]  # [B, D, H, W]
    
    targets_float = targets.float()
    
    intersection = (probs * targets_float).sum()
    union = probs.sum() + targets_float.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
    writer=None,
    amp_enabled=True,
):
    """Train for one epoch. Returns (epoch_loss, epoch_dice)."""
    model.train()
    
    total_loss = 0.0
    total = 0
    total_dice = 0.0
    
    global_step = epoch * len(dataloader)
    
    for step, batch in enumerate(dataloader):
        x = batch["x"].to(device, non_blocking=True)  # [B, 3, D, H, W]
        y = batch["y"].to(device, non_blocking=True)  # [B, D, H, W]
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
            logits = model(x)  # [B, num_classes, D, H, W]
            ce_loss = criterion(logits, y.long())
            dice_loss_val = dice_loss(logits, y)
            dice_score = 1.0 - float(dice_loss_val.detach())
            loss = ce_loss + dice_loss_val
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score * batch_size
        total += batch_size
        
        # Per-step logging
        if writer is not None and step % 10 == 0:
            step_idx = global_step + step
            writer.add_scalar("Train/Loss_step", loss.item(), step_idx)
            writer.add_scalar("Train/Dice_step", dice_score, step_idx)
    
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_dice = total_dice / total if total > 0 else 0.0
    return epoch_loss, epoch_dice

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer=None, amp_enabled=True):
    """Validate for one epoch. Returns (epoch_loss, epoch_dice)."""
    model.eval()
    
    total_loss = 0.0
    total = 0
    total_dice = 0.0
    
    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)  # [B, 3, D, H, W]
        y = batch["y"].to(device, non_blocking=True)  # [B, D, H, W]
        
        with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
            logits = model(x)  # [B, num_classes, D, H, W]
            ce_loss = criterion(logits, y.long())
            dice_loss_val = dice_loss(logits, y)
            dice_score = 1.0 - float(dice_loss_val)
            loss = ce_loss + dice_loss_val
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score * batch_size
        total += batch_size
    
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_dice = total_dice / total if total > 0 else 0.0
    # Log validation metrics
    if writer is not None:
        writer.add_scalar("Val/Dice_epoch", epoch_dice, epoch)
    return epoch_loss, epoch_dice


def _dice_per_sample_from_logits(logits, targets, smooth=1.0):
    """Compute per-sample Dice from logits and integer targets."""
    pred = torch.argmax(logits, dim=1)  # [B, D, H, W]
    tgt = targets.long()

    if logits.shape[1] == 2:
        pred_fg = (pred == 1).float()
        tgt_fg = (tgt == 1).float()
        inter = (pred_fg * tgt_fg).flatten(1).sum(dim=1)
        union = pred_fg.flatten(1).sum(dim=1) + tgt_fg.flatten(1).sum(dim=1)
        dice = (2.0 * inter + smooth) / (union + smooth)
        return dice

    # Multi-class: average dice over foreground classes (1..C-1)
    class_dices = []
    for c in range(1, logits.shape[1]):
        pred_c = (pred == c).float()
        tgt_c = (tgt == c).float()
        inter = (pred_c * tgt_c).flatten(1).sum(dim=1)
        union = pred_c.flatten(1).sum(dim=1) + tgt_c.flatten(1).sum(dim=1)
        class_dices.append((2.0 * inter + smooth) / (union + smooth))

    if not class_dices:
        return torch.ones((logits.shape[0],), device=logits.device)
    return torch.stack(class_dices, dim=1).mean(dim=1)


def _set_encoder_trainable(model, trainable=True):
    """
    Toggle trainability for the MRI encoder of the multimodal backbone.

    Returns:
        num_params_toggled (int): number of parameters for which requires_grad was set.
    """
    # Expected chain: MultimodalResEncUNet_3D.backbone (ResEncUNet_3D).backbone (ResidualEncoderUNet)
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "backbone"):
        raise AttributeError("Model does not expose expected backbone structure for encoder freezing.")

    core_backbone = model.backbone.backbone

    if not hasattr(core_backbone, "encoder"):
        raise AttributeError("Backbone does not expose an 'encoder' module for freezing.")

    num_params_toggled = 0
    for p in core_backbone.encoder.parameters():
        p.requires_grad = bool(trainable)
        num_params_toggled += p.numel()

    return num_params_toggled


def _get_encoder_stem_conv_weight(model):
    """Return the first encoder stem conv weight parameter that receives the added prior channel."""
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not expose the MRI backbone wrapper required for freezing.")

    mri_backbone = model.backbone
    if hasattr(mri_backbone, "stem_conv"):
        stem_conv = mri_backbone.stem_conv
    else:
        if not hasattr(mri_backbone, "backbone") or not hasattr(mri_backbone.backbone, "encoder"):
            raise AttributeError("Backbone does not expose an encoder stem convolution for freezing.")

        try:
            stem_conv = mri_backbone.backbone.encoder.stem.convs[0].conv
        except Exception as exc:
            raise AttributeError("Backbone does not expose the expected encoder stem convolution.") from exc

    if not isinstance(stem_conv, nn.Conv3d):
        raise AttributeError("Encoder stem convolution is not an nn.Conv3d instance.")

    if stem_conv.in_channels < 3:
        raise ValueError(
            f"Expected a 3-channel multimodal stem, got {stem_conv.in_channels} input channels."
        )

    return stem_conv.weight


def _freeze_encoder_pretrained_weights(model):
    """
    Freeze all pretrained encoder weights while leaving the added prior-channel slice trainable.

    Returns:
        tuple[nn.Parameter, torch.utils.hooks.RemovableHandle, int]: the stem conv weight parameter,
        a gradient hook handle that keeps pretrained stem channels frozen, and the count of frozen
        encoder parameters.
    """
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not expose the MRI backbone wrapper required for freezing.")

    mri_backbone = model.backbone
    encoder = getattr(mri_backbone, "encoder", None)
    if encoder is None and hasattr(mri_backbone, "backbone"):
        encoder = getattr(mri_backbone.backbone, "encoder", None)
    if encoder is None:
        raise AttributeError("Backbone does not expose an 'encoder' module for freezing.")

    num_params_toggled = 0
    for p in encoder.parameters():
        p.requires_grad = False
        num_params_toggled += p.numel()

    stem_weight = _get_encoder_stem_conv_weight(model)
    stem_weight.requires_grad = True

    pretrained_channels = 2

    def _mask_pretrained_stem_grad(grad):
        masked_grad = grad.clone()
        masked_grad[:, :pretrained_channels, ...] = 0
        return masked_grad

    grad_hook_handle = stem_weight.register_hook(_mask_pretrained_stem_grad)

    return stem_weight, grad_hook_handle, num_params_toggled - stem_weight.numel()


@torch.no_grad()
def infer_predictions(model, dataloader, criterion, device, amp_enabled=True):
    """Run inference and collect per-case metrics and aggregate summary."""
    model.eval()

    cases = {}
    total = 0
    ce_sum = 0.0
    dice_sum = 0.0
    loss_sum = 0.0

    for batch in dataloader:
        ids = batch["id"]
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
            logits = model(x)
            ce_loss = criterion(logits, y.long())
            dice_loss_val = dice_loss(logits, y)
            loss = ce_loss + dice_loss_val

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        dice_per_sample = _dice_per_sample_from_logits(logits, y)

        b = x.size(0)
        total += b
        ce_sum += float(ce_loss) * b
        dice_sum += float(dice_per_sample.mean()) * b
        loss_sum += float(loss) * b

        for i, sid in enumerate(ids):
            cases[sid] = {
                "dice": float(dice_per_sample[i]),
                "pred_foreground_voxels": int((pred[i] > 0).sum().item()),
                "gt_foreground_voxels": int((y[i].long() > 0).sum().item()),
                "mean_foreground_prob": float(probs[i, 1].mean().item()) if logits.shape[1] > 1 else None,
            }

    summary = {
        "num_cases": len(cases),
        "loss": (loss_sum / total) if total > 0 else None,
        "ce_loss": (ce_sum / total) if total > 0 else None,
        "dice": (dice_sum / total) if total > 0 else None,
    }
    return {"summary": summary, "cases": cases}


@torch.no_grad()
def generate_prediction_niftis(
    model,
    device,
    case_npz_by_id,
    prior_npy_by_id,
    output_dir,
    amp_enabled=True,
    patch_size=128,
    use_gaussian_weighting=True,
    gaussian_sigma_scale=1 / 8,
    gaussian_value_scaling_factor=10.0,
    accumulate_on_cpu=True,
):
    """Generate unthresholded full-volume prediction-probability NIfTIs via patch stitching."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    saved = 0
    skipped = 0

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    # Use dense sliding-window inference: step size = 0.25 * patch_size.
    patch_stride = max(1, int(round(patch_size * 0.25)))

    def _compute_gaussian_importance_map(
        patch_size_3d,
        sigma_scale=1 / 8,
        value_scaling_factor=10.0,
        dtype=torch.float32,
    ):
        patch_size_3d = tuple(int(v) for v in patch_size_3d)
        center = tuple(v // 2 for v in patch_size_3d)

        if _scipy_gaussian_filter is not None:
            gaussian = np.zeros(patch_size_3d, dtype=np.float32)
            gaussian[center] = 1.0
            sigmas = [max(float(v) * float(sigma_scale), 1e-6) for v in patch_size_3d]
            gaussian = _scipy_gaussian_filter(gaussian, sigma=sigmas, mode="constant", cval=0.0)
        else:
            coords = [torch.arange(v, dtype=torch.float32) for v in patch_size_3d]
            zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
            sigmas = [max(float(v) * float(sigma_scale), 1e-6) for v in patch_size_3d]
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
        gaussian[gaussian == 0] = float(nonzero.min())

        return torch.from_numpy(gaussian.astype(np.float32)).to(dtype=dtype).unsqueeze(0)

    def _extract_logits(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]
        if isinstance(output, dict):
            for key in ("logits", "out", "pred", "prediction"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        raise TypeError(f"Could not extract logits from model output type {type(output)!r}")

    def _compute_starts(dim, psize, stride):
        if dim <= psize:
            return [0]
        starts = list(range(0, dim - psize + 1, stride))
        last = dim - psize
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _predict_full_prob(x_full):
        # x_full: [3, D, H, W] float32
        _, d, h, w = x_full.shape
        original_shape = (d, h, w)

        pad_d = max(0, patch_size - d)
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        if pad_d or pad_h or pad_w:
            x_full = F.pad(x_full, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)

        _, d_pad, h_pad, w_pad = x_full.shape

        d_starts = _compute_starts(d_pad, patch_size, patch_stride)
        h_starts = _compute_starts(h_pad, patch_size, patch_stride)
        w_starts = _compute_starts(w_pad, patch_size, patch_stride)
        n_tiles = len(d_starts) * len(h_starts) * len(w_starts)

        overlap = 1.0 - (float(patch_stride) / float(patch_size))
        print(
            "[inference] sliding-window config | "
            f"patch_size=({patch_size}, {patch_size}, {patch_size}), "
            f"stride=({patch_stride}, {patch_stride}, {patch_stride}), "
            f"overlap=({overlap:.4f}, {overlap:.4f}, {overlap:.4f}), "
            f"tiles={n_tiles}, gaussian_weighting={bool(use_gaussian_weighting)}, "
            f"accumulate_on_cpu={bool(accumulate_on_cpu)}"
        )

        accumulator_device = torch.device("cpu") if accumulate_on_cpu else device
        gaussian = None
        if use_gaussian_weighting:
            gaussian = _compute_gaussian_importance_map(
                patch_size_3d=(patch_size, patch_size, patch_size),
                sigma_scale=gaussian_sigma_scale,
                value_scaling_factor=gaussian_value_scaling_factor,
                dtype=torch.float32,
            ).to(accumulator_device)

        out_channels = None
        logit_accum = None
        weight_accum = torch.zeros((d_pad, h_pad, w_pad), dtype=torch.float32, device=accumulator_device)

        for ds in d_starts:
            for hs in h_starts:
                for ws in w_starts:
                    de = ds + patch_size
                    he = hs + patch_size
                    we = ws + patch_size

                    patch = x_full[:, ds:de, hs:he, ws:we]

                    patch = patch.unsqueeze(0).to(device, non_blocking=True)

                    with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                        output = model(patch)
                        patch_logits = _extract_logits(output)

                    if patch_logits.ndim != 5 or patch_logits.shape[0] != 1:
                        raise RuntimeError(
                            f"Expected patch logits shape (1, C_out, p, p, p), got {tuple(patch_logits.shape)}"
                        )

                    patch_logits = patch_logits[0].detach().to(accumulator_device, dtype=torch.float32)

                    if tuple(patch_logits.shape[1:]) != (patch_size, patch_size, patch_size):
                        raise RuntimeError(
                            f"Unexpected patch logit spatial shape {tuple(patch_logits.shape[1:])}; "
                            f"expected {(patch_size, patch_size, patch_size)}"
                        )

                    if out_channels is None:
                        out_channels = int(patch_logits.shape[0])
                        logit_accum = torch.zeros(
                            (out_channels, d_pad, h_pad, w_pad), dtype=torch.float32, device=accumulator_device
                        )

                    if use_gaussian_weighting:
                        weighted_logits = patch_logits * gaussian
                        patch_weight = gaussian[0]
                    else:
                        weighted_logits = patch_logits
                        patch_weight = torch.ones(
                            (patch_size, patch_size, patch_size), dtype=torch.float32, device=accumulator_device
                        )

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

        return pred_prob

    for sid, npz_path in case_npz_by_id.items():
        if npz_path is None or not os.path.exists(npz_path):
            print(f"  [skip] Missing MRI npz path for {sid}")
            skipped += 1
            continue

        prior_path = prior_npy_by_id.get(sid)
        if prior_path is None or not os.path.exists(prior_path):
            print(f"  [skip] Missing prior npy path for {sid}")
            skipped += 1
            continue

        try:
            npz = np.load(npz_path, allow_pickle=True)
            img = np.asarray(npz["image"], dtype=np.float32)  # [2,D,H,W]
            affine = np.asarray(npz["affine"], dtype=np.float32)
            npz.close()
        except Exception as e:
            print(f"  [skip] Error loading {npz_path}: {e}")
            skipped += 1
            continue

        try:
            prior = np.load(prior_path).astype(np.float32)  # [D,H,W]
        except Exception as e:
            print(f"  [skip] Error loading {prior_path}: {e}")
            skipped += 1
            continue

        if img.ndim != 4 or img.shape[0] != 2:
            print(f"  [skip] Invalid image shape for {sid}: expected [2,D,H,W], got {img.shape}")
            skipped += 1
            continue
        if prior.ndim != 3:
            print(f"  [skip] Invalid prior shape for {sid}: expected [D,H,W], got {prior.shape}")
            skipped += 1
            continue
        if tuple(img.shape[1:]) != tuple(prior.shape):
            print(
                f"  [skip] Shape mismatch for {sid}: image spatial={tuple(img.shape[1:])}, "
                f"prior={tuple(prior.shape)}"
            )
            skipped += 1
            continue

        x_full = torch.from_numpy(np.concatenate([img, prior[None, ...]], axis=0)).float()

        pred_prob = _predict_full_prob(x_full).cpu().numpy().astype(np.float32)
        pred_prob = np.clip(pred_prob, 0.0, 1.0)

        nii_path = os.path.join(output_dir, f"{sid}_pred_prob.nii.gz")
        nib.save(nib.Nifti1Image(pred_prob, affine), nii_path)
        saved += 1

    print(f"  Saved {saved} prediction NIfTIs to {output_dir} ({skipped} skipped)")


def train(
    json_fold_path,
    fold_index,
    mri_data_dir,
    mri_checkpoint_path,
    num_classes=2,
    batch_size=2,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=50,
    log_root="././data/tmp/runs/multimodal",
    num_workers=0,
    test_mode=False,
    amp_enabled=True,
    prior_cache_dir=None,
    val_prior_dir=None,
    early_stopping=True,
    early_stopping_patience=200,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=150,
    early_stopping_smoothing_window=10,
    restore_best_checkpoint=True,
    freeze_encoder_epochs=0,
    multi_phase_learning=True,
    lr_unfreeze=0.5,
):
    """
    Main training entrypoint for multimodal MRI + EEG prior segmentation.
    
    Args:
        json_fold_path (str): Path to JSON with k-fold subject splits.
        fold_index (int): Which fold to train (0-4).
        mri_data_dir (str): Directory containing preprocessed MRI data (*.npz files).
        mri_checkpoint_path (str): Path to pre-trained MRI-only (2-channel) nnUNet checkpoint.
        num_classes (int): Number of output classes (default 2 for binary segmentation).
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for AdamW optimizer.
        epochs (int): Number of training epochs.
        log_root (str): Root directory for TensorBoard logs.
        num_workers (int): Number of DataLoader workers.
        test_mode (bool): If True, use limited data for quick testing.
        amp_enabled (bool): If True, use automatic mixed precision.
        prior_cache_dir (str): Optional directory to cache priors for faster re-initialization.
        val_prior_dir (str): Directory to recursively search for precomputed prior NIfTIs
            (<patient_id>_prior.nii.gz) produced by the validation end of the 10-fold EEG
            training runs. All priors (train and val) are loaded from these files.
        early_stopping (bool): Enable early stopping based on smoothed validation loss.
        early_stopping_patience (int): Epochs without improvement before stopping.
        early_stopping_min_delta (float): Minimum absolute improvement in smoothed val loss.
        early_stopping_warmup (int): Minimum epoch before early stopping can trigger.
        early_stopping_smoothing_window (int): Rolling average window for val loss smoothing.
        restore_best_checkpoint (bool): Restore best smoothed-loss checkpoint before returning.
        freeze_encoder_epochs (int): Freeze MRI encoder weights for the first N epochs,
            then unfreeze automatically from epoch N+1 onward.
        multi_phase_learning (bool): If True, turn the first early-stopping trigger into a
            second training phase by restoring the best checkpoint, unfreezing the encoder,
            reducing the learning rate, and resetting early stopping.
        lr_unfreeze (float): Multiplier applied to ``lr`` when phase 2 starts.
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load fold splits
    train_ids, val_ids = load_fold_split(json_fold_path, fold_index)
    print(f"Fold {fold_index}: {len(train_ids)} train subjects, {len(val_ids)} val subjects")
    
    # Limit to test data if requested
    if test_mode:
        train_ids = train_ids[:4]
        val_ids = val_ids[:2]
        print(f"Test mode: limited to {len(train_ids)} train and {len(val_ids)} val subjects")
    
    # Build case lists
    train_cases = build_cases(train_ids, mri_data_dir)
    val_cases = build_cases(val_ids, mri_data_dir)

    if val_prior_dir is None:
        raise ValueError(
            "--val_prior_dir is required. This training uses only precomputed prior NIfTIs "
            "from EEG validation outputs (recursive search)."
        )

    if prior_cache_dir is None:
        prior_cache_root = os.path.join(val_prior_dir, "_multimodal_prior_cache")
    else:
        prior_cache_root = prior_cache_dir

    train_prior_cache_dir = os.path.join(prior_cache_root, "train_from_nifti")
    val_prior_cache_dir = os.path.join(prior_cache_root, "val_from_nifti")

    # Load priors for ALL subjects (train + val in current fold) from EEG-validation NIfTI outputs.
    train_cases = prepare_val_prior_cache_from_niftis(
        val_cases=train_cases,
        val_prior_dir=val_prior_dir,
        out_cache_dir=train_prior_cache_dir,
    )
    val_cases = prepare_val_prior_cache_from_niftis(
        val_cases=val_cases,
        val_prior_dir=val_prior_dir,
        out_cache_dir=val_prior_cache_dir,
    )
    
    # Create datasets
    train_dataset = UNetWithPriorDataset(
        cases=train_cases,
        eeg_preds_by_id={},
        image_dtype=torch.float16,
        prior_dtype=torch.float16,
        gt_dtype=torch.uint8,
        clamp_min_sigma_vox=1.0,
        fallback_prior="zeros",
        return_float32=True,
        prior_cache_dir=train_prior_cache_dir,
        overwrite_cache=False,
        enable_augmentation=True,  # Enable augmentation for training
    )
    train_dataset.augment.p_channel_dropout = 0.0 # No prior dropout for phase 1, because that doesn't make sense.
    
    val_dataset = UNetWithPriorDataset(
        cases=val_cases,
        eeg_preds_by_id={},
        image_dtype=torch.float16,
        prior_dtype=torch.float16,
        gt_dtype=torch.uint8,
        clamp_min_sigma_vox=1.0,
        fallback_prior="zeros",
        return_float32=True,
        prior_cache_dir=val_prior_cache_dir,
        overwrite_cache=False,
        enable_augmentation=False,  # No augmentation for validation
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    
    # Build model
    model = ResEncUNet_3D_with_prior(num_classes=num_classes, use_prior=True).to(device)
    
    # Load pre-trained MRI weights and expand for prior channel
    print(f"Loading pre-trained MRI checkpoint from {mri_checkpoint_path}")
    model.load_mri_only_pretrained(
        checkpoint_path=mri_checkpoint_path,
        device=device,
        verbose=True,
    )

    if freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs must be >= 0")
    if lr_unfreeze <= 0:
        raise ValueError("lr_unfreeze must be > 0")

    encoder_is_frozen = False
    frozen_stem_weight = None
    frozen_stem_grad_hook_handle = None
    if freeze_encoder_epochs > 0:
        frozen_stem_weight, frozen_stem_grad_hook_handle, frozen_params = _freeze_encoder_pretrained_weights(model)
        encoder_is_frozen = True
        print(
            f"Freezing pretrained MRI encoder weights for first {freeze_encoder_epochs} epoch(s) "
            f"({frozen_params} parameters). The added prior-channel slice in the stem conv stays trainable."
        )

    def _make_optimizer(current_lr):
        return torch.optim.AdamW(
            model.parameters(),
            lr=current_lr,
            weight_decay=weight_decay,
        )
    
    # Loss function: combination of CE + Dice for segmentation
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = _make_optimizer(lr)
    
    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_enabled))
    
    # TensorBoard logging
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"multimodal_fold{fold_index}_{datestr}")
    os.makedirs(log_dir, exist_ok=True)

    run_fingerprint_payload = emit_run_fingerprint(
        script_name="UNet_with_prior_training",
        train_config={
            "json_fold_path": json_fold_path,
            "fold_index": fold_index,
            "mri_data_dir": mri_data_dir,
            "mri_checkpoint_path": mri_checkpoint_path,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "log_root": log_root,
            "num_workers": num_workers,
            "test_mode": test_mode,
            "amp_enabled": amp_enabled,
            "prior_cache_dir": prior_cache_dir,
            "val_prior_dir": val_prior_dir,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_warmup": early_stopping_warmup,
            "early_stopping_smoothing_window": early_stopping_smoothing_window,
            "restore_best_checkpoint": restore_best_checkpoint,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "multi_phase_learning": multi_phase_learning,
            "lr_unfreeze": lr_unfreeze,
        },
        model_kwargs={},
        effective_model_config={
            "model_class": model.__class__.__name__,
            "num_classes": num_classes,
            "use_prior": True,
        },
        extra={"device": str(device)},
    )
    run_fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(run_fingerprint_path, "w") as f:
        json.dump(run_fingerprint_payload, f, indent=2)
    print(f"Saved run fingerprint to: {run_fingerprint_path}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    es = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        warmup=early_stopping_warmup,
        smoothing_window=early_stopping_smoothing_window,
        enabled=early_stopping,
    )

    ckpt_last = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    ckpt_best_raw = os.path.join(checkpoint_dir, "checkpoint_best_raw_val_loss.pt")
    ckpt_best_smoothed = os.path.join(checkpoint_dir, "checkpoint_best_smoothed_val_loss.pt")

    def _make_checkpoint(epoch_1based, val_loss_val, val_dice_val, smoothed_val_loss_val, es_info):
        scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
        return {
            "epoch": epoch_1based,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler_state,
            "val_loss": val_loss_val,
            "val_dice": val_dice_val,
            "smoothed_val_loss": smoothed_val_loss_val,
            "best_raw_val_loss": es_info["best_raw_val_loss"],
            "best_smoothed_val_loss": es_info["best_smoothed_val_loss"],
            "args": {
                "json_fold_path": json_fold_path,
                "fold_index": fold_index,
                "mri_data_dir": mri_data_dir,
                "mri_checkpoint_path": mri_checkpoint_path,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "log_root": log_root,
                "num_workers": num_workers,
                "test_mode": test_mode,
                "amp_enabled": amp_enabled,
                "prior_cache_dir": prior_cache_dir,
                "val_prior_dir": val_prior_dir,
                "early_stopping": early_stopping,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_warmup": early_stopping_warmup,
                "early_stopping_smoothing_window": early_stopping_smoothing_window,
                "restore_best_checkpoint": restore_best_checkpoint,
                "freeze_encoder_epochs": freeze_encoder_epochs,
                "multi_phase_learning": multi_phase_learning,
                "lr_unfreeze": lr_unfreeze,
            },
        }
    
    # Save hyperparameters
    hparams = {
        "fold": fold_index,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "freeze_encoder_epochs": freeze_encoder_epochs,
        "multi_phase_learning": multi_phase_learning,
        "lr_unfreeze": lr_unfreeze,
    }
    writer.add_text("hyperparameters", json.dumps(hparams, indent=2))
    writer.add_text(
        "early_stopping",
        json.dumps(
            {
                "enabled": early_stopping,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "warmup": early_stopping_warmup,
                "smoothing_window": early_stopping_smoothing_window,
            },
            indent=2,
        ),
    )

    best_val_loss = float("inf")
    best_val_dice = float("-inf")
    stopped_early = False
    multi_phase_started = False
    phase_2_lr = lr * lr_unfreeze

    if multi_phase_learning:
        writer.add_text(
            "multi_phase_learning",
            json.dumps(
                {
                    "enabled": True,
                    "lr_unfreeze_multiplier": lr_unfreeze,
                    "phase2_lr": phase_2_lr,
                },
                indent=2,
            ),
        )
    else:
        writer.add_text(
            "multi_phase_learning",
            json.dumps({"enabled": False}, indent=2),
        )
    
    for epoch in range(epochs):
        epoch_start_time = time.time()

        if encoder_is_frozen and epoch >= freeze_encoder_epochs:
            unfrozen_params = _set_encoder_trainable(model, trainable=True)
            encoder_is_frozen = False
            print(
                f"Unfroze MRI encoder at epoch {epoch + 1} "
                f"({unfrozen_params} parameters trainable again)."
            )
        
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'=' * 60}")
        
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            epoch,
            writer,
            amp_enabled=amp_enabled,
        )
        
        val_loss, val_dice = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            writer,
            amp_enabled=amp_enabled,
        )
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, DSC: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, DSC: {val_dice:.4f}")
        print(f"Epoch duration: {epoch_duration:.1f}s")
        print(f"Phase: {'2' if multi_phase_started else '1'}")
        
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Dice", {"Train": train_dice, "Val": val_dice}, epoch)
        writer.add_scalar("train/encoder_frozen", 1 if encoder_is_frozen else 0, epoch)
        writer.add_scalar("train/phase", 2 if multi_phase_started else 1, epoch)

        es_info = es.update(epoch, val_loss)
        smoothed_val_loss = es_info["smoothed_val_loss"]
        best_epoch_1based = (es_info["best_epoch"] + 1) if es_info["best_epoch"] is not None else None
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Smoothed val loss: {smoothed_val_loss:.6f} | "
            f"best_smoothed: {es_info['best_smoothed_val_loss']:.6f} @ epoch {best_epoch_1based} | "
            f"best_raw: {es_info['best_raw_val_loss']:.6f} | "
            f"no_improve: {es_info['epochs_without_improvement']}/{early_stopping_patience} | "
            f"lr: {lr_current:.2e}"
        )

        writer.add_scalar("val/loss_raw", val_loss, epoch)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss, epoch)
        writer.add_scalar("early_stopping/best_smoothed_val_loss", es_info["best_smoothed_val_loss"], epoch)
        writer.add_scalar("early_stopping/epochs_without_improvement", es_info["epochs_without_improvement"], epoch)
        if best_epoch_1based is not None:
            writer.add_scalar("early_stopping/best_epoch", best_epoch_1based, epoch)

        ckpt_data = _make_checkpoint(epoch + 1, val_loss, val_dice, smoothed_val_loss, es_info)
        torch.save(ckpt_data, ckpt_last)

        if es_info["raw_improved"]:
            torch.save(ckpt_data, ckpt_best_raw)
            print(f"✓ New best raw val loss: {val_loss:.4f} (DSC: {val_dice:.4f})")

        if es_info["improved"]:
            torch.save(ckpt_data, ckpt_best_smoothed)
            print(f"✓ New best smoothed val loss: {smoothed_val_loss:.6f}")
        
        # Backward-compatible best-by-dice checkpoint
        if val_dice > best_val_dice:
            best_val_loss = val_loss
            best_val_dice = val_dice
            checkpoint_path = os.path.join(log_dir, "checkpoint_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                },
                checkpoint_path,
            )
            print(f"✓ New best validation DSC: {val_dice:.4f} (loss: {val_loss:.4f})")

        if es_info["should_stop"]:
            if multi_phase_learning and not multi_phase_started:
                if not os.path.exists(ckpt_best_smoothed):
                    raise FileNotFoundError(
                        "Multi-phase learning requested, but the best smoothed checkpoint does not exist: "
                        f"{ckpt_best_smoothed}"
                    )

                print(
                    "\nEarly stopping triggered. Starting multi-phase learning phase 2."
                )
                print(f"Restoring best checkpoint from: {ckpt_best_smoothed}")
                restored = torch.load(ckpt_best_smoothed, map_location=device)
                model.load_state_dict(restored["model_state_dict"])

                if frozen_stem_grad_hook_handle is not None:
                    frozen_stem_grad_hook_handle.remove()
                    frozen_stem_grad_hook_handle = None
                    frozen_stem_weight = None

                unfrozen_params = _set_encoder_trainable(model, trainable=True)
                optimizer = _make_optimizer(phase_2_lr)
                encoder_is_frozen = False
                es = EarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    warmup=early_stopping_warmup,
                    smoothing_window=early_stopping_smoothing_window,
                    enabled=early_stopping,
                )
                multi_phase_started = True

                train_dataset.augment.p_channel_dropout = 0.2  # Enable some prior dropout here for robustness.

                print(
                    f"Multi-phase learning enabled: encoder unfrozen ({unfrozen_params} parameters), "
                    f"optimizer reset, learning rate set to {phase_2_lr:.2e} "
                    f"({lr_unfreeze:.3f} x base lr {lr:.2e}), early stopping reset."
                    f"\nSetting train augmentation prior dropout to {train_dataset.augment.p_channel_dropout} for phase 2."
                    f"\nContinuing training from epoch {epoch + 1} with restored weights from best checkpoint so far and new optimizer..."
                )
                writer.add_text(
                    "multi_phase_learning/transition",
                    json.dumps(
                        {
                            "trigger_epoch": epoch + 1,
                            "restored_checkpoint": ckpt_best_smoothed,
                            "phase2_lr": phase_2_lr,
                            "lr_unfreeze_multiplier": lr_unfreeze,
                            "encoder_unfrozen_params": unfrozen_params,
                        },
                        indent=2,
                    ),
                    epoch,
                )
                continue

            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}. "
                f"Best smoothed validation loss was {es_info['best_smoothed_val_loss']:.6f} "
                f"at epoch {best_epoch_1based}."
            )
            stopped_early = True
            break

    if restore_best_checkpoint and os.path.exists(ckpt_best_smoothed):
        restored = torch.load(ckpt_best_smoothed, map_location=device)
        model.load_state_dict(restored["model_state_dict"])
        restored_epoch = restored["epoch"]
        print(f"\nRestored best smoothed-loss checkpoint (epoch {restored_epoch}).")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(log_dir, "checkpoint_final.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
            "best_epoch": (es.best_epoch + 1) if es.best_epoch is not None else None,
            "best_raw_val_loss": es.best_raw_val_loss,
            "best_smoothed_val_loss": es.best_smoothed_val_loss,
            "best_val_dice": best_val_dice,
            "stopped_early": stopped_early,
            "freeze_encoder_epochs": freeze_encoder_epochs,
        },
        final_checkpoint_path,
    )
    
    writer.close()
    print(f"Saved final checkpoint to: {final_checkpoint_path}")
    print(f"\n{'=' * 60}")
    print("Training complete!" + (" (early stop)" if stopped_early else ""))
    print(f"Best raw val loss:      {es.best_raw_val_loss:.4f} @ epoch {(es.best_raw_epoch + 1) if es.best_raw_epoch is not None else 'N/A'}")
    print(f"Best smoothed val loss: {es.best_smoothed_val_loss:.6f} @ epoch {(es.best_epoch + 1) if es.best_epoch is not None else 'N/A'}")
    print(f"Best validation DSC:    {best_val_dice:.4f}")
    print(f"{'=' * 60}")

    return {
        "best_checkpoint": ckpt_best_smoothed,
        "best_raw_checkpoint": ckpt_best_raw,
        "last_checkpoint": ckpt_last,
        "final_checkpoint": final_checkpoint_path,
        "log_dir": log_dir,
    }


def run_inference(
    checkpoint_path,
    json_fold_path,
    fold_index,
    mri_data_dir,
    val_prior_dir,
    output_dir=None,
    num_classes=2,
    batch_size=1,
    num_workers=0,
    prior_cache_dir=None,
    amp_enabled=True,
    test_mode=False,
    infer_test_set=False,
    generate_niftis=False,
    nifti_patch_size=128,
    use_gaussian_weighting=True,
    gaussian_sigma_scale=1 / 8,
    gaussian_value_scaling_factor=10.0,
    accumulate_on_cpu=True,
):
    """Run multimodal UNet inference on train/val fold and export JSON summaries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    train_ids, val_ids = load_fold_split(json_fold_path, fold_index)
    test_ids = []
    if infer_test_set:
        with open(json_fold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fold_payload = payload["folds"]
        fold_key = f"fold_{fold_index}"
        test_ids = fold_payload[fold_key].get("test_ids", [])
    if test_mode:
        train_ids = train_ids[:4]
        val_ids = val_ids[:2]
        test_ids = test_ids[:2]

    train_cases = build_cases(train_ids, mri_data_dir)
    val_cases = build_cases(val_ids, mri_data_dir)
    test_cases = build_cases(test_ids, mri_data_dir) if infer_test_set else []

    if prior_cache_dir is None:
        prior_cache_root = os.path.join(val_prior_dir, "_multimodal_prior_cache_infer")
    else:
        prior_cache_root = os.path.join(prior_cache_dir, "infer")
    train_prior_cache_dir = os.path.join(prior_cache_root, "train_from_nifti")
    val_prior_cache_dir = os.path.join(prior_cache_root, "val_from_nifti")
    test_prior_cache_dir = os.path.join(prior_cache_root, "test_from_nifti")

    train_cases = prepare_val_prior_cache_from_niftis(
        val_cases=train_cases,
        val_prior_dir=val_prior_dir,
        out_cache_dir=train_prior_cache_dir,
        split_dir_token="val",
    )
    val_cases = prepare_val_prior_cache_from_niftis(
        val_cases=val_cases,
        val_prior_dir=val_prior_dir,
        out_cache_dir=val_prior_cache_dir,
        split_dir_token="val",
    )
    if infer_test_set:
        test_cases = prepare_val_prior_cache_from_niftis(
            val_cases=test_cases,
            val_prior_dir=val_prior_dir,
            out_cache_dir=test_prior_cache_dir,
            split_dir_token="test",
        )

    train_dataset = UNetWithPriorDataset(
        cases=train_cases,
        eeg_preds_by_id={},
        image_dtype=torch.float16,
        prior_dtype=torch.float16,
        gt_dtype=torch.uint8,
        clamp_min_sigma_vox=1.0,
        fallback_prior="zeros",
        return_float32=True,
        prior_cache_dir=train_prior_cache_dir,
        overwrite_cache=False,
        enable_augmentation=False,
    )
    val_dataset = UNetWithPriorDataset(
        cases=val_cases,
        eeg_preds_by_id={},
        image_dtype=torch.float16,
        prior_dtype=torch.float16,
        gt_dtype=torch.uint8,
        clamp_min_sigma_vox=1.0,
        fallback_prior="zeros",
        return_float32=True,
        prior_cache_dir=val_prior_cache_dir,
        overwrite_cache=False,
        enable_augmentation=False,
    )
    test_dataset = None
    if infer_test_set:
        test_dataset = UNetWithPriorDataset(
            cases=test_cases,
            eeg_preds_by_id={},
            image_dtype=torch.float16,
            prior_dtype=torch.float16,
            gt_dtype=torch.uint8,
            clamp_min_sigma_vox=1.0,
            fallback_prior="zeros",
            return_float32=True,
            prior_cache_dir=test_prior_cache_dir,
            overwrite_cache=False,
            enable_augmentation=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    test_loader = None
    if infer_test_set and test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else 4,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    model = ResEncUNet_3D_with_prior(num_classes=num_classes, use_prior=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    train_case_npz_by_id = {c["id"]: c["npy"] for c in train_cases}
    val_case_npz_by_id = {c["id"]: c["npy"] for c in val_cases}
    test_case_npz_by_id = {c["id"]: c["npy"] for c in test_cases}
    train_prior_npy_by_id = {
        c["id"]: os.path.join(train_prior_cache_dir, f"{c['id']}_prior.npy") for c in train_cases
    }
    val_prior_npy_by_id = {
        c["id"]: os.path.join(val_prior_cache_dir, f"{c['id']}_prior.npy") for c in val_cases
    }
    test_prior_npy_by_id = {
        c["id"]: os.path.join(test_prior_cache_dir, f"{c['id']}_prior.npy") for c in test_cases
    }

    print("\nRunning inference on training set...")
    train_out = infer_predictions(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    print(f"Train summary: {train_out['summary']}")

    print("Running inference on validation set...")
    val_out = infer_predictions(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    print(f"Val summary: {val_out['summary']}")

    test_out = None
    if infer_test_set and test_loader is not None:
        print("Running inference on test set...")
        test_out = infer_predictions(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )
        print(f"Test summary: {test_out['summary']}")

    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    inference_settings = {
        "checkpoint_path": checkpoint_path,
        "json_fold_path": json_fold_path,
        "fold_index": fold_index,
        "mri_data_dir": mri_data_dir,
        "val_prior_dir": val_prior_dir,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prior_cache_dir": prior_cache_dir,
        "amp_enabled": amp_enabled,
        "test_mode": test_mode,
        "infer_test_set": bool(infer_test_set),
        "generate_niftis": generate_niftis,
        "nifti_patch_size": nifti_patch_size,
        "nifti_patch_stride": max(1, int(round(nifti_patch_size * 0.25))),
        "nifti_patch_stride_factor": 0.25,
        "use_gaussian_weighting": bool(use_gaussian_weighting),
        "gaussian_sigma_scale": float(gaussian_sigma_scale),
        "gaussian_value_scaling_factor": float(gaussian_value_scaling_factor),
        "accumulate_on_cpu": bool(accumulate_on_cpu),
    }
    inference_settings_path = os.path.join(output_dir, "inference_settings.json")
    with open(inference_settings_path, "w") as f:
        json.dump(inference_settings, f, indent=2)
    print(f"Saved inference settings to: {inference_settings_path}")

    output_path = os.path.join(output_dir, "predictions.json")
    predictions_payload = {"train": train_out["cases"], "val": val_out["cases"]}
    if infer_test_set and test_out is not None:
        predictions_payload["test"] = test_out["cases"]
    with open(output_path, "w") as f:
        json.dump(predictions_payload, f, indent=2)
    print(f"Saved predictions to: {output_path}")

    metrics_path = os.path.join(output_dir, "validation.json")
    with open(metrics_path, "w") as f:
        json.dump({"summary": val_out["summary"], "cases": val_out["cases"]}, f, indent=2)
    print(f"Saved validation metrics to: {metrics_path}")

    if generate_niftis:
        print("\nGenerating unthresholded prediction NIfTIs for training set...")
        generate_prediction_niftis(
            model=model,
            device=device,
            case_npz_by_id=train_case_npz_by_id,
            prior_npy_by_id=train_prior_npy_by_id,
            output_dir=os.path.join(output_dir, "pred_niftis", "train"),
            amp_enabled=amp_enabled,
            patch_size=nifti_patch_size,
            use_gaussian_weighting=use_gaussian_weighting,
            gaussian_sigma_scale=gaussian_sigma_scale,
            gaussian_value_scaling_factor=gaussian_value_scaling_factor,
            accumulate_on_cpu=accumulate_on_cpu,
        )
        print("Generating unthresholded prediction NIfTIs for validation set...")
        generate_prediction_niftis(
            model=model,
            device=device,
            case_npz_by_id=val_case_npz_by_id,
            prior_npy_by_id=val_prior_npy_by_id,
            output_dir=os.path.join(output_dir, "pred_niftis", "val"),
            amp_enabled=amp_enabled,
            patch_size=nifti_patch_size,
            use_gaussian_weighting=use_gaussian_weighting,
            gaussian_sigma_scale=gaussian_sigma_scale,
            gaussian_value_scaling_factor=gaussian_value_scaling_factor,
            accumulate_on_cpu=accumulate_on_cpu,
        )
        if infer_test_set and test_out is not None:
            print("Generating unthresholded prediction NIfTIs for test set...")
            generate_prediction_niftis(
                model=model,
                device=device,
                case_npz_by_id=test_case_npz_by_id,
                prior_npy_by_id=test_prior_npy_by_id,
                output_dir=os.path.join(output_dir, "pred_niftis", "test"),
                amp_enabled=amp_enabled,
                patch_size=nifti_patch_size,
                use_gaussian_weighting=use_gaussian_weighting,
                gaussian_sigma_scale=gaussian_sigma_scale,
                gaussian_value_scaling_factor=gaussian_value_scaling_factor,
                accumulate_on_cpu=accumulate_on_cpu,
            )

    return train_out, val_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multimodal MRI + EEG prior segmentation model."
    )
    
    # Required paths
    parser.add_argument(
        "--fold_json", type=str, required=True,
        help="Path to JSON with k-fold subject splits."
    )
    parser.add_argument(
        "--mri_data_dir", type=str, required=True,
        help="Directory containing preprocessed MRI data (*.npz files)."
    )
    parser.add_argument(
        "--mri_checkpoint", type=str, required=True,
        help="Path to pre-trained MRI-only nnUNet checkpoint."
    )
    parser.add_argument(
        "--val_prior_dir", type=str, required=True,
        help=(
            "Directory to recursively search for precomputed prior NIfTIs "
            "(<patient_id>_prior.nii.gz) from EEG validation outputs. "
            "All priors used by this script are loaded from these NIfTIs."
        )
    )
    
    # Optional
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which fold to train (0-4)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--log_root", type=str, default="./runs_multimodal",
        help="Root directory for TensorBoard logs."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of DataLoader worker processes."
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of output classes (2 for binary segmentation)."
    )
    parser.add_argument(
        "--prior_cache_dir", type=str, default=None,
        help="Optional directory to cache priors for faster re-initialization."
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Enable automatic mixed precision (default: True). Use --no-amp to disable."
    )
    parser.add_argument(
        "--test_mode", action="store_true",
        help="If set, runs in test mode with limited data."
    )
    parser.add_argument(
        "--skip_inference", action="store_true",
        help="If set, skips inference after training."
    )
    parser.add_argument(
        "--infer_test_set", action="store_true",
        help="If set, run inference on test set in addition to train/val (default: False)."
    )
    parser.add_argument(
        "--generate_niftis", action="store_true",
        help="If set, export unthresholded prediction NIfTIs during inference."
    )
    parser.add_argument(
        "--nifti_patch_size", type=int, default=128,
        help="Patch size for sliding-window NIfTI inference export (default: 128)."
    )
    parser.add_argument("--use_gaussian_weighting", action="store_true", default=True)
    parser.add_argument("--no_gaussian_weighting", action="store_false", dest="use_gaussian_weighting")
    parser.add_argument("--gaussian_sigma_scale", type=float, default=1 / 8)
    parser.add_argument("--gaussian_value_scaling_factor", type=float, default=10.0)
    parser.add_argument("--accumulate_on_cpu", action="store_true", default=True)
    parser.add_argument("--no_accumulate_on_cpu", action="store_false", dest="accumulate_on_cpu")
    parser.add_argument(
        "--restore_best_checkpoint", action=argparse.BooleanOptionalAction, default=True,
        help="Restore best smoothed-loss checkpoint before inference (default: True)."
    )
    parser.add_argument(
        "--early_stopping", action=argparse.BooleanOptionalAction, default=True,
        help="Enable early stopping (default: True). Use --no-early-stopping to disable."
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=20,
        help="Epochs without improvement before stopping (default: 20)."
    )
    parser.add_argument(
        "--early_stopping_min_delta", type=float, default=0.0,
        help="Minimum absolute improvement in smoothed val loss (default: 0.0)."
    )
    parser.add_argument(
        "--early_stopping_warmup", type=int, default=20,
        help="Epochs before early stopping may trigger (default: 20)."
    )
    parser.add_argument(
        "--early_stopping_smoothing_window", type=int, default=5,
        help="Rolling window size for smoothed val loss (default: 5)."
    )
    parser.add_argument(
        "--freeze-encoder-epochs", type=int, default=1000,
        help="Freeze pre-trained MRI encoder weights for first N epochs (default: 1000, way longer than typical training so effectively never in phase 1)."
    )
    parser.add_argument(
        "--multi-phase-learning", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Enable a second training phase after the first early-stopping trigger. "
            "Defaults to True; use --no-multi-phase-learning to disable."
        ),
    )
    parser.add_argument(
        "--lr-unfreeze", type=float, default=0.20,
        help=(
            "Multiplier applied to the base learning rate when phase 2 starts after "
            "early stopping (default: 0.20)."
        ),
    )
    
    args = parser.parse_args()
    
    train_results = train(
        json_fold_path=args.fold_json,
        fold_index=args.fold,
        mri_data_dir=args.mri_data_dir,
        mri_checkpoint_path=args.mri_checkpoint,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_root=args.log_root,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
        prior_cache_dir=args.prior_cache_dir,
        val_prior_dir=args.val_prior_dir,
        amp_enabled=args.amp,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_warmup=args.early_stopping_warmup,
        early_stopping_smoothing_window=args.early_stopping_smoothing_window,
        restore_best_checkpoint=args.restore_best_checkpoint,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        multi_phase_learning=args.multi_phase_learning,
        lr_unfreeze=args.lr_unfreeze,
    )

    if not args.skip_inference and train_results is not None:
        best_ckpt = train_results["best_checkpoint"]
        log_dir = train_results["log_dir"]

        if os.path.exists(best_ckpt):
            print("\n" + "=" * 80)
            print("RUNNING INFERENCE WITH BEST CHECKPOINT")
            print("=" * 80)
            run_inference(
                checkpoint_path=best_ckpt,
                json_fold_path=args.fold_json,
                fold_index=args.fold,
                mri_data_dir=args.mri_data_dir,
                val_prior_dir=args.val_prior_dir,
                output_dir=log_dir,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prior_cache_dir=args.prior_cache_dir,
                amp_enabled=args.amp,
                test_mode=args.test_mode,
                infer_test_set=args.infer_test_set,
                generate_niftis=args.generate_niftis,
                nifti_patch_size=args.nifti_patch_size,
                use_gaussian_weighting=args.use_gaussian_weighting,
                gaussian_sigma_scale=args.gaussian_sigma_scale,
                gaussian_value_scaling_factor=args.gaussian_value_scaling_factor,
                accumulate_on_cpu=args.accumulate_on_cpu,
            )
        else:
            print(f"Warning: Best checkpoint not found at {best_ckpt}")
    elif args.skip_inference:
        print("\nInference skipped (--skip_inference flag set).")
