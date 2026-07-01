"""
training/eeg_spike_mil_mh_training.py

Core EEG spike localization training script (multi-head MIL).

Trains SpikeMILModel with:
  - Encoder:       SpikeEncoder_T_S  (temporal 1D multiscale CNN + GNN spatial mixing)
  - MIL pooling:   attention-weighted or mean pooling over per-patient spike bags
  - Spatial head:  DeconvSpatialHead  (3D deconvolutional prior map — PRIMARY OUTPUT)
  - Aux heads:     heteroscedastic coordinate regression, hemisphere classification,
                   lobe classification  (all optional, disabled by default)

The canonical training command uses:
    --spatial_head deconv             (default)
    --encoder_type t_s_cnn            (default)
    --lambda_coord 0.0                (disable coord head for deconv-only runs)
    --lambda_hemi  0.0                (disable hemi head for deconv-only runs)
    --lambda_lobe  0.0                (disable lobe head for deconv-only runs)

After training, inference is run automatically via run_inference().
For standalone inference see inference/eeg_spike_mil_mh_inference.py.
"""

import os
import csv
import argparse
import time
import hashlib
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from models.eeg import SpikeMILModel
from datasets.eeg import (
    PatientMILSpikeDataset,
    MultiHeadTargetDataset,
    mil_multitask_collate,
    load_split,
    load_multitask_targets,
    find_patient_files,
    LOBE_LABEL_TO_INT,
    HEMI_LABEL_TO_INT,
    LOBE_CLASSES,
)
from util import emit_run_fingerprint, EarlyStopping


MNI_EXTENT_MM = torch.tensor([90.0, 126.0, 72.0], dtype=torch.float32)
# Must match preprocessing/mri/preprocess_mri.py TARGET_SHAPE.
MRI_PREPROC_SHAPE = (160, 192, 160)


def resolve_gaussian_sigma_bounds(
    gaussian_output_space: str,
    gaussian_sigma_min: Optional[float],
    gaussian_sigma_max: Optional[float],
):
    """Resolve sigma bounds using unit-aware defaults and consistency checks."""
    if gaussian_output_space not in {"normalized", "mni_mm"}:
        raise ValueError(
            f"Unsupported gaussian_output_space={gaussian_output_space!r}. "
            "Expected 'normalized' or 'mni_mm'."
        )

    # Unit-aware defaults:
    # - normalized: sigma is in [-1, 1]-scaled coordinate units
    # - mni_mm:     sigma is in physical millimeters
    if gaussian_sigma_min is None:
        gaussian_sigma_min = 0.02 if gaussian_output_space == "normalized" else 2.0
    if gaussian_sigma_max is None:
        gaussian_sigma_max = 0.25 if gaussian_output_space == "normalized" else 100.0

    if gaussian_sigma_min <= 0:
        raise ValueError(f"gaussian_sigma_min must be > 0, got {gaussian_sigma_min}")
    if gaussian_sigma_max <= gaussian_sigma_min:
        raise ValueError(
            "gaussian_sigma_max must be > gaussian_sigma_min, "
            f"got min={gaussian_sigma_min}, max={gaussian_sigma_max}"
        )

    # Guard against accidental mm-like values when output is normalized.
    if gaussian_output_space == "normalized" and gaussian_sigma_max > 1.0:
        raise ValueError(
            "gaussian_sigma_max is too large for normalized coordinates. "
            f"Got {gaussian_sigma_max}; expected values roughly in (0, 1]. "
            "If you intended millimeters, set --gaussian_output_space mni_mm."
        )

    return float(gaussian_sigma_min), float(gaussian_sigma_max)


def norm_attention_entropy(weights, mask=None, eps=1e-8):
    """Compute normalized attention entropy in [0, 1]."""
    if mask is not None:
        weights = weights * mask
        weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = weights / weights_sum
        k = mask.sum(dim=1).float().clamp_min(1.0)
    else:
        b, n = weights.shape
        k = torch.full((b,), float(n), device=weights.device)

    prob = weights.clamp_min(eps)
    h = -(prob * prob.log()).sum(dim=1)
    h_max = k.log()
    norm = h / (h_max + eps)
    norm = torch.where(k <= 1, torch.ones_like(norm), norm)
    return norm


def heteroscedastic_gaussian_nll(mu, log_sigma, target, log_sigma_min=-4.0, log_sigma_max=1.0):
    log_sigma = torch.clamp(log_sigma, min=log_sigma_min, max=log_sigma_max)
    inv_var = torch.exp(-2.0 * log_sigma)
    loss = 0.5 * ((target - mu) ** 2 * inv_var + 2.0 * log_sigma)
    return loss.mean(), log_sigma


def gaussian_mixture_centroid_nll(gaussian_pred, target_coord, output_space="normalized", eps=1e-8):
    """Negative log-likelihood of target centroid under predicted Gaussian mixture."""
    mu = gaussian_pred["mu"]
    sigma = gaussian_pred["sigma"]
    weights = gaussian_pred["weights"]

    if output_space == "mni_mm":
        extent = MNI_EXTENT_MM.to(device=target_coord.device, dtype=target_coord.dtype)
        target_coord = target_coord * extent

    target = target_coord[:, None, :]  # (B, 1, 3)
    diff = target - mu

    if sigma.shape[-1] == 1:
        sigma_sq = sigma.squeeze(-1).pow(2).clamp_min(eps)  # (B, K)
        sq_mahal = diff.pow(2).sum(dim=-1) / sigma_sq
        log_det = mu.shape[-1] * torch.log(sigma.squeeze(-1).clamp_min(eps))
    else:
        sigma_sq = sigma.pow(2).clamp_min(eps)  # (B, K, 3)
        sq_mahal = (diff.pow(2) / sigma_sq).sum(dim=-1)
        log_det = torch.log(sigma.clamp_min(eps)).sum(dim=-1)

    log_component = torch.log(weights.clamp_min(eps)) - 0.5 * sq_mahal - log_det
    loss = -torch.logsumexp(log_component, dim=-1).mean()

    return loss


def gaussian_mixture_metrics(gaussian_pred, target_coord, output_space="normalized"):
    """Compute centroid-based metrics for Gaussian-mixture predictions."""
    mu = gaussian_pred["mu"]
    sigma = gaussian_pred["sigma"]
    weights = gaussian_pred["weights"]

    if output_space == "mni_mm":
        extent = MNI_EXTENT_MM.to(device=target_coord.device, dtype=target_coord.dtype)
        target_coord = target_coord * extent

    target = target_coord[:, None, :]
    dists = torch.linalg.norm(mu - target, dim=-1)  # (B, K)

    max_idx = weights.argmax(dim=-1)
    row_idx = torch.arange(weights.shape[0], device=weights.device)
    dist_top_weight = dists[row_idx, max_idx].mean()
    dist_nearest = dists.min(dim=-1).values.mean()

    expected_mu = (weights[..., None] * mu).sum(dim=1)
    dist_expected = torch.linalg.norm(expected_mu - target_coord, dim=-1).mean()

    sigma_mean = sigma.mean()
    sigma_min = sigma.min()
    sigma_max = sigma.max()

    weight_entropy = -(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
    max_weight = weights.max(dim=-1).values.mean()

    return {
        "dist_top_weight": dist_top_weight,
        "dist_nearest": dist_nearest,
        "dist_expected": dist_expected,
        "sigma_mean": sigma_mean,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "weight_entropy": weight_entropy,
        "max_weight": max_weight,
    }


def validate_gaussian_targets(train_dataset, val_dataset, args):
    """Validate spatial supervision required for the Gaussian-mixture head."""
    if args.spatial_head != "gaussian_mixture":
        return

    if args.gaussian_target in {"mask", "both"}:
        raise ValueError(
            "Gaussian target mode requires mask supervision, but this pipeline only exposes centroid targets "
            "(normalized_mu) via MultiHeadTargetDataset. Use --gaussian_target centroid or extend dataset targets "
            "with aligned mask fields before enabling mask loss."
        )

    stats = {
        "total": 0,
        "with_centroid": 0,
        "with_mask": 0,
        "missing": 0,
        "coord_min": torch.full((3,), float("inf")),
        "coord_max": torch.full((3,), float("-inf")),
        "mask_shapes": set(),
    }

    def _scan(ds, split_name):
        for pid in ds.patient_ids:
            t = ds.target_by_pid[pid]
            stats["total"] += 1

            mu = t.get("mu", None)
            if mu is None:
                stats["missing"] += 1
                continue
            mu = mu.float().view(-1)
            if mu.numel() != 3:
                raise ValueError(f"[{split_name}] Invalid centroid shape for {pid}: expected [3], got {tuple(mu.shape)}")
            if not torch.isfinite(mu).all():
                raise ValueError(f"[{split_name}] Non-finite centroid for {pid}: {mu.tolist()}")

            stats["with_centroid"] += 1
            stats["coord_min"] = torch.minimum(stats["coord_min"], mu)
            stats["coord_max"] = torch.maximum(stats["coord_max"], mu)

            mask = t.get("mask", None)
            if mask is not None:
                if not torch.is_tensor(mask):
                    raise ValueError(f"[{split_name}] mask for {pid} exists but is not a tensor")
                if mask.numel() == 0:
                    raise ValueError(f"[{split_name}] Empty mask tensor for {pid}")
                if not torch.isfinite(mask).all():
                    raise ValueError(f"[{split_name}] Non-finite mask tensor for {pid}")
                stats["with_mask"] += 1
                stats["mask_shapes"].add(tuple(mask.shape))

    _scan(train_dataset, "train")
    _scan(val_dataset, "val")

    if stats["with_centroid"] == 0:
        raise ValueError(
            "No centroid supervision found, but --spatial_head gaussian_mixture was requested. "
            "Provide normalized_mu targets or disable the Gaussian head."
        )

    if args.gaussian_output_space == "normalized":
        if torch.any(stats["coord_min"] < -1.05) or torch.any(stats["coord_max"] > 1.05):
            raise ValueError(
                "Centroid targets are out of expected normalized range [-1, 1], "
                f"observed min={stats['coord_min'].tolist()} max={stats['coord_max'].tolist()}.\n"
                "Full stats: " + json.dumps({k: (v.tolist() if isinstance(v, torch.Tensor) else list(v)) for k, v in stats.items()}, indent=2)
            )

    if args.gaussian_make_heatmap and args.gaussian_heatmap_shape is None:
        raise ValueError(
            "--gaussian_make_heatmap was enabled without --gaussian_heatmap_shape. "
            "Provide a 3D shape, or disable heatmap generation."
        )

    print("Gaussian target validation summary:")
    print(f"  subjects total: {stats['total']}")
    print(f"  with centroids: {stats['with_centroid']}")
    print(f"  with masks: {stats['with_mask']}")
    print(f"  missing spatial targets: {stats['missing']}")
    print(f"  coord range min: {stats['coord_min'].tolist()}")
    print(f"  coord range max: {stats['coord_max'].tolist()}")
    print(f"  mask shapes: {sorted(stats['mask_shapes']) if stats['mask_shapes'] else 'none'}")


def masked_cross_entropy(logits, target, mask=None):
    loss = F.cross_entropy(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def accuracy_from_logits(logits, target, mask=None):
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float()
    if mask is None:
        return correct.mean()
    mask = mask.float()
    return (correct * mask).sum() / mask.sum().clamp_min(1.0)


def _center_crop_or_pad_3d(mask_np: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop/pad a 3D array to target shape (matches preprocess_mri.py behavior)."""
    if mask_np.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {mask_np.shape}")

    in_shape = np.array(mask_np.shape, dtype=int)
    tgt = np.array(target_shape, dtype=int)

    # Crop around center.
    start = np.maximum((in_shape - tgt) // 2, 0)
    end = np.minimum(start + tgt, in_shape)
    cropped = mask_np[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Pad around center.
    out = np.zeros(tuple(tgt.tolist()), dtype=mask_np.dtype)
    c_shape = np.array(cropped.shape, dtype=int)
    out_start = np.maximum((tgt - c_shape) // 2, 0)
    out_end = out_start + c_shape
    out[out_start[0]:out_end[0], out_start[1]:out_end[1], out_start[2]:out_end[2]] = cropped
    return out


def load_deconv_brain_mask(
    mask_path: str,
    output_shape: Tuple[int, int, int],
    device: torch.device,
    preproc_shape: Tuple[int, int, int] = MRI_PREPROC_SHAPE,
) -> torch.Tensor:
    """Load and resize an MNI brain mask to [1, 1, D, H, W] on the target device."""
    if mask_path is None:
        raise ValueError("deconv brain-mask path is required when --deconv_use_brain_mask is enabled")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Brain mask file not found: {mask_path}")

    suffix = os.path.splitext(mask_path)[1].lower()
    if suffix == ".npy":
        mask_np = np.load(mask_path).astype(np.float32)
    elif suffix == ".npz":
        npz = np.load(mask_path, allow_pickle=True)
        try:
            if "mask" in npz:
                mask_np = np.asarray(npz["mask"], dtype=np.float32)
            else:
                keys = list(npz.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ brain mask {mask_path!r} has multiple arrays {keys} and no 'mask' key."
                    )
                mask_np = np.asarray(npz[keys[0]], dtype=np.float32)
        finally:
            npz.close()
    else:
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "nibabel is required to load NIfTI brain masks. Install nibabel or provide .npy/.npz mask."
            ) from exc
        mask_np = np.asarray(nib.load(mask_path).get_fdata(dtype=np.float32), dtype=np.float32)

    mask = torch.from_numpy(mask_np).float()
    while mask.ndim > 3:
        mask = mask.squeeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D brain mask, got shape {tuple(mask.shape)} from {mask_path}")
    if not torch.isfinite(mask).all():
        raise ValueError(f"Non-finite values in brain mask: {mask_path}")

    # Align template-space mask to the MRI preprocessing crop used for npz['gt'].
    mask_np = _center_crop_or_pad_3d(mask.cpu().numpy(), target_shape=tuple(preproc_shape))
    mask = torch.from_numpy(mask_np).float()

    mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    if tuple(mask.shape[-3:]) != tuple(output_shape):
        mask = F.interpolate(mask, size=tuple(output_shape), mode="nearest")
    return mask.to(device)


def validate_deconv_targets(train_dataset, val_dataset, args):
    """Validate spatial mask supervision required for deconv spatial head."""
    if args.spatial_head != "deconv":
        return

    expected_shape = tuple(args.deconv_output_shape)
    stats = {
        "total": 0,
        "with_mask": 0,
        "missing": 0,
        "mask_shapes": set(),
        "voxel_min": None,
        "voxel_max": None,
        "spaces": set(),
    }

    def _scan(ds, split_name):
        for pid in ds.patient_ids:
            stats["total"] += 1
            t = ds.target_by_pid[pid]
            has_npz_mask = False
            if getattr(ds, "deconv_mask_npz_dir", None):
                npz_path = os.path.join(ds.deconv_mask_npz_dir, f"{pid}_preproc.npz")
                has_npz_mask = os.path.exists(npz_path)

            if has_npz_mask:
                space = "mni_npz"
            else:
                space = str(t.get("mask_space", "unknown")).lower()
            stats["spaces"].add(space)

            has_mask = has_npz_mask or (t.get("target_mask") is not None) or (t.get("mask_path") is not None)
            if not has_mask:
                stats["missing"] += 1
                continue

            if (not has_npz_mask) and space in {"", "unknown", "native", "t1", "patient", "subject"}:
                raise ValueError(
                    "Deconv spatial head requires target masks aligned to the decoder output space. "
                    "Native-space masks without transforms are not supported. "
                    f"[{split_name}] patient={pid} mask_space={space!r}."
                )

            mask = ds.get_deconv_target_mask(pid)
            if mask.ndim != 4 or mask.shape[0] != 1:
                raise ValueError(f"[{split_name}] Invalid deconv target mask shape for {pid}: {tuple(mask.shape)}")
            if tuple(mask.shape[-3:]) != expected_shape:
                raise ValueError(
                    f"[{split_name}] Resampled mask for {pid} has shape {tuple(mask.shape[-3:])}, "
                    f"expected {expected_shape}"
                )
            if not torch.isfinite(mask).all():
                raise ValueError(f"[{split_name}] Non-finite values in deconv target mask for {pid}")

            voxels = float(mask.sum().item())
            if voxels <= 0:
                raise ValueError(f"[{split_name}] Empty deconv target mask for {pid}")

            stats["with_mask"] += 1
            stats["mask_shapes"].add(tuple(mask.shape[-3:]))
            stats["voxel_min"] = voxels if stats["voxel_min"] is None else min(stats["voxel_min"], voxels)
            stats["voxel_max"] = voxels if stats["voxel_max"] is None else max(stats["voxel_max"], voxels)

    _scan(train_dataset, "train")
    _scan(val_dataset, "val")

    if stats["with_mask"] == 0:
        raise ValueError(
            "No deconv target masks are available, but --spatial_head deconv was requested. "
            "Provide MRI preprocessed <pid>_preproc.npz files with 'gt' in --mri_npy_dir, "
            "or provide aligned MNI-space masks in targets JSON."
        )

    if args.deconv_use_brain_mask:
        if args.deconv_brain_mask_path is None:
            raise ValueError("--deconv_use_brain_mask is enabled but --deconv_brain_mask_path is missing")
        if not os.path.exists(args.deconv_brain_mask_path):
            raise FileNotFoundError(f"Brain mask path does not exist: {args.deconv_brain_mask_path}")

    print("Deconv target validation summary:")
    print(f"  subjects total: {stats['total']}")
    print(f"  with masks: {stats['with_mask']}")
    print(f"  missing masks: {stats['missing']}")
    print(f"  mask shapes: {sorted(stats['mask_shapes']) if stats['mask_shapes'] else 'none'}")
    print(f"  mask voxel-count range: [{stats['voxel_min']}, {stats['voxel_max']}]")
    print(f"  target spaces: {sorted(stats['spaces'])}")


def soft_dice_loss(prob, target, mask=None, eps=1e-6):
    if mask is not None:
        prob = prob * mask
        target = target * mask
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def masked_bce_with_logits(logits, target, mask=None, pos_weight=None):
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pw,
    )
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1e-6)
    return loss.mean()


def focal_loss_with_logits(logits, target, mask=None, pos_weight=None, gamma=2.0):
    prob = torch.sigmoid(logits)
    bce = masked_bce_with_logits(logits, target, mask=None, pos_weight=pos_weight)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    mod = (1.0 - pt).clamp_min(1e-6).pow(gamma)
    per_voxel = mod * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is not None:
        per_voxel = per_voxel * mask
        return per_voxel.sum() / mask.sum().clamp_min(1e-6)
    return 0.5 * (bce + per_voxel.mean())


def total_variation_3d(prob, mask=None):
    if mask is not None:
        prob = prob * mask
    dx = torch.abs(prob[:, :, 1:, :, :] - prob[:, :, :-1, :, :]).mean()
    dy = torch.abs(prob[:, :, :, 1:, :] - prob[:, :, :, :-1, :]).mean()
    dz = torch.abs(prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1]).mean()
    return dx + dy + dz


def deconv_spatial_entropy_from_logits(logits: torch.Tensor, brain_mask: Optional[torch.Tensor] = None):
    """Compute normalized spatial entropy of deconv probabilities over voxels."""
    prob = torch.sigmoid(logits)
    if brain_mask is not None:
        prob = prob * brain_mask
    p = prob / (prob.sum(dim=(-3, -2, -1), keepdim=True) + 1e-6)
    entropy = -(p * torch.log(p + 1e-6)).sum(dim=(-3, -2, -1)).mean()
    return entropy


def gaussian_blur_3d(x: torch.Tensor, sigma: float):
    """Apply isotropic 3D Gaussian blur to a BCHWD tensor using depthwise conv."""
    if sigma is None or sigma <= 0:
        return x

    radius = max(1, int(3.0 * float(sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g1 = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    g1 = g1 / g1.sum().clamp_min(1e-6)

    g3 = (
        g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    )
    g3 = g3 / g3.sum().clamp_min(1e-6)
    kernel = g3[None, None, :, :, :]

    b, c, _, _, _ = x.shape
    kernel = kernel.repeat(c, 1, 1, 1, 1)
    x_blur = F.conv3d(x, kernel, padding=radius, groups=c)
    return x_blur


def make_soft_deconv_target(target_mask: torch.Tensor, sigma: float):
    """Build a soft [0,1] target map from a binary/float deconv target mask."""
    target = target_mask.float()
    if sigma is not None and sigma > 0:
        target = gaussian_blur_3d(target, sigma=sigma)

    max_val = target.amax(dim=(-3, -2, -1), keepdim=True)
    target = target / (max_val + 1e-6)
    return target.clamp(0.0, 1.0)


def soft_bce_with_logits(logits, soft_target, brain_mask=None):
    loss_vox = F.binary_cross_entropy_with_logits(
        logits,
        soft_target,
        reduction="none",
    )

    if brain_mask is not None:
        loss_vox = loss_vox * brain_mask
        return loss_vox.sum() / (brain_mask.sum() + 1e-6)

    return loss_vox.mean()


def coverage_loss(prob, soft_target, brain_mask=None):
    """Encourage coverage of known target region without punishing distant hotspots directly."""
    if brain_mask is not None:
        prob = prob * brain_mask
        soft_target = soft_target * brain_mask

    numerator = (prob * soft_target).sum(dim=(-3, -2, -1))
    denominator = soft_target.sum(dim=(-3, -2, -1)) + 1e-6

    coverage = numerator / denominator
    loss = -torch.log(coverage + 1e-6)
    return loss.mean()


def mass_loss(prob, brain_mask=None):
    """Penalize excessive total probability mass to avoid whole-brain activation."""
    if brain_mask is not None:
        prob = prob * brain_mask
        return prob.sum() / (brain_mask.sum() + 1e-6)

    return prob.mean()


def effective_volume(prob, brain_mask=None):
    """Return effective spatial support size in voxels from a normalized probability map."""
    if brain_mask is not None:
        prob = prob * brain_mask

    p = prob / (prob.sum(dim=(-3, -2, -1), keepdim=True) + 1e-6)
    eff = 1.0 / ((p ** 2).sum(dim=(-3, -2, -1)) + 1e-6)
    return eff.mean()


def compute_deconv_diagnostics(
    logits,
    target_mask=None,
    brain_mask=None,
    target_blur_sigma=3.0,
):
    """Compute deconv diagnostics for training/validation/inference in one place."""
    prob = torch.sigmoid(logits)

    if brain_mask is not None:
        prob_masked = prob * brain_mask
        denom = brain_mask.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    else:
        prob_masked = prob
        denom = torch.full_like(
            prob[:, :, 0, 0, 0],
            float(prob.shape[-3] * prob.shape[-2] * prob.shape[-1]),
        )

    pred_max = prob_masked.amax(dim=(-3, -2, -1))
    pred_mean = prob_masked.sum(dim=(-3, -2, -1)) / denom

    p = prob_masked / (prob_masked.sum(dim=(-3, -2, -1), keepdim=True) + 1e-6)
    eff_volume = 1.0 / ((p ** 2).sum(dim=(-3, -2, -1)) + 1e-6)

    metrics = {
        "pred_max": pred_max,
        "pred_mean_inside_brain": pred_mean,
        "mass_value": pred_mean,
        "effective_volume_voxels": eff_volume,
    }

    if target_mask is not None:
        soft_target = make_soft_deconv_target(target_mask, sigma=target_blur_sigma)

        if brain_mask is not None:
            target_masked = soft_target * brain_mask
        else:
            target_masked = soft_target

        coverage_value = (
            (prob_masked * target_masked).sum(dim=(-3, -2, -1))
            / (target_masked.sum(dim=(-3, -2, -1)) + 1e-6)
        )

        bce_vox = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
        if brain_mask is not None:
            bce_vox = bce_vox * brain_mask
            bce_per_sample = bce_vox.sum(dim=(-3, -2, -1)) / (brain_mask.sum(dim=(-3, -2, -1)) + 1e-6)
        else:
            bce_per_sample = bce_vox.mean(dim=(-3, -2, -1))

        cov_loss_per_sample = -torch.log(coverage_value + 1e-6)
        mass_loss_per_sample = pred_mean

        metrics.update(
            {
                "coverage_value": coverage_value,
                "soft_bce_loss": bce_per_sample,
                "coverage_loss": cov_loss_per_sample,
                "mass_loss": mass_loss_per_sample,
            }
        )

    return metrics


def compute_deconv_spatial_loss(
    logits,
    target_mask,
    brain_mask=None,
    loss_type="dice_bce",
    target_blur_sigma=3.0,
    bce_weight=1.0,
    coverage_weight=0.2,
    mass_weight=0.01,
    entropy_weight=0.0,
    tv_weight=0.0,
    pos_weight=None,
    outside_brain_penalty_weight=0.01,
):
    if logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits shape [B,1,D,H,W], got {tuple(logits.shape)}")
    if target_mask.ndim != 5 or target_mask.shape[1] != 1:
        raise ValueError(f"Expected deconv target shape [B,1,D,H,W], got {tuple(target_mask.shape)}")
    if logits.shape != target_mask.shape:
        raise ValueError(
            "Expected logits and target_mask to have identical shape, got "
            f"logits={tuple(logits.shape)} target={tuple(target_mask.shape)}"
        )

    if brain_mask is not None:
        if brain_mask.ndim != 5:
            raise ValueError(f"Expected brain_mask [B,1,D,H,W]-broadcastable, got {tuple(brain_mask.shape)}")
        try:
            torch.broadcast_shapes(logits.shape, brain_mask.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"brain_mask shape {tuple(brain_mask.shape)} is not broadcastable to logits shape {tuple(logits.shape)}"
            ) from exc

    prob = torch.sigmoid(logits)
    target_hard = target_mask.float()
    diag = compute_deconv_diagnostics(
        logits=logits,
        target_mask=target_mask,
        brain_mask=brain_mask,
        target_blur_sigma=target_blur_sigma,
    )

    if loss_type == "soft_bce":
        # MultiHeadTargetDataset already applies optional blur + [0,1] normalization
        # via deconv_target_blur_sigma, so soft_bce should consume that soft target
        # directly and avoid double smoothing.
        soft_target = target_hard.clamp(0.0, 1.0)
        loss_vox = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
        if brain_mask is not None:
            loss_vox = loss_vox * brain_mask
            loss = loss_vox.sum() / (brain_mask.sum() + 1e-6)
        else:
            loss = loss_vox.mean()
    elif loss_type == "soft_bce_coverage":
        soft_target = make_soft_deconv_target(target_mask, sigma=target_blur_sigma)

        bce = soft_bce_with_logits(
            logits=logits,
            soft_target=soft_target,
            brain_mask=brain_mask,
        )
        cov = coverage_loss(
            prob=prob,
            soft_target=soft_target,
            brain_mask=brain_mask,
        )
        mass = mass_loss(
            prob=prob,
            brain_mask=brain_mask,
        )

        loss = (
            bce_weight * bce
            + coverage_weight * cov
            + mass_weight * mass
        )
    elif loss_type == "dice":
        dice = soft_dice_loss(prob, target_hard, mask=brain_mask)
        loss = dice
    elif loss_type == "bce":
        bce = masked_bce_with_logits(logits, target_hard, mask=brain_mask, pos_weight=pos_weight)
        loss = bce
    elif loss_type == "dice_bce":
        dice = soft_dice_loss(prob, target_hard, mask=brain_mask)
        bce = masked_bce_with_logits(logits, target_hard, mask=brain_mask, pos_weight=pos_weight)
        loss = 0.5 * (dice + bce)
    elif loss_type == "dice_focal":
        dice = soft_dice_loss(prob, target_hard, mask=brain_mask)
        focal = focal_loss_with_logits(logits, target_hard, mask=brain_mask, pos_weight=pos_weight)
        loss = 0.5 * (dice + focal)
    else:
        raise ValueError(f"Unsupported deconv loss type: {loss_type}")

    if tv_weight > 0:
        loss = loss + tv_weight * total_variation_3d(prob, mask=brain_mask)

    if brain_mask is not None and outside_brain_penalty_weight > 0:
        outside_prob = prob * (1.0 - brain_mask)
        outside_penalty = outside_prob.mean()
        loss = loss + outside_brain_penalty_weight * outside_penalty

    if entropy_weight > 0:
        entropy = deconv_spatial_entropy_from_logits(logits, brain_mask=brain_mask)
        loss = loss - entropy_weight * entropy

    metrics = {
        "deconv_soft_bce": (
            soft_bce_with_logits(logits, make_soft_deconv_target(target_mask, sigma=target_blur_sigma), brain_mask=brain_mask)
            if target_mask is not None else torch.tensor(0.0, device=logits.device)
        ).detach(),
        "deconv_coverage_loss": (
            coverage_loss(prob, make_soft_deconv_target(target_mask, sigma=target_blur_sigma), brain_mask=brain_mask)
            if target_mask is not None else torch.tensor(0.0, device=logits.device)
        ).detach(),
        "deconv_mass_loss": mass_loss(prob, brain_mask=brain_mask).detach(),
        "deconv_total_loss": loss.detach(),
        "deconv_coverage_value": (
            diag["coverage_value"].mean().detach() if "coverage_value" in diag else torch.tensor(0.0, device=logits.device)
        ),
        "deconv_mass_value": diag["mass_value"].mean().detach(),
        "deconv_pred_max": diag["pred_max"].mean().detach(),
        "deconv_pred_mean_inside_brain": diag["pred_mean_inside_brain"].mean().detach(),
        "deconv_effective_volume_voxels": diag["effective_volume_voxels"].mean().detach(),
    }

    return loss, metrics


def deconv_metrics(prob, target, brain_mask=None, eps=1e-8):
    pred = prob
    tgt = target
    pred_mean_inside = pred.mean()
    if brain_mask is not None:
        pred = pred * brain_mask
        tgt = tgt * brain_mask
        pred_mean_inside = (prob * brain_mask).sum() / brain_mask.sum().clamp_min(eps)

    dice = 1.0 - soft_dice_loss(pred, tgt)
    pred_sum = pred.sum().clamp_min(eps)
    mass_in_gt = (pred * tgt).sum() / pred_sum

    b, _, d, h, w = pred.shape
    flat = pred.view(b, -1)
    peak_idx = flat.argmax(dim=-1)
    px = peak_idx // (h * w)
    py = (peak_idx % (h * w)) // w
    pz = peak_idx % w

    tgt_flat = tgt.view(b, -1)
    idx = torch.arange(d * h * w, device=pred.device, dtype=pred.dtype)
    gx = (idx // (h * w)).view(1, -1)
    gy = ((idx % (h * w)) // w).view(1, -1)
    gz = (idx % w).view(1, -1)
    tgt_mass = tgt_flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    cx = (tgt_flat * gx).sum(dim=-1) / tgt_mass.squeeze(-1)
    cy = (tgt_flat * gy).sum(dim=-1) / tgt_mass.squeeze(-1)
    cz = (tgt_flat * gz).sum(dim=-1) / tgt_mass.squeeze(-1)
    peak_dist = torch.sqrt((px.float() - cx) ** 2 + (py.float() - cy) ** 2 + (pz.float() - cz) ** 2).mean()

    topk = min(100, d * h * w)
    topk_idx = torch.topk(flat, k=topk, dim=-1).indices
    topk_hit = (tgt_flat.gather(1, topk_idx) > 0.0).any(dim=1).float().mean()

    outside_mass = torch.tensor(0.0, device=pred.device)
    if brain_mask is not None:
        outside_mass = (prob * (1.0 - brain_mask)).sum() / prob.sum().clamp_min(eps)

    return {
        "dice": dice,
        "mass_in_gt": mass_in_gt,
        "peak_distance": peak_dist,
        "topk_hit": topk_hit,
        "pred_mean": pred_mean_inside,
        "pred_max": pred.max(),
        "outside_brain_mass": outside_mass,
    }


def lesion_center_transversal_slice_idx(target_3d: torch.Tensor) -> int:
    """Return z-index (last axis) at lesion center; fallback to center when empty."""
    if target_3d.ndim != 3:
        raise ValueError(f"Expected target_3d shape [D,H,W], got {tuple(target_3d.shape)}")

    weights = target_3d.float()
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return int(target_3d.shape[-1] // 2)

    z_coords = torch.arange(target_3d.shape[-1], device=weights.device, dtype=weights.dtype)
    z_weights = weights.sum(dim=(0, 1))
    z_center = (z_weights * z_coords).sum() / z_weights.sum().clamp_min(1e-8)
    z_idx = int(torch.round(z_center).item())
    return max(0, min(target_3d.shape[-1] - 1, z_idx))


def heatmap_rgb(slice_2d: torch.Tensor) -> torch.Tensor:
    """Render a 2D tensor in [0, 1] as a simple RGB heatmap for TensorBoard."""
    if slice_2d.ndim != 2:
        raise ValueError(f"Expected 2D slice, got {tuple(slice_2d.shape)}")

    x = slice_2d.float().clamp(0.0, 1.0)

    # Black -> blue -> cyan -> yellow -> red
    r = torch.clamp(2.0 * x - 0.5, 0.0, 1.0)
    g = torch.clamp(2.0 * x, 0.0, 1.0) * torch.clamp(2.0 - 2.0 * x, 0.0, 1.0)
    b = torch.clamp(1.5 - 2.0 * x, 0.0, 1.0)

    return torch.stack([r, g, b], dim=0)


def _extract_batch_to_device(batch_targets, device):
    out = {
        "coord_target": batch_targets["coord_target"].to(device),
        "gaussian_coord_target": batch_targets["coord_target"].to(device),
        "hemi_target": batch_targets["hemi_target"].to(device),
        "lobe_target": batch_targets["lobe_target"].to(device),
        "hemi_mask": batch_targets["hemi_mask"].to(device),
        "lobe_mask": batch_targets["lobe_mask"].to(device),
    }
    if "target_mask" in batch_targets:
        out["target_mask"] = batch_targets["target_mask"].to(device)
    return out


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scaler,
    epoch,
    lambda_coord=1.0,
    lambda_hemi=0.2,
    lambda_lobe=0.2,
    spatial_head="none",
    gaussian_loss_weight=1.0,
    gaussian_target="centroid",
    gaussian_output_space="normalized",
    deconv_loss_weight=1.0,
    deconv_loss="dice_bce",
    deconv_target_blur_sigma=3.0,
    deconv_bce_weight=1.0,
    deconv_coverage_weight=0.2,
    deconv_mass_weight=0.01,
    deconv_entropy_weight=0.0,
    deconv_tv_weight=0.0,
    deconv_pos_weight=None,
    deconv_outside_brain_penalty_weight=0.01,
    deconv_mask_outside_brain=True,
    deconv_brain_mask=None,
    attn_entropy_lambda=0.005,
    sigma_reg_lambda=5e-4,
    gaussian_sigma_reg_lambda=5e-4,
    writer=None,
):
    model.train()
    total_loss = 0.0
    total_coord = 0.0
    total_hemi = 0.0
    total_lobe = 0.0
    total_gaussian = 0.0
    total_deconv = 0.0
    total_euclidean_norm = 0.0
    total_euclidean_mm = 0.0
    total_hemi_acc = 0.0
    total_lobe_acc = 0.0
    total_sigma = 0.0
    total_attn_entropy = 0.0
    total_gm_sigma = 0.0
    total_gm_sigma_min = 0.0
    total_gm_sigma_max = 0.0
    total_gm_entropy = 0.0
    total_gm_max_weight = 0.0
    total_gm_dist_top_weight = 0.0
    total_gm_dist_nearest = 0.0
    total_gm_dist_expected = 0.0
    total_deconv_dice = 0.0
    total_deconv_mass_in_gt = 0.0
    total_deconv_peak_distance = 0.0
    total_deconv_topk_hit = 0.0
    total_deconv_pred_mean = 0.0
    total_deconv_pred_max = 0.0
    total_deconv_outside_brain_mass = 0.0
    total_deconv_target_soft_max = 0.0
    total_deconv_target_soft_mean = 0.0
    total_deconv_entropy = 0.0
    total_deconv_soft_bce = 0.0
    total_deconv_coverage_loss = 0.0
    total_deconv_mass_loss = 0.0
    total_deconv_coverage_value = 0.0
    total_deconv_mass_value = 0.0
    total_deconv_effective_volume = 0.0
    total = 0

    global_step = epoch * len(dataloader)

    for i, (x, mask, batch_targets) in enumerate(dataloader):
        x = x.to(device)
        mask = mask.to(device)
        t = _extract_batch_to_device(batch_targets, device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(x, mask=mask)
            mu = outputs["mu"]
            log_sigma = outputs["log_sigma"]
            hemi_logits = outputs["hemi_logits"]
            lobe_logits = outputs["lobe_logits"]
            attn_weights = outputs["attn_weights"]
            gaussian_mixture = outputs.get("gaussian_mixture")
            deconv_spatial = outputs.get("deconv_spatial")

            loss = torch.tensor(0.0, device=device)

            if mu is not None:
                coord_loss, log_sigma_clamped = heteroscedastic_gaussian_nll(mu, log_sigma, t["coord_target"])
                sigma_reg = log_sigma_clamped.mean()
                loss = loss + lambda_coord * coord_loss + sigma_reg_lambda * sigma_reg
            else:
                coord_loss = torch.tensor(0.0, device=device)
                sigma_reg = torch.tensor(0.0, device=device)

            if hemi_logits is not None:
                hemi_loss = masked_cross_entropy(
                    hemi_logits, t["hemi_target"], mask=t["hemi_mask"]
                )
                loss = loss + lambda_hemi * hemi_loss
            else:
                hemi_loss = torch.tensor(0.0, device=device)

            if lobe_logits is not None:
                lobe_loss = masked_cross_entropy(
                    lobe_logits, t["lobe_target"], mask=t["lobe_mask"]
                )
                loss = loss + lambda_lobe * lobe_loss
            else:
                lobe_loss = torch.tensor(0.0, device=device)

            if spatial_head == "gaussian_mixture" and gaussian_mixture is not None:
                if gaussian_target not in {"centroid", "both"}:
                    raise ValueError(
                        f"Unsupported gaussian_target={gaussian_target!r} for this pipeline. "
                        "Use 'centroid' or 'both' with dataset mask support."
                    )
                gaussian_loss = gaussian_mixture_centroid_nll(
                    gaussian_mixture,
                    t["gaussian_coord_target"],
                    output_space=gaussian_output_space,
                )
                gaussian_sigma_reg = torch.log(gaussian_mixture["sigma"].clamp_min(1e-8)).mean()
                loss = loss + gaussian_loss_weight * gaussian_loss
                loss = loss + gaussian_sigma_reg_lambda * gaussian_sigma_reg
            else:
                gaussian_loss = torch.tensor(0.0, device=device)
                gaussian_sigma_reg = torch.tensor(0.0, device=device)

            if spatial_head == "deconv":
                if deconv_spatial is None:
                    raise ValueError("Model did not return deconv_spatial outputs while --spatial_head deconv is enabled")
                if "target_mask" not in t:
                    raise ValueError("Deconv training requires batch target 'target_mask' with shape [B,1,D,H,W]")
                target_mask = t["target_mask"]
                if target_mask.ndim != 5 or target_mask.shape[1] != 1:
                    raise ValueError(f"Expected target_mask shape [B,1,D,H,W], got {tuple(target_mask.shape)}")

                logits_deconv = deconv_spatial["logits"]

                if tuple(logits_deconv.shape[-3:]) != tuple(target_mask.shape[-3:]):
                    raise ValueError(
                        "Mismatch between deconv output shape and target mask shape: "
                        f"pred={tuple(logits_deconv.shape[-3:])}, target={tuple(target_mask.shape[-3:])}"
                    )

                loss_mask = deconv_brain_mask if deconv_mask_outside_brain else None
                deconv_loss_val, deconv_loss_metrics = compute_deconv_spatial_loss(
                    logits=logits_deconv,
                    target_mask=target_mask,
                    brain_mask=loss_mask,
                    loss_type=deconv_loss,
                    target_blur_sigma=deconv_target_blur_sigma,
                    bce_weight=deconv_bce_weight,
                    coverage_weight=deconv_coverage_weight,
                    mass_weight=deconv_mass_weight,
                    entropy_weight=deconv_entropy_weight,
                    tv_weight=deconv_tv_weight,
                    pos_weight=deconv_pos_weight,
                    outside_brain_penalty_weight=deconv_outside_brain_penalty_weight,
                )
                loss = loss + deconv_loss_weight * deconv_loss_val

                target_soft = target_mask.float().clamp(0.0, 1.0)

                target_soft_max = target_soft.max()
                if loss_mask is not None:
                    target_soft_mean = (target_soft * loss_mask).sum() / loss_mask.sum().clamp_min(1e-6)
                else:
                    target_soft_mean = target_soft.mean()

                deconv_entropy_val = deconv_spatial_entropy_from_logits(logits_deconv, brain_mask=loss_mask)
            else:
                deconv_loss_val = torch.tensor(0.0, device=device)
                deconv_loss_metrics = {
                    "deconv_soft_bce": torch.tensor(0.0, device=device),
                    "deconv_coverage_loss": torch.tensor(0.0, device=device),
                    "deconv_mass_loss": torch.tensor(0.0, device=device),
                    "deconv_coverage_value": torch.tensor(0.0, device=device),
                    "deconv_mass_value": torch.tensor(0.0, device=device),
                    "deconv_effective_volume_voxels": torch.tensor(0.0, device=device),
                    "deconv_pred_max": torch.tensor(0.0, device=device),
                    "deconv_pred_mean_inside_brain": torch.tensor(0.0, device=device),
                }
                target_soft_max = torch.tensor(0.0, device=device)
                target_soft_mean = torch.tensor(0.0, device=device)
                deconv_entropy_val = torch.tensor(0.0, device=device)

            if attn_weights is not None:
                attn_entropy = norm_attention_entropy(attn_weights.float(), mask).mean()
                loss = loss - attn_entropy_lambda * attn_entropy
            else:
                attn_entropy = torch.tensor(0.0, device=device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            if mu is not None:
                diff_norm = mu.detach() - t["coord_target"]
                euclidean_norm = torch.norm(diff_norm, dim=-1).mean()
                extent = MNI_EXTENT_MM.to(device=device, dtype=mu.dtype)
                diff_mm = diff_norm * extent
                euclidean_mm_per_sample = torch.linalg.norm(diff_mm, dim=-1)
                euclidean_mm = euclidean_mm_per_sample.mean()
                sigma_mean = torch.exp(log_sigma.detach()).mean()
            else:
                euclidean_norm = torch.tensor(0.0, device=device)
                euclidean_mm = torch.tensor(0.0, device=device)
                euclidean_mm_per_sample = torch.zeros(x.size(0), device=device)
                sigma_mean = torch.tensor(0.0, device=device)

            hemi_acc = (
                accuracy_from_logits(hemi_logits.detach(), t["hemi_target"], mask=t["hemi_mask"])
                if hemi_logits is not None else torch.tensor(0.0, device=device)
            )
            lobe_acc = (
                accuracy_from_logits(lobe_logits.detach(), t["lobe_target"], mask=t["lobe_mask"])
                if lobe_logits is not None else torch.tensor(0.0, device=device)
            )

            if spatial_head == "gaussian_mixture" and gaussian_mixture is not None:
                gm_metrics = gaussian_mixture_metrics(
                    gaussian_mixture,
                    t["gaussian_coord_target"],
                    output_space=gaussian_output_space,
                )
            else:
                gm_metrics = {
                    "dist_top_weight": torch.tensor(0.0, device=device),
                    "dist_nearest": torch.tensor(0.0, device=device),
                    "dist_expected": torch.tensor(0.0, device=device),
                    "sigma_mean": torch.tensor(0.0, device=device),
                    "sigma_min": torch.tensor(0.0, device=device),
                    "sigma_max": torch.tensor(0.0, device=device),
                    "weight_entropy": torch.tensor(0.0, device=device),
                    "max_weight": torch.tensor(0.0, device=device),
                }

            if spatial_head == "deconv" and deconv_spatial is not None:
                metric_mask = deconv_brain_mask if deconv_mask_outside_brain else None
                dc_metrics = deconv_metrics(
                    deconv_spatial["prob"],
                    t["target_mask"],
                    brain_mask=metric_mask,
                )
            else:
                dc_metrics = {
                    "dice": torch.tensor(0.0, device=device),
                    "mass_in_gt": torch.tensor(0.0, device=device),
                    "peak_distance": torch.tensor(0.0, device=device),
                    "topk_hit": torch.tensor(0.0, device=device),
                    "pred_mean": torch.tensor(0.0, device=device),
                    "pred_max": torch.tensor(0.0, device=device),
                    "outside_brain_mass": torch.tensor(0.0, device=device),
                }

        b = x.size(0)
        total_loss += loss.item() * b
        total_coord += coord_loss.item() * b
        total_hemi += hemi_loss.item() * b
        total_lobe += lobe_loss.item() * b
        total_gaussian += gaussian_loss.item() * b
        total_deconv += deconv_loss_val.item() * b
        total_euclidean_norm += euclidean_norm.item() * b
        total_euclidean_mm += euclidean_mm.item() * b
        total_hemi_acc += hemi_acc.item() * b
        total_lobe_acc += lobe_acc.item() * b
        total_sigma += sigma_mean.item() * b
        total_attn_entropy += attn_entropy.item() * b
        total_gm_sigma += gm_metrics["sigma_mean"].item() * b
        total_gm_sigma_min += gm_metrics["sigma_min"].item() * b
        total_gm_sigma_max += gm_metrics["sigma_max"].item() * b
        total_gm_entropy += gm_metrics["weight_entropy"].item() * b
        total_gm_max_weight += gm_metrics["max_weight"].item() * b
        total_gm_dist_top_weight += gm_metrics["dist_top_weight"].item() * b
        total_gm_dist_nearest += gm_metrics["dist_nearest"].item() * b
        total_gm_dist_expected += gm_metrics["dist_expected"].item() * b
        total_deconv_dice += dc_metrics["dice"].item() * b
        total_deconv_mass_in_gt += dc_metrics["mass_in_gt"].item() * b
        total_deconv_peak_distance += dc_metrics["peak_distance"].item() * b
        total_deconv_topk_hit += dc_metrics["topk_hit"].item() * b
        total_deconv_pred_mean += dc_metrics["pred_mean"].item() * b
        total_deconv_pred_max += dc_metrics["pred_max"].item() * b
        total_deconv_outside_brain_mass += dc_metrics["outside_brain_mass"].item() * b
        total_deconv_target_soft_max += target_soft_max.item() * b
        total_deconv_target_soft_mean += target_soft_mean.item() * b
        total_deconv_entropy += deconv_entropy_val.item() * b
        total_deconv_soft_bce += deconv_loss_metrics["deconv_soft_bce"].item() * b
        total_deconv_coverage_loss += deconv_loss_metrics["deconv_coverage_loss"].item() * b
        total_deconv_mass_loss += deconv_loss_metrics["deconv_mass_loss"].item() * b
        total_deconv_coverage_value += deconv_loss_metrics["deconv_coverage_value"].item() * b
        total_deconv_mass_value += deconv_loss_metrics["deconv_mass_value"].item() * b
        total_deconv_effective_volume += deconv_loss_metrics["deconv_effective_volume_voxels"].item() * b
        total += b

        if writer is not None:
            step = global_step + i
            writer.add_scalar("loss/total_step", loss.item(), step)
            if mu is not None:
                writer.add_scalar("loss/coord_step", coord_loss.item(), step)
                writer.add_scalar("coord/euclidean_mm_step", euclidean_mm.item(), step)
                writer.add_scalar("coord/sigma_step", sigma_mean.item(), step)
            if hemi_logits is not None:
                writer.add_scalar("loss/hemi_step", hemi_loss.item(), step)
                writer.add_scalar("hemi/acc_step", hemi_acc.item(), step)
            if lobe_logits is not None:
                writer.add_scalar("loss/lobe_step", lobe_loss.item(), step)
                writer.add_scalar("lobe/acc_step", lobe_acc.item(), step)
            if spatial_head == "gaussian_mixture" and gaussian_mixture is not None:
                writer.add_scalar("loss/gaussian_step", gaussian_loss.item(), step)
                writer.add_scalar("gaussian/sigma_mean_step", gm_metrics["sigma_mean"].item(), step)
                writer.add_scalar("gaussian/weight_entropy_step", gm_metrics["weight_entropy"].item(), step)
                writer.add_scalar("gaussian/max_weight_step", gm_metrics["max_weight"].item(), step)
                writer.add_scalar("gaussian/dist_top_weight_step", gm_metrics["dist_top_weight"].item(), step)
            if spatial_head == "deconv" and deconv_spatial is not None:
                writer.add_scalar("loss/deconv_step", deconv_loss_val.item(), step)
                writer.add_scalar("deconv_loss/soft_bce", deconv_loss_metrics["deconv_soft_bce"].item(), step)
                writer.add_scalar("deconv_loss/coverage", deconv_loss_metrics["deconv_coverage_loss"].item(), step)
                writer.add_scalar("deconv_loss/mass", deconv_loss_metrics["deconv_mass_loss"].item(), step)
                writer.add_scalar("deconv/pred_mean_step", dc_metrics["pred_mean"].item(), step)
                writer.add_scalar("deconv/pred_max_step", dc_metrics["pred_max"].item(), step)
                writer.add_scalar("deconv/mass_in_gt_step", dc_metrics["mass_in_gt"].item(), step)
                writer.add_scalar("deconv/target_soft_max_step", target_soft_max.item(), step)
                writer.add_scalar("deconv/target_soft_mean_step", target_soft_mean.item(), step)
                writer.add_scalar("deconv/pred_mean_inside_brain", deconv_loss_metrics["deconv_pred_mean_inside_brain"].item(), step)
                writer.add_scalar("deconv/target_soft_mean_inside_brain", target_soft_mean.item(), step)
                writer.add_scalar("deconv/coverage_value", deconv_loss_metrics["deconv_coverage_value"].item(), step)
                writer.add_scalar("deconv/mass_value", deconv_loss_metrics["deconv_mass_value"].item(), step)
                writer.add_scalar("deconv/effective_volume_voxels", deconv_loss_metrics["deconv_effective_volume_voxels"].item(), step)
                if deconv_entropy_weight > 0:
                    writer.add_scalar("deconv/entropy_step", deconv_entropy_val.item(), step)

            if epoch % 10 == 0 and i == 0:
                if attn_weights is not None:
                    writer.add_scalar("attention/train_entropy", attn_entropy, epoch)
                    writer.add_histogram(
                        "attention/train_weights_hist", attn_weights[0].detach().cpu().numpy(), epoch
                    )
                    attn_img = attn_weights[0].detach().cpu()
                    attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)
                    total_elements = attn_img.numel()
                    height = int(torch.sqrt(torch.tensor(total_elements / 2, dtype=torch.float32)).ceil().item())
                    width = height * 2
                    target_elements = width * height
                    padding_needed = target_elements - total_elements
                    if padding_needed > 0:
                        attn_img = F.pad(attn_img, (0, padding_needed))
                    attn_img = attn_img.view(1, height, width)
                    writer.add_image("attention/train_weights_image", attn_img, epoch)

                if mu is not None:
                    mu_cpu = mu.detach().float().cpu()
                    writer.add_histogram("coord/train_pred_x", mu_cpu[:, 0], epoch)
                    writer.add_histogram("coord/train_pred_y", mu_cpu[:, 1], epoch)
                    writer.add_histogram("coord/train_pred_z", mu_cpu[:, 2], epoch)

                    sigma_cpu = torch.exp(log_sigma.detach()).float().cpu()
                    writer.add_histogram("coord/train_sigma_x", sigma_cpu[:, 0], epoch)
                    writer.add_histogram("coord/train_sigma_y", sigma_cpu[:, 1], epoch)
                    writer.add_histogram("coord/train_sigma_z", sigma_cpu[:, 2], epoch)

                    writer.add_histogram(
                        "coord/train_euclidean_mm_hist",
                        euclidean_mm_per_sample.detach().float().cpu(),
                        epoch,
                    )

                if spatial_head == "deconv" and deconv_spatial is not None:
                    pred_3d = deconv_spatial["prob"][0, 0].detach().float().cpu()
                    tgt_3d = t["target_mask"][0, 0].detach().float().cpu()
                    z_idx = lesion_center_transversal_slice_idx(tgt_3d)
                    pred_slice = torch.flip(pred_3d[:, :, z_idx].transpose(0, 1), dims=(0,))
                    tgt_slice = torch.flip(tgt_3d[:, :, z_idx].transpose(0, 1), dims=(0,))
                    writer.add_image("deconv/train_pred_axial", heatmap_rgb(pred_slice), epoch)
                    writer.add_image("deconv/train_target_axial", heatmap_rgb(tgt_slice), epoch)
                    if deconv_brain_mask is not None:
                        bm_3d = deconv_brain_mask[0, 0].detach().float().cpu()
                        writer.add_image("deconv/train_brain_mask_axial", torch.flip(bm_3d[:, :, z_idx].transpose(0, 1), dims=(0,)).unsqueeze(0), epoch)

    return {
        "loss": total_loss / total,
        "coord_loss": total_coord / total, "coord_euclidean_norm": total_euclidean_norm / total,
        "coord_euclidean_mm": total_euclidean_mm / total, "sigma": total_sigma / total,
        "hemi_loss": total_hemi / total, "hemi_acc": total_hemi_acc / total,
        "lobe_loss": total_lobe / total, "lobe_acc": total_lobe_acc / total,
        "attn_entropy": total_attn_entropy / total,
        "gaussian_loss": total_gaussian / total,
        "gaussian_sigma": total_gm_sigma / total, "gaussian_sigma_min": total_gm_sigma_min / total,
        "gaussian_sigma_max": total_gm_sigma_max / total,
        "gaussian_weight_entropy": total_gm_entropy / total, "gaussian_max_weight": total_gm_max_weight / total,
        "gaussian_dist_top_weight": total_gm_dist_top_weight / total,
        "gaussian_dist_nearest": total_gm_dist_nearest / total,
        "gaussian_dist_expected": total_gm_dist_expected / total,
        "deconv_loss": total_deconv / total, "deconv_dice": total_deconv_dice / total,
        "deconv_mass_in_gt": total_deconv_mass_in_gt / total,
        "deconv_peak_distance": total_deconv_peak_distance / total, "deconv_topk_hit": total_deconv_topk_hit / total,
        "deconv_pred_mean": total_deconv_pred_mean / total, "deconv_pred_max": total_deconv_pred_max / total,
        "deconv_outside_brain_mass": total_deconv_outside_brain_mass / total,
        "deconv_target_soft_max": total_deconv_target_soft_max / total,
        "deconv_target_soft_mean": total_deconv_target_soft_mean / total,
        "deconv_entropy": total_deconv_entropy / total,
        "deconv_soft_bce": total_deconv_soft_bce / total,
        "deconv_coverage_loss": total_deconv_coverage_loss / total,
        "deconv_mass_loss": total_deconv_mass_loss / total,
        "deconv_coverage_value": total_deconv_coverage_value / total,
        "deconv_mass_value": total_deconv_mass_value / total,
        "deconv_effective_volume_voxels": total_deconv_effective_volume / total,
    }


@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    epoch,
    lambda_coord=1.0,
    lambda_hemi=0.2,
    lambda_lobe=0.2,
    spatial_head="none",
    gaussian_loss_weight=1.0,
    gaussian_target="centroid",
    gaussian_output_space="normalized",
    gaussian_sigma_reg_lambda=5e-4,
    deconv_loss_weight=1.0,
    deconv_loss="dice_bce",
    deconv_target_blur_sigma=3.0,
    deconv_bce_weight=1.0,
    deconv_coverage_weight=0.2,
    deconv_mass_weight=0.01,
    deconv_entropy_weight=0.0,
    deconv_tv_weight=0.0,
    deconv_pos_weight=None,
    deconv_outside_brain_penalty_weight=0.01,
    deconv_mask_outside_brain=True,
    deconv_brain_mask=None,
    deconv_val_image_log_every=20,
    writer=None,
):
    model.eval()
    total_loss = 0.0
    total_coord = 0.0
    total_hemi = 0.0
    total_lobe = 0.0
    total_gaussian = 0.0
    total_deconv = 0.0
    total_euclidean_norm = 0.0
    total_euclidean_mm = 0.0
    total_hemi_acc = 0.0
    total_lobe_acc = 0.0
    total_sigma = 0.0
    total_attn_entropy = 0.0
    total_gm_sigma = 0.0
    total_gm_sigma_min = 0.0
    total_gm_sigma_max = 0.0
    total_gm_entropy = 0.0
    total_gm_max_weight = 0.0
    total_gm_dist_top_weight = 0.0
    total_gm_dist_nearest = 0.0
    total_gm_dist_expected = 0.0
    total_deconv_dice = 0.0
    total_deconv_mass_in_gt = 0.0
    total_deconv_peak_distance = 0.0
    total_deconv_topk_hit = 0.0
    total_deconv_pred_mean = 0.0
    total_deconv_pred_max = 0.0
    total_deconv_outside_brain_mass = 0.0
    total_deconv_target_soft_max = 0.0
    total_deconv_target_soft_mean = 0.0
    total_deconv_entropy = 0.0
    total_deconv_soft_bce = 0.0
    total_deconv_coverage_loss = 0.0
    total_deconv_mass_loss = 0.0
    total_deconv_coverage_value = 0.0
    total_deconv_mass_value = 0.0
    total_deconv_effective_volume = 0.0
    total = 0

    for i, (x, mask, batch_targets) in enumerate(dataloader):
        x = x.to(device)
        mask = mask.to(device)
        t = _extract_batch_to_device(batch_targets, device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(x, mask=mask)
            mu = outputs["mu"]
            log_sigma = outputs["log_sigma"]
            hemi_logits = outputs["hemi_logits"]
            lobe_logits = outputs["lobe_logits"]
            attn_weights = outputs["attn_weights"]
            gaussian_mixture = outputs.get("gaussian_mixture")
            deconv_spatial = outputs.get("deconv_spatial")

            loss = torch.tensor(0.0, device=device)

            if mu is not None:
                coord_loss, _ = heteroscedastic_gaussian_nll(mu, log_sigma, t["coord_target"])
                loss = loss + lambda_coord * coord_loss
            else:
                coord_loss = torch.tensor(0.0, device=device)

            if hemi_logits is not None:
                hemi_loss = masked_cross_entropy(
                    hemi_logits, t["hemi_target"], mask=t["hemi_mask"]
                )
                loss = loss + lambda_hemi * hemi_loss
            else:
                hemi_loss = torch.tensor(0.0, device=device)

            if lobe_logits is not None:
                lobe_loss = masked_cross_entropy(
                    lobe_logits, t["lobe_target"], mask=t["lobe_mask"]
                )
                loss = loss + lambda_lobe * lobe_loss
            else:
                lobe_loss = torch.tensor(0.0, device=device)

            if spatial_head == "gaussian_mixture" and gaussian_mixture is not None:
                if gaussian_target not in {"centroid", "both"}:
                    raise ValueError(
                        f"Unsupported gaussian_target={gaussian_target!r} for this pipeline. "
                        "Use 'centroid' or 'both' with dataset mask support."
                    )
                gaussian_loss = gaussian_mixture_centroid_nll(
                    gaussian_mixture,
                    t["gaussian_coord_target"],
                    output_space=gaussian_output_space,
                )
                loss = loss + gaussian_loss_weight * gaussian_loss
                gaussian_sigma_reg = torch.log(gaussian_mixture["sigma"].clamp_min(1e-8)).mean()
                loss = loss + gaussian_sigma_reg_lambda * gaussian_sigma_reg
            else:
                gaussian_loss = torch.tensor(0.0, device=device)
                gaussian_sigma_reg = torch.tensor(0.0, device=device)

            if spatial_head == "deconv":
                if deconv_spatial is None:
                    raise ValueError("Model did not return deconv_spatial outputs while --spatial_head deconv is enabled")
                if "target_mask" not in t:
                    raise ValueError("Deconv validation requires batch target 'target_mask' with shape [B,1,D,H,W]")

                logits_deconv = deconv_spatial["logits"]
                prob_deconv = deconv_spatial["prob"]
                target_mask = t["target_mask"]

                if tuple(logits_deconv.shape[-3:]) != tuple(target_mask.shape[-3:]):
                    raise ValueError(
                        "Mismatch between deconv output shape and target mask shape: "
                        f"pred={tuple(logits_deconv.shape[-3:])}, target={tuple(target_mask.shape[-3:])}"
                    )

                loss_mask = deconv_brain_mask if deconv_mask_outside_brain else None
                deconv_loss_val, deconv_loss_metrics = compute_deconv_spatial_loss(
                    logits=logits_deconv,
                    target_mask=target_mask,
                    brain_mask=loss_mask,
                    loss_type=deconv_loss,
                    target_blur_sigma=deconv_target_blur_sigma,
                    bce_weight=deconv_bce_weight,
                    coverage_weight=deconv_coverage_weight,
                    mass_weight=deconv_mass_weight,
                    entropy_weight=deconv_entropy_weight,
                    tv_weight=deconv_tv_weight,
                    pos_weight=deconv_pos_weight,
                    outside_brain_penalty_weight=deconv_outside_brain_penalty_weight,
                )
                loss = loss + deconv_loss_weight * deconv_loss_val

                target_soft = target_mask.float().clamp(0.0, 1.0)

                target_soft_max = target_soft.max()
                if loss_mask is not None:
                    target_soft_mean = (target_soft * loss_mask).sum() / loss_mask.sum().clamp_min(1e-6)
                else:
                    target_soft_mean = target_soft.mean()

                deconv_entropy_val = deconv_spatial_entropy_from_logits(logits_deconv, brain_mask=loss_mask)
            else:
                deconv_loss_val = torch.tensor(0.0, device=device)
                deconv_loss_metrics = {
                    "deconv_soft_bce": torch.tensor(0.0, device=device),
                    "deconv_coverage_loss": torch.tensor(0.0, device=device),
                    "deconv_mass_loss": torch.tensor(0.0, device=device),
                    "deconv_coverage_value": torch.tensor(0.0, device=device),
                    "deconv_mass_value": torch.tensor(0.0, device=device),
                    "deconv_effective_volume_voxels": torch.tensor(0.0, device=device),
                }
                target_soft_max = torch.tensor(0.0, device=device)
                target_soft_mean = torch.tensor(0.0, device=device)
                deconv_entropy_val = torch.tensor(0.0, device=device)

            if attn_weights is not None:
                attn_entropy = norm_attention_entropy(attn_weights.float(), mask).mean()
            else:
                attn_entropy = torch.tensor(0.0, device=device)

        if mu is not None:
            diff_norm = mu.detach() - t["coord_target"]
            euclidean_norm = torch.norm(diff_norm, dim=-1).mean()
            extent = MNI_EXTENT_MM.to(device=device, dtype=mu.dtype)
            diff_mm = diff_norm * extent
            euclidean_mm_per_sample = torch.linalg.norm(diff_mm, dim=-1)
            euclidean_mm = euclidean_mm_per_sample.mean()
            sigma_mean = torch.exp(log_sigma.detach()).mean()
        else:
            euclidean_norm = torch.tensor(0.0, device=device)
            euclidean_mm = torch.tensor(0.0, device=device)
            euclidean_mm_per_sample = torch.zeros(x.size(0), device=device)
            sigma_mean = torch.tensor(0.0, device=device)

        hemi_acc = (
            accuracy_from_logits(hemi_logits.detach(), t["hemi_target"], mask=t["hemi_mask"])
            if hemi_logits is not None else torch.tensor(0.0, device=device)
        )
        lobe_acc = (
            accuracy_from_logits(lobe_logits.detach(), t["lobe_target"], mask=t["lobe_mask"])
            if lobe_logits is not None else torch.tensor(0.0, device=device)
        )

        if spatial_head == "gaussian_mixture" and gaussian_mixture is not None:
            gm_metrics = gaussian_mixture_metrics(
                gaussian_mixture,
                t["gaussian_coord_target"],
                output_space=gaussian_output_space,
            )
        else:
            gm_metrics = {
                "dist_top_weight": torch.tensor(0.0, device=device),
                "dist_nearest": torch.tensor(0.0, device=device),
                "dist_expected": torch.tensor(0.0, device=device),
                "sigma_mean": torch.tensor(0.0, device=device),
                "sigma_min": torch.tensor(0.0, device=device),
                "sigma_max": torch.tensor(0.0, device=device),
                "weight_entropy": torch.tensor(0.0, device=device),
                "max_weight": torch.tensor(0.0, device=device),
            }

        if spatial_head == "deconv" and deconv_spatial is not None:
            metric_mask = deconv_brain_mask if deconv_mask_outside_brain else None
            dc_metrics = deconv_metrics(
                deconv_spatial["prob"],
                t["target_mask"],
                brain_mask=metric_mask,
            )
        else:
            dc_metrics = {
                "dice": torch.tensor(0.0, device=device),
                "mass_in_gt": torch.tensor(0.0, device=device),
                "peak_distance": torch.tensor(0.0, device=device),
                "topk_hit": torch.tensor(0.0, device=device),
                "pred_mean": torch.tensor(0.0, device=device),
                "pred_max": torch.tensor(0.0, device=device),
                "outside_brain_mass": torch.tensor(0.0, device=device),
            }

        b = x.size(0)
        total_loss += loss.item() * b
        total_coord += coord_loss.item() * b
        total_hemi += hemi_loss.item() * b
        total_lobe += lobe_loss.item() * b
        total_gaussian += gaussian_loss.item() * b
        total_deconv += deconv_loss_val.item() * b
        total_euclidean_norm += euclidean_norm.item() * b
        total_euclidean_mm += euclidean_mm.item() * b
        total_hemi_acc += hemi_acc.item() * b
        total_lobe_acc += lobe_acc.item() * b
        total_sigma += sigma_mean.item() * b
        total_attn_entropy += attn_entropy.item() * b
        total_gm_sigma += gm_metrics["sigma_mean"].item() * b
        total_gm_sigma_min += gm_metrics["sigma_min"].item() * b
        total_gm_sigma_max += gm_metrics["sigma_max"].item() * b
        total_gm_entropy += gm_metrics["weight_entropy"].item() * b
        total_gm_max_weight += gm_metrics["max_weight"].item() * b
        total_gm_dist_top_weight += gm_metrics["dist_top_weight"].item() * b
        total_gm_dist_nearest += gm_metrics["dist_nearest"].item() * b
        total_gm_dist_expected += gm_metrics["dist_expected"].item() * b
        total_deconv_dice += dc_metrics["dice"].item() * b
        total_deconv_mass_in_gt += dc_metrics["mass_in_gt"].item() * b
        total_deconv_peak_distance += dc_metrics["peak_distance"].item() * b
        total_deconv_topk_hit += dc_metrics["topk_hit"].item() * b
        total_deconv_pred_mean += dc_metrics["pred_mean"].item() * b
        total_deconv_pred_max += dc_metrics["pred_max"].item() * b
        total_deconv_outside_brain_mass += dc_metrics["outside_brain_mass"].item() * b
        total_deconv_target_soft_max += target_soft_max.item() * b
        total_deconv_target_soft_mean += target_soft_mean.item() * b
        total_deconv_entropy += deconv_entropy_val.item() * b
        total_deconv_soft_bce += deconv_loss_metrics["deconv_soft_bce"].item() * b
        total_deconv_coverage_loss += deconv_loss_metrics["deconv_coverage_loss"].item() * b
        total_deconv_mass_loss += deconv_loss_metrics["deconv_mass_loss"].item() * b
        total_deconv_coverage_value += deconv_loss_metrics["deconv_coverage_value"].item() * b
        total_deconv_mass_value += deconv_loss_metrics["deconv_mass_value"].item() * b
        total_deconv_effective_volume += deconv_loss_metrics["deconv_effective_volume_voxels"].item() * b
        total += b

        if writer is not None and epoch % 10 == 0 and i == 0:
            if attn_weights is not None:
                writer.add_scalar("attention/val_entropy", attn_entropy, epoch)
                writer.add_histogram(
                    "attention/val_weights_hist", attn_weights[0].detach().cpu().numpy(), epoch
                )
                attn_img = attn_weights[0].detach().cpu()
                attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)
                total_elements = attn_img.numel()
                height = int(torch.sqrt(torch.tensor(total_elements / 2, dtype=torch.float32)).ceil().item())
                width = height * 2
                target_elements = width * height
                padding_needed = target_elements - total_elements
                if padding_needed > 0:
                    attn_img = F.pad(attn_img, (0, padding_needed))
                attn_img = attn_img.view(1, height, width)
                writer.add_image("attention/val_weights_image", attn_img, epoch)

            if mu is not None:
                mu_cpu = mu.detach().float().cpu()
                writer.add_histogram("coord/val_pred_x", mu_cpu[:, 0], epoch)
                writer.add_histogram("coord/val_pred_y", mu_cpu[:, 1], epoch)
                writer.add_histogram("coord/val_pred_z", mu_cpu[:, 2], epoch)

                sigma_cpu = torch.exp(log_sigma.detach()).float().cpu()
                writer.add_histogram("coord/val_sigma_x", sigma_cpu[:, 0], epoch)
                writer.add_histogram("coord/val_sigma_y", sigma_cpu[:, 1], epoch)
                writer.add_histogram("coord/val_sigma_z", sigma_cpu[:, 2], epoch)

                writer.add_histogram(
                    "coord/val_euclidean_mm_hist",
                    euclidean_mm_per_sample.detach().float().cpu(),
                    epoch,
                )

        should_log_val_deconv_image = (
            writer is not None
            and i == 0
            and spatial_head == "deconv"
            and deconv_spatial is not None
            and deconv_val_image_log_every is not None
            and deconv_val_image_log_every > 0
            and ((epoch + 1) % deconv_val_image_log_every == 0)
        )
        if should_log_val_deconv_image:
            pred_3d = deconv_spatial["prob"][0, 0].detach().float().cpu()
            tgt_3d = t["target_mask"][0, 0].detach().float().cpu()
            z_idx = lesion_center_transversal_slice_idx(tgt_3d)
            pred_slice = torch.flip(pred_3d[:, :, z_idx].transpose(0, 1), dims=(0,))
            tgt_slice = torch.flip(tgt_3d[:, :, z_idx].transpose(0, 1), dims=(0,))
            writer.add_image("deconv/val_pred_axial", heatmap_rgb(pred_slice), epoch)
            writer.add_image("deconv/val_target_axial", heatmap_rgb(tgt_slice), epoch)
            if deconv_brain_mask is not None:
                bm_3d = deconv_brain_mask[0, 0].detach().float().cpu()
                writer.add_image("deconv/val_brain_mask_axial", torch.flip(bm_3d[:, :, z_idx].transpose(0, 1), dims=(0,)).unsqueeze(0), epoch)

    return {
        "loss": total_loss / total,
        "coord_loss": total_coord / total, "coord_euclidean_norm": total_euclidean_norm / total,
        "coord_euclidean_mm": total_euclidean_mm / total, "sigma": total_sigma / total,
        "hemi_loss": total_hemi / total, "hemi_acc": total_hemi_acc / total,
        "lobe_loss": total_lobe / total, "lobe_acc": total_lobe_acc / total,
        "attn_entropy": total_attn_entropy / total,
        "gaussian_loss": total_gaussian / total,
        "gaussian_sigma": total_gm_sigma / total, "gaussian_sigma_min": total_gm_sigma_min / total,
        "gaussian_sigma_max": total_gm_sigma_max / total,
        "gaussian_weight_entropy": total_gm_entropy / total, "gaussian_max_weight": total_gm_max_weight / total,
        "gaussian_dist_top_weight": total_gm_dist_top_weight / total,
        "gaussian_dist_nearest": total_gm_dist_nearest / total,
        "gaussian_dist_expected": total_gm_dist_expected / total,
        "deconv_loss": total_deconv / total, "deconv_dice": total_deconv_dice / total,
        "deconv_mass_in_gt": total_deconv_mass_in_gt / total,
        "deconv_peak_distance": total_deconv_peak_distance / total, "deconv_topk_hit": total_deconv_topk_hit / total,
        "deconv_pred_mean": total_deconv_pred_mean / total, "deconv_pred_max": total_deconv_pred_max / total,
        "deconv_outside_brain_mass": total_deconv_outside_brain_mass / total,
        "deconv_target_soft_max": total_deconv_target_soft_max / total,
        "deconv_target_soft_mean": total_deconv_target_soft_mean / total,
        "deconv_entropy": total_deconv_entropy / total,
        "deconv_soft_bce": total_deconv_soft_bce / total,
        "deconv_coverage_loss": total_deconv_coverage_loss / total,
        "deconv_mass_loss": total_deconv_mass_loss / total,
        "deconv_coverage_value": total_deconv_coverage_value / total,
        "deconv_mass_value": total_deconv_mass_value / total,
        "deconv_effective_volume_voxels": total_deconv_effective_volume / total,
    }


def run_smoke_test(model, train_loader, device):
    """Run a quick forward/backward pass on one batch."""
    model.train()

    x = torch.randn(2, 50, 21, 384, device=device)
    mask = torch.ones(2, 50, device=device)
    out = model(x, mask=mask)
    if out["mu"] is not None:
        assert out["mu"].shape == (2, 3)
        assert out["log_sigma"].shape == (2, 3)
    if out["hemi_logits"] is not None:
        assert out["hemi_logits"].shape[0] == 2
    if out["lobe_logits"] is not None:
        assert out["lobe_logits"].shape[0] == 2
    if out["attn_weights"] is not None:
        assert out["attn_weights"].shape == (2, 50)
    if out.get("gaussian_mixture") is not None:
        gm = out["gaussian_mixture"]
        assert gm["mu"].ndim == 3
        assert gm["mu"].shape[-1] == 3
        assert torch.isfinite(gm["mu"]).all()
        assert torch.isfinite(gm["sigma"]).all()
        assert torch.isfinite(gm["weights"]).all()
        assert torch.allclose(
            gm["weights"].sum(dim=-1),
            torch.ones_like(gm["weights"].sum(dim=-1)),
            atol=1e-4,
        )
    if out.get("deconv_spatial") is not None:
        deconv_pred = out["deconv_spatial"]
        logits = deconv_pred["logits"]
        prob = deconv_pred["prob"]
        assert logits.ndim == 5 and logits.shape[1] == 1
        assert prob.ndim == 5 and prob.shape[1] == 1
        assert torch.isfinite(logits).all()
        assert torch.isfinite(prob).all()
        assert float(prob.min()) >= 0.0
        assert float(prob.max()) <= 1.0

    xb, mb, tb = next(iter(train_loader))
    xb = xb.to(device)
    mb = mb.to(device)
    t = _extract_batch_to_device(tb, device)

    outputs = model(xb, mask=mb)
    loss = torch.tensor(0.0, device=device)
    if outputs["mu"] is not None:
        coord_loss, _ = heteroscedastic_gaussian_nll(outputs["mu"], outputs["log_sigma"], t["coord_target"])
        loss = loss + coord_loss
    if outputs["hemi_logits"] is not None:
        loss = loss + 0.2 * masked_cross_entropy(outputs["hemi_logits"], t["hemi_target"], t["hemi_mask"])
    if outputs["lobe_logits"] is not None:
        loss = loss + 0.2 * masked_cross_entropy(outputs["lobe_logits"], t["lobe_target"], t["lobe_mask"])
    if outputs.get("gaussian_mixture") is not None:
        gm = outputs["gaussian_mixture"]
        assert torch.isfinite(gm["mu"]).all()
        assert torch.isfinite(gm["sigma"]).all()
        assert torch.isfinite(gm["weights"]).all()
        loss = loss + gaussian_mixture_centroid_nll(gm, t["gaussian_coord_target"])
    if outputs.get("deconv_spatial") is not None:
        if "target_mask" not in t:
            raise ValueError("Deconv smoke test requires target_mask in the batch targets")
        target_mask = t["target_mask"]
        assert target_mask.ndim == 5 and target_mask.shape[1] == 1
        assert torch.isfinite(target_mask).all()
        assert float(target_mask.max()) > 0.0

        logits = outputs["deconv_spatial"]["logits"]
        prob = outputs["deconv_spatial"]["prob"]
        assert tuple(logits.shape[-3:]) == tuple(target_mask.shape[-3:])
        assert tuple(prob.shape[-3:]) == tuple(target_mask.shape[-3:])
        deconv_loss_val, _ = compute_deconv_spatial_loss(
            logits=logits,
            target_mask=target_mask,
            brain_mask=None,
            loss_type="dice_bce",
        )
        loss = loss + deconv_loss_val
    loss.backward()

    print("Smoke test passed: forward/backward successful.")


def load_pretrained_encoder(model, checkpoint_path, freeze_mode="none"):
    """
    Load pretrained encoder weights from an ``encoder_best.pt`` checkpoint
    (saved by ``eeg_spike_encoder_pretraining.py``) into *model*.

    Parameters
    ----------
    model : nn.Module
        MIL model whose ``model.encoder`` sub-module will receive the weights.
    checkpoint_path : str
        Path to ``encoder_best.pt``.
    freeze_mode : {"none", "full", "temporal"}
        - ``"none"``     : weights are loaded; all parameters remain trainable.
        - ``"full"``     : the entire encoder is frozen.
        - ``"temporal"`` : only sub-modules of the encoder whose name contains
                           ``"temporal"`` are frozen (e.g. ``temporal_cnn`` /
                           ``temporal_projection`` in ``SpikeEncoder_T_S``).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    enc_state = ckpt.get("encoder_state_dict")
    if enc_state is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path!r} has no 'encoder_state_dict' key. "
            "Make sure it was saved by eeg_spike_encoder_pretraining.py."
        )
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=True)
    if missing:
        print(f"  [pretrained encoder] Missing keys: {missing}")
    if unexpected:
        print(f"  [pretrained encoder] Unexpected keys: {unexpected}")
    print(
        f"  Loaded pretrained encoder weights from {checkpoint_path!r} "
        f"(freeze_mode={freeze_mode!r})"
    )

    if freeze_mode == "full":
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder fully frozen.")

    elif freeze_mode == "temporal":
        frozen = []
        for name, module in model.encoder.named_children():
            if "temporal" in name:
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(name)
        if frozen:
            print(f"  Encoder temporal sub-modules frozen: {frozen}")
        else:
            print(
                f"  Warning: freeze_mode='temporal' but no sub-module named "
                f"'temporal*' found in {type(model.encoder).__name__}. "
                "Nothing was frozen."
            )


def train(
    json_split_path,
    fold_index,
    data_dir,
    json_targets_path,
    lobe_json_path=None,
    hemi_json_path=None,
    in_channels=21,
    max_spikes=32,
    min_spikes_per_patient=64,
    batch_size=4,
    emb_dim=None,
    hidden=None,
    dropout=None,
    lr=1e-4,
    weight_decay=1e-4,
    epochs=50,
    log_root="./runs",
    num_workers=0,
    test_mode=False,
    encoder_type=None,
    pooling=None,
    lambda_coord=1.0,
    lambda_hemi=0.2,
    lambda_lobe=0.2,
    spatial_head="none",
    num_gaussians=3,
    gaussian_coord_dim=3,
    gaussian_sigma_min=None,
    gaussian_sigma_max=None,
    gaussian_isotropic=True,
    gaussian_output_space="normalized",
    gaussian_make_heatmap=False,
    gaussian_heatmap_shape=None,
    gaussian_loss_weight=1.0,
    gaussian_target="centroid",
    gaussian_target_blur_sigma=5.0,
    deconv_output_shape=(32, 40, 32),
    deconv_latent_shape=(4, 5, 4),
    deconv_base_channels=128,
    deconv_dropout=0.0,
    deconv_loss_weight=1.0,
    deconv_target_blur_sigma=3.0,
    deconv_bce_weight=1.0,
    deconv_coverage_weight=0.2,
    deconv_mass_weight=0.01,
    deconv_entropy_weight=0.0,
    deconv_use_brain_mask=True,
    deconv_brain_mask_path=None,
    deconv_loss="dice_bce",
    deconv_mask_outside_brain=True,
    deconv_tv_weight=0.0,
    deconv_pos_weight=None,
    deconv_outside_brain_penalty_weight=0.01,
    deconv_val_image_log_every=20,
    sigma_reg_lambda=5e-4,
    gaussian_sigma_reg_lambda=5e-4,
    harmonize_lobe_locations=True,
    lobe_harmonization_factor=1.0,
    lesion_metric_every=50,
    mri_npy_dir=None,
    lesion_mask_threshold=0.5,
    dry_run=False,
    early_stopping=True,
    early_stopping_patience=150,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=150,
    early_stopping_smoothing_window=10,
    restore_best_checkpoint=True,
    pretrained_encoder_path=None,
    freeze_encoder="none",
):
    use_coord_head = lambda_coord > 0.0
    use_hemi_head = lambda_hemi > 0.0
    use_lobe_head = lambda_lobe > 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        f"Active heads — coord: {use_coord_head} (λ={lambda_coord}), "
        f"hemi: {use_hemi_head} (λ={lambda_hemi}), "
        f"lobe: {use_lobe_head} (λ={lambda_lobe}), "
        f"spatial: {spatial_head}"
    )
    if lesion_metric_every is not None and lesion_metric_every > 0:
        if mri_npy_dir is None:
            print(
                "Warning: lesion metric is enabled by cadence, but mri_npy_dir is None. "
                "Lesion metric computation will be skipped."
            )
        else:
            print(
                f"Lesion metric enabled: every {lesion_metric_every} epochs "
                f"(threshold={lesion_mask_threshold}, mri_npy_dir={mri_npy_dir})"
            )

    gaussian_sigma_min, gaussian_sigma_max = resolve_gaussian_sigma_bounds(
        gaussian_output_space=gaussian_output_space,
        gaussian_sigma_min=gaussian_sigma_min,
        gaussian_sigma_max=gaussian_sigma_max,
    )
    if deconv_loss == "soft_bce_coverage" and spatial_head != "deconv":
        raise ValueError("deconv_loss='soft_bce_coverage' requires --spatial_head deconv")
    if deconv_bce_weight < 0 or deconv_coverage_weight < 0 or deconv_mass_weight < 0:
        raise ValueError("deconv_bce_weight, deconv_coverage_weight, and deconv_mass_weight must be non-negative")

    if spatial_head == "gaussian_mixture":
        print("Gaussian-mixture spatial head enabled:")
        print(f"  num_gaussians: {num_gaussians}")
        print(f"  coord_dim: {gaussian_coord_dim}")
        print(f"  sigma range: [{gaussian_sigma_min}, {gaussian_sigma_max}]")
        print(f"  isotropic: {gaussian_isotropic}")
        print(f"  output_space: {gaussian_output_space}")
        print(f"  loss_weight: {gaussian_loss_weight}")
        print(f"  target mode: {gaussian_target}")
        print(f"  make_heatmap: {gaussian_make_heatmap}")
        print(f"  heatmap shape: {gaussian_heatmap_shape}")
    if spatial_head == "deconv":
        print("Deconv spatial head enabled:")
        print(f"  output shape: {tuple(deconv_output_shape)}")
        print(f"  latent shape: {tuple(deconv_latent_shape)}")
        print(f"  base channels: {deconv_base_channels}")
        print(f"  target source (preferred): {mri_npy_dir}/<pid>_preproc.npz['gt']")
        print(f"  brain mask path: {deconv_brain_mask_path}")
        print(f"  loss type: {deconv_loss}")
        print(f"  target blur sigma: {deconv_target_blur_sigma}")
        print(f"  soft_bce_coverage bce weight: {deconv_bce_weight}")
        print(f"  soft_bce_coverage coverage weight: {deconv_coverage_weight}")
        print(f"  soft_bce_coverage mass weight: {deconv_mass_weight}")
        print(f"  entropy weight: {deconv_entropy_weight}")
        print(f"  loss weight: {deconv_loss_weight}")
        print(f"  outside-brain penalty weight: {deconv_outside_brain_penalty_weight}")
        print(f"  mask outside brain: {deconv_mask_outside_brain}")

    train_ids, val_ids = load_split(json_split_path, fold_index)
    print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")

    target_dict = load_multitask_targets(
        json_targets_path,
        lobe_json_path=lobe_json_path,
        hemi_json_path=hemi_json_path,
    )

    train_ids, train_files, train_targets = find_patient_files(
        data_dir, train_ids, target_dict, test_mode=test_mode
    )
    val_ids, val_files, val_targets = find_patient_files(
        data_dir, val_ids, target_dict, test_mode=test_mode
    )

    # Base dataset keeps existing spike sampling/augmentation behavior.
    train_base = PatientMILSpikeDataset(
        train_ids,
        train_files,
        train_targets,
        max_spikes_per_bag=max_spikes,
        min_spikes_per_patient=min_spikes_per_patient,
        training=True,
    )
    val_base = PatientMILSpikeDataset(
        val_ids,
        val_files,
        val_targets,
        max_spikes_per_bag=max_spikes,
        min_spikes_per_patient=min_spikes_per_patient,
        training=False,
    )

    train_dataset = MultiHeadTargetDataset(
        train_base,
        target_by_pid=target_dict,
        deconv_enabled=(spatial_head == "deconv"),
        deconv_output_shape=tuple(deconv_output_shape),
        deconv_target_blur_sigma=deconv_target_blur_sigma,
        deconv_require_mni_alignment=(spatial_head == "deconv"),
        deconv_mask_npz_dir=mri_npy_dir if spatial_head == "deconv" else None,
        harmonize_lobe_locations=harmonize_lobe_locations,
        lobe_harmonization_factor=lobe_harmonization_factor,
    )
    val_dataset = MultiHeadTargetDataset(
        val_base,
        target_by_pid=target_dict,
        deconv_enabled=(spatial_head == "deconv"),
        deconv_output_shape=tuple(deconv_output_shape),
        deconv_target_blur_sigma=deconv_target_blur_sigma,
        deconv_require_mni_alignment=(spatial_head == "deconv"),
        deconv_mask_npz_dir=mri_npy_dir if spatial_head == "deconv" else None,
    )

    train_sampler = None
    if harmonize_lobe_locations and getattr(train_dataset, "sample_weights", None) is not None:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        print(
            "Using lobe-harmonized weighted sampling for training "
            f"(factor={lobe_harmonization_factor:.3f})."
        )
    elif harmonize_lobe_locations:
        print("Lobe harmonization requested, but no sampling weights were available; using uniform training sampling.")

    gaussian_args = argparse.Namespace(
        spatial_head=spatial_head,
        gaussian_target=gaussian_target,
        gaussian_output_space=gaussian_output_space,
        gaussian_make_heatmap=gaussian_make_heatmap,
        gaussian_heatmap_shape=gaussian_heatmap_shape,
    )
    validate_gaussian_targets(train_dataset, val_dataset, gaussian_args)

    deconv_args = argparse.Namespace(
        spatial_head=spatial_head,
        deconv_output_shape=tuple(deconv_output_shape),
        deconv_use_brain_mask=deconv_use_brain_mask,
        deconv_brain_mask_path=deconv_brain_mask_path,
    )
    validate_deconv_targets(train_dataset, val_dataset, deconv_args)

    deconv_brain_mask = None
    if spatial_head == "deconv" and deconv_use_brain_mask:
        deconv_brain_mask = load_deconv_brain_mask(
            mask_path=deconv_brain_mask_path,
            output_shape=tuple(deconv_output_shape),
            device=device,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=mil_multitask_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_multitask_collate,
        pin_memory=True,
    )

    model_kwargs = {k: v for k, v in dict(
        emb_dim=emb_dim, hidden=hidden, dropout=dropout,
        encoder_type=encoder_type, pooling=pooling,
        spatial_head=spatial_head,
        num_gaussians=num_gaussians,
        gaussian_coord_dim=gaussian_coord_dim,
        gaussian_sigma_min=gaussian_sigma_min,
        gaussian_sigma_max=gaussian_sigma_max,
        gaussian_isotropic=gaussian_isotropic,
        gaussian_output_space=gaussian_output_space,
        gaussian_make_heatmap=gaussian_make_heatmap,
        gaussian_heatmap_shape=gaussian_heatmap_shape,
        deconv_output_shape=tuple(deconv_output_shape),
        deconv_latent_shape=tuple(deconv_latent_shape),
        deconv_base_channels=deconv_base_channels,
        deconv_dropout=deconv_dropout,
    ).items() if v is not None}
    print(f"Model kwargs (overrides): {model_kwargs}")
    model = SpikeMILModel(
        in_channels=in_channels,
        n_hemi_classes=len(HEMI_LABEL_TO_INT),
        n_lobe_classes=len(LOBE_CLASSES),
        use_coord_head=use_coord_head,
        use_hemi_head=use_hemi_head,
        use_lobe_head=use_lobe_head,
        **model_kwargs,
    ).to(device)

    if pretrained_encoder_path is not None:
        load_pretrained_encoder(model, pretrained_encoder_path, freeze_mode=freeze_encoder)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"eeg_mil_mh_fold{fold_index}_{datestr}")
    os.makedirs(log_dir, exist_ok=True)

    run_fingerprint_payload = emit_run_fingerprint(
        script_name="eeg_spike_mil_mh_training",
        train_config={
            "json_split_path": json_split_path,
            "fold_index": fold_index,
            "data_dir": data_dir,
            "json_targets_path": json_targets_path,
            "lobe_json_path": lobe_json_path,
            "hemi_json_path": hemi_json_path,
            "in_channels": in_channels,
            "max_spikes": max_spikes,
            "min_spikes_per_patient": min_spikes_per_patient,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "log_root": log_root,
            "num_workers": num_workers,
            "test_mode": test_mode,
            "encoder_type": encoder_type,
            "pooling": pooling,
            "lambda_coord": lambda_coord,
            "lambda_hemi": lambda_hemi,
            "lambda_lobe": lambda_lobe,
            "spatial_head": spatial_head,
            "num_gaussians": num_gaussians,
            "gaussian_coord_dim": gaussian_coord_dim,
            "gaussian_sigma_min": gaussian_sigma_min,
            "gaussian_sigma_max": gaussian_sigma_max,
            "gaussian_isotropic": gaussian_isotropic,
            "gaussian_output_space": gaussian_output_space,
            "gaussian_make_heatmap": gaussian_make_heatmap,
            "gaussian_heatmap_shape": gaussian_heatmap_shape,
            "gaussian_loss_weight": gaussian_loss_weight,
            "gaussian_target": gaussian_target,
            "gaussian_target_blur_sigma": gaussian_target_blur_sigma,
            "deconv_output_shape": tuple(deconv_output_shape),
            "deconv_latent_shape": tuple(deconv_latent_shape),
            "deconv_base_channels": deconv_base_channels,
            "deconv_dropout": deconv_dropout,
            "deconv_loss_weight": deconv_loss_weight,
            "deconv_target_blur_sigma": deconv_target_blur_sigma,
            "deconv_bce_weight": deconv_bce_weight,
            "deconv_coverage_weight": deconv_coverage_weight,
            "deconv_mass_weight": deconv_mass_weight,
            "deconv_entropy_weight": deconv_entropy_weight,
            "deconv_use_brain_mask": deconv_use_brain_mask,
            "deconv_brain_mask_path": deconv_brain_mask_path,
            "deconv_loss": deconv_loss,
            "deconv_mask_outside_brain": deconv_mask_outside_brain,
            "deconv_tv_weight": deconv_tv_weight,
            "deconv_pos_weight": deconv_pos_weight,
            "deconv_outside_brain_penalty_weight": deconv_outside_brain_penalty_weight,
            "deconv_val_image_log_every": deconv_val_image_log_every,
            "sigma_reg_lambda": sigma_reg_lambda,
            "gaussian_sigma_reg_lambda": gaussian_sigma_reg_lambda,
            "harmonize_lobe_locations": harmonize_lobe_locations,
            "lobe_harmonization_factor": lobe_harmonization_factor,
            "lesion_metric_every": lesion_metric_every,
            "mri_npy_dir": mri_npy_dir,
            "lesion_mask_threshold": lesion_mask_threshold,
            "pretrained_encoder_path": pretrained_encoder_path,
            "freeze_encoder": freeze_encoder,
        },
        model_kwargs=model_kwargs,
        effective_model_config={
            "model_class": model.__class__.__name__,
            "emb_dim": model.emb_dim,
            "hidden": model.hidden,
            "dropout": model.dropout,
            "pooling": model.pooling,
            "encoder_type": model.encoder_type,
            "use_coord_head": model.use_coord_head,
            "use_hemi_head": model.use_hemi_head,
            "use_lobe_head": model.use_lobe_head,
            "spatial_head": model.spatial_head,
            "use_gaussian_mixture_head": model.use_gaussian_mixture_head,
            "num_gaussians": model.num_gaussians,
            "gaussian_coord_dim": model.gaussian_coord_dim,
            "gaussian_sigma_min": model.gaussian_sigma_min,
            "gaussian_sigma_max": model.gaussian_sigma_max,
            "gaussian_isotropic": model.gaussian_isotropic,
            "gaussian_output_space": model.gaussian_output_space,
            "gaussian_make_heatmap": model.gaussian_make_heatmap,
            "gaussian_heatmap_shape": model.gaussian_heatmap_shape,
            "use_deconv_spatial_head": getattr(model, "use_deconv_spatial_head", False),
            "deconv_output_shape": getattr(model, "deconv_output_shape", None),
            "deconv_latent_shape": getattr(model, "deconv_latent_shape", None),
            "deconv_base_channels": getattr(model, "deconv_base_channels", None),
            "deconv_dropout": getattr(model, "deconv_dropout", None),
        },
        extra={"device": str(device)},
    )

    # Persist run settings/fingerprint alongside TensorBoard and inference outputs.
    run_fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(run_fingerprint_path, "w") as f:
        json.dump(run_fingerprint_payload, f, indent=2)
    print(f"Saved run fingerprint to: {run_fingerprint_path}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    if dry_run:
        run_smoke_test(model, train_loader, device)
        return None

    writer = SummaryWriter(log_dir=log_dir)
    print("Logging to:", log_dir)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping setup
    es = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        warmup=early_stopping_warmup,
        smoothing_window=early_stopping_smoothing_window,
        enabled=early_stopping,
    )
    best_val_euclidean_mm = float("inf")

    # Checkpoint paths
    ckpt_last = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    ckpt_best_raw = os.path.join(checkpoint_dir, "checkpoint_best_raw_val_loss.pt")
    ckpt_best_smoothed = os.path.join(checkpoint_dir, "checkpoint_best_smoothed_val_loss.pt")

    def _make_checkpoint(epoch_1based, val_loss_val, smoothed_val_loss_val, es_info):
        scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
        return {
            "epoch": epoch_1based,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": scaler_state,
            "val_loss": val_loss_val,
            "smoothed_val_loss": smoothed_val_loss_val,
            "best_raw_val_loss": es_info["best_raw_val_loss"],
            "best_smoothed_val_loss": es_info["best_smoothed_val_loss"],
            "args": {
                "json_split_path": json_split_path,
                "fold_index": fold_index,
                "data_dir": data_dir,
                "json_targets_path": json_targets_path,
                "lobe_json_path": lobe_json_path,
                "hemi_json_path": hemi_json_path,
                "in_channels": in_channels,
                "max_spikes": max_spikes,
                "min_spikes_per_patient": min_spikes_per_patient,
                "batch_size": batch_size,
                "emb_dim": model.emb_dim,
                "hidden": model.hidden,
                "dropout": model.dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "num_workers": num_workers,
                "encoder_type": model.encoder_type,
                "pooling": model.pooling,
                "lambda_coord": lambda_coord,
                "lambda_hemi": lambda_hemi,
                "lambda_lobe": lambda_lobe,
                "spatial_head": spatial_head,
                "num_gaussians": num_gaussians,
                "gaussian_coord_dim": gaussian_coord_dim,
                "gaussian_sigma_min": gaussian_sigma_min,
                "gaussian_sigma_max": gaussian_sigma_max,
                "gaussian_isotropic": gaussian_isotropic,
                "gaussian_output_space": gaussian_output_space,
                "gaussian_make_heatmap": gaussian_make_heatmap,
                "gaussian_heatmap_shape": gaussian_heatmap_shape,
                "gaussian_loss_weight": gaussian_loss_weight,
                "gaussian_target": gaussian_target,
                "gaussian_target_blur_sigma": gaussian_target_blur_sigma,
                "deconv_output_shape": tuple(deconv_output_shape),
                "deconv_latent_shape": tuple(deconv_latent_shape),
                "deconv_base_channels": deconv_base_channels,
                "deconv_dropout": deconv_dropout,
                "deconv_loss_weight": deconv_loss_weight,
                "deconv_target_blur_sigma": deconv_target_blur_sigma,
                "deconv_bce_weight": deconv_bce_weight,
                "deconv_coverage_weight": deconv_coverage_weight,
                "deconv_mass_weight": deconv_mass_weight,
                "deconv_entropy_weight": deconv_entropy_weight,
                "deconv_use_brain_mask": deconv_use_brain_mask,
                "deconv_brain_mask_path": deconv_brain_mask_path,
                "deconv_loss": deconv_loss,
                "deconv_mask_outside_brain": deconv_mask_outside_brain,
                "deconv_tv_weight": deconv_tv_weight,
                "deconv_pos_weight": deconv_pos_weight,
                "deconv_outside_brain_penalty_weight": deconv_outside_brain_penalty_weight,
                "deconv_val_image_log_every": deconv_val_image_log_every,
                "sigma_reg_lambda": sigma_reg_lambda,
                "gaussian_sigma_reg_lambda": gaussian_sigma_reg_lambda,
                "lesion_metric_every": lesion_metric_every,
                "mri_npy_dir": mri_npy_dir,
                "lesion_mask_threshold": lesion_mask_threshold,
                "use_coord_head": use_coord_head,
                "use_hemi_head": use_hemi_head,
                "use_lobe_head": use_lobe_head,
                "early_stopping": early_stopping,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_warmup": early_stopping_warmup,
                "early_stopping_smoothing_window": early_stopping_smoothing_window,
                "restore_best_checkpoint": restore_best_checkpoint,
            },
            "lobe_classes": LOBE_CLASSES,
            "val_coord_euclidean_mm": None,  # filled below if available
            "val_coord_loss": None,
        }

    stopped_early = False
    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            lambda_coord=lambda_coord,
            lambda_hemi=lambda_hemi,
            lambda_lobe=lambda_lobe,
            spatial_head=spatial_head,
            gaussian_loss_weight=gaussian_loss_weight,
            gaussian_target=gaussian_target,
            gaussian_output_space=gaussian_output_space,
            deconv_loss_weight=deconv_loss_weight,
            deconv_loss=deconv_loss,
            deconv_target_blur_sigma=deconv_target_blur_sigma,
            deconv_bce_weight=deconv_bce_weight,
            deconv_coverage_weight=deconv_coverage_weight,
            deconv_mass_weight=deconv_mass_weight,
            deconv_entropy_weight=deconv_entropy_weight,
            deconv_tv_weight=deconv_tv_weight,
            deconv_pos_weight=deconv_pos_weight,
            deconv_outside_brain_penalty_weight=deconv_outside_brain_penalty_weight,
            deconv_mask_outside_brain=deconv_mask_outside_brain,
            deconv_brain_mask=deconv_brain_mask,
            sigma_reg_lambda=sigma_reg_lambda,
            gaussian_sigma_reg_lambda=gaussian_sigma_reg_lambda,
            writer=writer,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            epoch,
            lambda_coord=lambda_coord,
            lambda_hemi=lambda_hemi,
            lambda_lobe=lambda_lobe,
            spatial_head=spatial_head,
            gaussian_loss_weight=gaussian_loss_weight,
            gaussian_target=gaussian_target,
            gaussian_output_space=gaussian_output_space,
            gaussian_sigma_reg_lambda=gaussian_sigma_reg_lambda,
            deconv_loss_weight=deconv_loss_weight,
            deconv_loss=deconv_loss,
            deconv_target_blur_sigma=deconv_target_blur_sigma,
            deconv_bce_weight=deconv_bce_weight,
            deconv_coverage_weight=deconv_coverage_weight,
            deconv_mass_weight=deconv_mass_weight,
            deconv_entropy_weight=deconv_entropy_weight,
            deconv_tv_weight=deconv_tv_weight,
            deconv_pos_weight=deconv_pos_weight,
            deconv_outside_brain_penalty_weight=deconv_outside_brain_penalty_weight,
            deconv_mask_outside_brain=deconv_mask_outside_brain,
            deconv_brain_mask=deconv_brain_mask,
            deconv_val_image_log_every=deconv_val_image_log_every,
            writer=writer,
        )

        lesion_contrast_metrics = None
        should_compute_lesion_metric = (
            mri_npy_dir is not None
            and lesion_metric_every is not None
            and lesion_metric_every > 0
            and ((epoch + 1) % lesion_metric_every == 0)
        )
        if should_compute_lesion_metric:
            val_preds_for_lesion_metric = infer_predictions(
                model,
                val_loader,
                val_dataset.patient_ids,
                device,
            )
            lesion_contrast_metrics = compute_lesion_contrast_metrics(
                val_preds_for_lesion_metric,
                mri_npy_dir=mri_npy_dir,
                gaussian_output_space=gaussian_output_space,
                lesion_threshold=lesion_mask_threshold,
            )

            inside_mean = lesion_contrast_metrics["inside_mean"]
            outside_mean = lesion_contrast_metrics["outside_mean"]
            contrast = lesion_contrast_metrics["contrast"]
            voxel_mae = lesion_contrast_metrics["voxel_mae"]

            print(
                "Lesion contrast (val) "
                f"[cases={lesion_contrast_metrics['valid_cases']}, skipped={lesion_contrast_metrics['skipped_cases']}]: "
                f"inside={inside_mean if inside_mean is not None else 'N/A'}, "
                f"outside={outside_mean if outside_mean is not None else 'N/A'}, "
                f"inside_minus_outside={contrast if contrast is not None else 'N/A'}, "
                f"voxel_mae={voxel_mae if voxel_mae is not None else 'N/A'}"
            )

            if writer is not None:
                if inside_mean is not None:
                    writer.add_scalar("lesion/val_inside_mean", inside_mean, epoch)
                if outside_mean is not None:
                    writer.add_scalar("lesion/val_outside_mean", outside_mean, epoch)
                if contrast is not None:
                    writer.add_scalar("lesion/val_inside_minus_outside", contrast, epoch)
                if voxel_mae is not None:
                    writer.add_scalar("lesion/val_voxel_mae", voxel_mae, epoch)

        epoch_duration = time.time() - epoch_start_time

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs}")

        train_log = (
            f"Train - Loss: {train_metrics['loss']:.4f}"
        )
        val_log = (
            f"Val   - Loss: {val_metrics['loss']:.4f}"
        )
        if use_coord_head:
            train_log += (
                f", Coord: {train_metrics['coord_loss']:.4f}"
                f", Euclidean(mm): {train_metrics['coord_euclidean_mm']:.2f}"
            )
            val_log += (
                f", Coord: {val_metrics['coord_loss']:.4f}"
                f", Euclidean(mm): {val_metrics['coord_euclidean_mm']:.2f}"
            )
        if use_hemi_head:
            train_log += (
                f", Hemi: {train_metrics['hemi_loss']:.4f}"
                f", HemiAcc: {train_metrics['hemi_acc']:.3f}"
            )
            val_log += (
                f", Hemi: {val_metrics['hemi_loss']:.4f}"
                f", HemiAcc: {val_metrics['hemi_acc']:.3f}"
            )
        if use_lobe_head:
            train_log += (
                f", Lobe: {train_metrics['lobe_loss']:.4f}"
                f", LobeAcc: {train_metrics['lobe_acc']:.3f}"
            )
            val_log += (
                f", Lobe: {val_metrics['lobe_loss']:.4f}"
                f", LobeAcc: {val_metrics['lobe_acc']:.3f}"
            )
        if spatial_head == "gaussian_mixture":
            train_log += (
                f", Gauss: {train_metrics['gaussian_loss']:.4f}"
                f", GmDistTop: {train_metrics['gaussian_dist_top_weight']:.4f}"
            )
            val_log += (
                f", Gauss: {val_metrics['gaussian_loss']:.4f}"
                f", GmDistTop: {val_metrics['gaussian_dist_top_weight']:.4f}"
            )
        if spatial_head == "deconv":
            train_log += (
                f", Deconv: {train_metrics['deconv_loss']:.4f}"
                f", Dice: {train_metrics['deconv_dice']:.4f}"
                f", MassInGT: {train_metrics['deconv_mass_in_gt']:.4f}"
            )
            val_log += (
                f", Deconv: {val_metrics['deconv_loss']:.4f}"
                f", Dice: {val_metrics['deconv_dice']:.4f}"
                f", PeakDist: {val_metrics['deconv_peak_distance']:.2f}"
            )

        print(train_log)
        print(val_log)
        print(f"Epoch duration: {epoch_duration:.1f}s")

        writer.add_scalars("loss/total", {"Train": train_metrics["loss"], "Val": val_metrics["loss"]}, epoch)
        if use_coord_head:
            writer.add_scalars(
                "loss/coord",
                {"Train": train_metrics["coord_loss"], "Val": val_metrics["coord_loss"]},
                epoch,
            )
            writer.add_scalars(
                "coord/euclidean_mm",
                {"Train": train_metrics["coord_euclidean_mm"], "Val": val_metrics["coord_euclidean_mm"]},
                epoch,
            )
            writer.add_scalars(
                "coord/euclidean_norm",
                {"Train": train_metrics["coord_euclidean_norm"], "Val": val_metrics["coord_euclidean_norm"]},
                epoch,
            )
            writer.add_scalars(
                "coord/sigma",
                {"Train": train_metrics["sigma"], "Val": val_metrics["sigma"]},
                epoch,
            )
        if use_hemi_head:
            writer.add_scalars(
                "loss/hemi",
                {"Train": train_metrics["hemi_loss"], "Val": val_metrics["hemi_loss"]},
                epoch,
            )
            writer.add_scalars(
                "hemi/acc",
                {"Train": train_metrics["hemi_acc"], "Val": val_metrics["hemi_acc"]},
                epoch,
            )
            writer.add_scalars(
                "loss/lobe",
                {"Train": train_metrics["lobe_loss"], "Val": val_metrics["lobe_loss"]},
                epoch,
            )
            writer.add_scalars(
                "lobe/acc",
                {"Train": train_metrics["lobe_acc"], "Val": val_metrics["lobe_acc"]},
                epoch,
            )
        if spatial_head == "gaussian_mixture":
            writer.add_scalars(
                "loss/gaussian",
                {"Train": train_metrics["gaussian_loss"], "Val": val_metrics["gaussian_loss"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/sigma_mean",
                {"Train": train_metrics["gaussian_sigma"], "Val": val_metrics["gaussian_sigma"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/sigma_min",
                {"Train": train_metrics["gaussian_sigma_min"], "Val": val_metrics["gaussian_sigma_min"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/sigma_max",
                {"Train": train_metrics["gaussian_sigma_max"], "Val": val_metrics["gaussian_sigma_max"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/weight_entropy",
                {"Train": train_metrics["gaussian_weight_entropy"], "Val": val_metrics["gaussian_weight_entropy"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/max_weight",
                {"Train": train_metrics["gaussian_max_weight"], "Val": val_metrics["gaussian_max_weight"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/dist_top_weight",
                {"Train": train_metrics["gaussian_dist_top_weight"], "Val": val_metrics["gaussian_dist_top_weight"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/dist_nearest",
                {"Train": train_metrics["gaussian_dist_nearest"], "Val": val_metrics["gaussian_dist_nearest"]},
                epoch,
            )
            writer.add_scalars(
                "gaussian/dist_expected",
                {"Train": train_metrics["gaussian_dist_expected"], "Val": val_metrics["gaussian_dist_expected"]},
                epoch,
            )
        if spatial_head == "deconv":
            writer.add_scalars(
                "loss/deconv",
                {"Train": train_metrics["deconv_loss"], "Val": val_metrics["deconv_loss"]},
                epoch,
            )
            if deconv_loss == "soft_bce":
                writer.add_scalar("deconv/val_soft_bce_loss", val_metrics["deconv_loss"], epoch)
            if deconv_loss == "soft_bce_coverage":
                writer.add_scalar("deconv/val_soft_bce_coverage_loss", val_metrics["deconv_loss"], epoch)
            writer.add_scalars(
                "val/deconv_dice",
                {"Train": train_metrics["deconv_dice"], "Val": val_metrics["deconv_dice"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_mass_in_gt",
                {"Train": train_metrics["deconv_mass_in_gt"], "Val": val_metrics["deconv_mass_in_gt"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_peak_distance",
                {"Train": train_metrics["deconv_peak_distance"], "Val": val_metrics["deconv_peak_distance"]},
                epoch,
            )
            writer.add_scalars(
                "deconv/pred_max",
                {"Train": train_metrics["deconv_pred_max"], "Val": val_metrics["deconv_pred_max"]},
                epoch,
            )
            writer.add_scalars(
                "deconv/pred_mean",
                {"Train": train_metrics["deconv_pred_mean"], "Val": val_metrics["deconv_pred_mean"]},
                epoch,
            )
            writer.add_scalars(
                "deconv/target_soft_max",
                {"Train": train_metrics["deconv_target_soft_max"], "Val": val_metrics["deconv_target_soft_max"]},
                epoch,
            )
            writer.add_scalars(
                "deconv/target_soft_mean",
                {"Train": train_metrics["deconv_target_soft_mean"], "Val": val_metrics["deconv_target_soft_mean"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_soft_bce",
                {"Train": train_metrics["deconv_soft_bce"], "Val": val_metrics["deconv_soft_bce"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_coverage_loss",
                {"Train": train_metrics["deconv_coverage_loss"], "Val": val_metrics["deconv_coverage_loss"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_mass_loss",
                {"Train": train_metrics["deconv_mass_loss"], "Val": val_metrics["deconv_mass_loss"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_coverage_value",
                {"Train": train_metrics["deconv_coverage_value"], "Val": val_metrics["deconv_coverage_value"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_mass_value",
                {"Train": train_metrics["deconv_mass_value"], "Val": val_metrics["deconv_mass_value"]},
                epoch,
            )
            writer.add_scalars(
                "val/deconv_effective_volume_voxels",
                {"Train": train_metrics["deconv_effective_volume_voxels"], "Val": val_metrics["deconv_effective_volume_voxels"]},
                epoch,
            )
            if deconv_entropy_weight > 0:
                writer.add_scalars(
                    "deconv/entropy",
                    {"Train": train_metrics["deconv_entropy"], "Val": val_metrics["deconv_entropy"]},
                    epoch,
                )
        writer.add_scalars(
            "attention/entropy",
            {"Train": train_metrics["attn_entropy"], "Val": val_metrics["attn_entropy"]},
            epoch,
        )

        val_loss = val_metrics["loss"]

        # Update early stopping state
        es_info = es.update(epoch, val_loss)
        smoothed_val_loss = es_info["smoothed_val_loss"]

        # Track best euclidean for print summary
        if es_info["raw_improved"] and use_coord_head:
            best_val_euclidean_mm = val_metrics["coord_euclidean_mm"]

        best_epoch_1based = (es_info["best_epoch"] + 1) if es_info["best_epoch"] is not None else None
        lr_current = optimizer.param_groups[0]["lr"]

        # Smoothed/ES log line
        print(
            f"Smoothed val loss: {smoothed_val_loss:.6f} | "
            f"best_smoothed: {es_info['best_smoothed_val_loss']:.6f} @ epoch {best_epoch_1based} | "
            f"best_raw: {es_info['best_raw_val_loss']:.6f} | "
            f"no_improve: {es_info['epochs_without_improvement']}/{early_stopping_patience} | "
            f"lr: {lr_current:.2e}"
        )

        # TensorBoard ES scalars
        writer.add_scalar("val/loss_raw", val_loss, epoch)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss, epoch)
        writer.add_scalar("early_stopping/best_smoothed_val_loss", es_info["best_smoothed_val_loss"], epoch)
        writer.add_scalar("early_stopping/epochs_without_improvement", es_info["epochs_without_improvement"], epoch)
        if best_epoch_1based is not None:
            writer.add_scalar("early_stopping/best_epoch", best_epoch_1based, epoch)

        ckpt_data = _make_checkpoint(epoch + 1, val_loss, smoothed_val_loss, es_info)
        ckpt_data["val_coord_euclidean_mm"] = val_metrics.get("coord_euclidean_mm")
        ckpt_data["val_coord_loss"] = val_metrics.get("coord_loss")
        ckpt_data["val_deconv_loss"] = val_metrics.get("deconv_loss")
        ckpt_data["val_deconv_dice"] = val_metrics.get("deconv_dice")
        ckpt_data["val_deconv_mass_in_gt"] = val_metrics.get("deconv_mass_in_gt")
        ckpt_data["val_deconv_peak_distance"] = val_metrics.get("deconv_peak_distance")
        ckpt_data["val_lesion_inside_mean"] = (
            lesion_contrast_metrics["inside_mean"] if lesion_contrast_metrics is not None else None
        )
        ckpt_data["val_lesion_outside_mean"] = (
            lesion_contrast_metrics["outside_mean"] if lesion_contrast_metrics is not None else None
        )
        ckpt_data["val_lesion_inside_minus_outside"] = (
            lesion_contrast_metrics["contrast"] if lesion_contrast_metrics is not None else None
        )
        ckpt_data["val_lesion_voxel_mae"] = (
            lesion_contrast_metrics["voxel_mae"] if lesion_contrast_metrics is not None else None
        )

        # Save last checkpoint every epoch
        torch.save(ckpt_data, ckpt_last)

        # Save best raw-loss checkpoint
        if es_info["raw_improved"]:
            torch.save(ckpt_data, ckpt_best_raw)
            print(
                f"✓ New best raw val loss: {val_loss:.4f}"
                + (f" (Euclidean: {val_metrics['coord_euclidean_mm']:.4f} mm)" if use_coord_head else "")
            )

        # Save best smoothed-loss checkpoint
        if es_info["improved"]:
            torch.save(ckpt_data, ckpt_best_smoothed)
            print(f"✓ New best smoothed val loss: {smoothed_val_loss:.6f}")

        # Early stopping check
        if es_info["should_stop"]:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}. "
                f"Best smoothed validation loss was {es_info['best_smoothed_val_loss']:.6f} "
                f"at epoch {best_epoch_1based}."
            )
            stopped_early = True
            break

    # Restore best smoothed checkpoint before inference/export
    if restore_best_checkpoint and os.path.exists(ckpt_best_smoothed):
        restored = torch.load(ckpt_best_smoothed, map_location=device)
        model.load_state_dict(restored["model_state_dict"])
        restored_epoch = restored["epoch"]
        print(f"\nRestored best smoothed-loss checkpoint (epoch {restored_epoch}).")

    final_ckpt_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler_state,
            "best_epoch": (es.best_epoch + 1) if es.best_epoch is not None else None,
            "best_raw_val_loss": es.best_raw_val_loss,
            "best_smoothed_val_loss": es.best_smoothed_val_loss,
            "best_val_euclidean_mm": best_val_euclidean_mm,
            "stopped_early": stopped_early,
            "lobe_classes": LOBE_CLASSES,
        },
        final_ckpt_path,
    )

    writer.close()
    print(f"Saved final checkpoint to: {final_ckpt_path}")
    print(f"\n{'=' * 60}")
    print("Training complete!" + (" (early stop)" if stopped_early else ""))
    print(f"Best raw val loss:      {es.best_raw_val_loss:.4f} @ epoch {(es.best_raw_epoch + 1) if es.best_raw_epoch is not None else 'N/A'}")
    print(f"Best smoothed val loss: {es.best_smoothed_val_loss:.6f} @ epoch {(es.best_epoch + 1) if es.best_epoch is not None else 'N/A'}")
    if use_coord_head:
        print(f"Best Euclidean (mm):    {best_val_euclidean_mm:.4f}")
    print(f"{'=' * 60}")

    return {
        "best_checkpoint": ckpt_best_smoothed, "best_raw_checkpoint": ckpt_best_raw,
        "last_checkpoint": ckpt_last, "final_checkpoint": final_ckpt_path,
        "log_dir": log_dir,
    }


@torch.no_grad()
def infer_predictions(
    model,
    dataloader,
    case_ids,
    device,
):
    """
    Run inference and collect predictions.

    Returns
    -------
    dict
        {case_id: {
            "mu": [x, y, z],
            "sigma": [sx, sy, sz],
            "gaussian_mixture": {...} | None,
            "hemi_pred": int,
            "lobe_pred": int,
            "hemi_probs": [..],
            "lobe_probs": [..],
            "deconv_spatial_stats": {...} | None,
        }}
    """
    model.eval()
    predictions = {}
    case_idx = 0

    for x, mask, batch_targets in dataloader:
        x = x.to(device)
        mask = mask.to(device)
        target_mask = batch_targets.get("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(device)

        outputs = model(x, mask=mask)
        mu_hat = outputs["mu"]
        log_sigma_hat = outputs["log_sigma"]
        hemi_logits = outputs["hemi_logits"]
        lobe_logits = outputs["lobe_logits"]
        gaussian_mixture = outputs.get("gaussian_mixture")
        deconv_spatial = outputs.get("deconv_spatial")

        sigma_hat = torch.exp(log_sigma_hat) if log_sigma_hat is not None else None
        hemi_probs = torch.softmax(hemi_logits, dim=-1) if hemi_logits is not None else None
        lobe_probs = torch.softmax(lobe_logits, dim=-1) if lobe_logits is not None else None
        hemi_pred = hemi_probs.argmax(dim=-1) if hemi_probs is not None else None
        lobe_pred = lobe_probs.argmax(dim=-1) if lobe_probs is not None else None

        batch_size = x.size(0)
        for i in range(batch_size):
            if case_idx >= len(case_ids):
                break

            deconv_stats = None
            if deconv_spatial is not None:
                prob_i = deconv_spatial["prob"][i : i + 1]
                logits_i = deconv_spatial["logits"][i : i + 1]
                diag = compute_deconv_diagnostics(
                    logits=logits_i,
                    target_mask=None,
                    brain_mask=None,
                )
                deconv_stats = {
                    "pred_mean": float(diag["pred_mean_inside_brain"].detach().cpu().item()),
                    "pred_max": float(diag["pred_max"].detach().cpu().item()),
                    "pred_mean_inside_brain": float(diag["pred_mean_inside_brain"].detach().cpu().item()),
                    "effective_volume_voxels": float(diag["effective_volume_voxels"].detach().cpu().item()),
                }

                # If target masks are present in the collated batch, also report lesion metrics.
                if target_mask is not None:
                    target_i = target_mask[i : i + 1]
                    target_soft = make_soft_deconv_target(target_i, sigma=0.0)
                    dc_metrics = deconv_metrics(
                        prob_i,
                        target_i,
                        brain_mask=None,
                    )
                    _, deconv_loss_metrics = compute_deconv_spatial_loss(
                        logits=logits_i,
                        target_mask=target_i,
                        brain_mask=None,
                        loss_type="soft_bce_coverage",
                        target_blur_sigma=0.0,
                    )
                    target_soft_max = float(target_soft.max().detach().cpu().item())
                    target_soft_mean = float(target_soft.mean().detach().cpu().item())
                    deconv_stats.update(
                        {
                            "dice": float(dc_metrics["dice"].detach().cpu().item()),
                            "mass_in_gt": float(dc_metrics["mass_in_gt"].detach().cpu().item()),
                            "peak_distance": float(dc_metrics["peak_distance"].detach().cpu().item()),
                            "topk_hit": float(dc_metrics["topk_hit"].detach().cpu().item()),
                            "target_soft_max": target_soft_max,
                            "target_soft_mean": target_soft_mean,
                            "soft_bce": float(deconv_loss_metrics["deconv_soft_bce"].detach().cpu().item()),
                            "coverage_loss": float(deconv_loss_metrics["deconv_coverage_loss"].detach().cpu().item()),
                            "mass_loss": float(deconv_loss_metrics["deconv_mass_loss"].detach().cpu().item()),
                            "coverage_value": float(deconv_loss_metrics["deconv_coverage_value"].detach().cpu().item()),
                            "mass_value": float(deconv_loss_metrics["deconv_mass_value"].detach().cpu().item()),
                        }
                    )

            case_id = case_ids[case_idx]
            predictions[case_id] = {
                "mu":         mu_hat[i].detach().cpu().tolist() if mu_hat is not None else None,
                "sigma":      sigma_hat[i].detach().cpu().tolist() if sigma_hat is not None else None,
                "gaussian_mixture": {
                    "mu": gaussian_mixture["mu"][i].detach().cpu().tolist(),
                    "sigma": gaussian_mixture["sigma"][i].detach().cpu().tolist(),
                    "logits": gaussian_mixture["logits"][i].detach().cpu().tolist(),
                    "weights": gaussian_mixture["weights"][i].detach().cpu().tolist(),
                } if gaussian_mixture is not None else None,
                "hemi_pred":  int(hemi_pred[i].item()) if hemi_pred is not None else None,
                "lobe_pred":  int(lobe_pred[i].item()) if lobe_pred is not None else None,
                "hemi_probs": hemi_probs[i].detach().cpu().tolist() if hemi_probs is not None else None,
                "lobe_probs": lobe_probs[i].detach().cpu().tolist() if lobe_probs is not None else None,
                "deconv_spatial_stats": deconv_stats,
            }
            case_idx += 1

    return predictions


def _build_prior_volume_from_prediction(
    pred,
    img,
    affine,
    clamp_min_sigma_vox=1.0,
    gaussian_output_space="normalized",
):
    """Build a prior volume from one prediction dict and MRI metadata."""
    from datasets.multimodal import norm_to_mm, mm_to_vox, sigma_mm_to_vox, gaussian_prior_ijk

    _, d, h, w = img.shape
    prior = np.zeros((d, h, w), dtype=np.float32)

    gm = pred.get("gaussian_mixture", None)
    if gm is not None:
        gm_mu = np.asarray(gm["mu"], dtype=np.float32)            # (K, 3)
        gm_sigma = np.asarray(gm["sigma"], dtype=np.float32)      # (K, 1) or (K, 3)
        gm_weights = np.asarray(gm["weights"], dtype=np.float32)  # (K,)

        if gm_sigma.ndim == 2 and gm_sigma.shape[-1] == 1:
            gm_sigma = np.repeat(gm_sigma, repeats=3, axis=-1)

        if gm_mu.ndim != 2 or gm_mu.shape[-1] != 3:
            raise ValueError(f"Invalid gaussian_mixture mu shape: {gm_mu.shape}. Expected (K, 3).")
        if gm_sigma.ndim != 2 or gm_sigma.shape[-1] != 3:
            raise ValueError(f"Invalid gaussian_mixture sigma shape: {gm_sigma.shape}. Expected (K, 1) or (K, 3).")

        wsum = float(gm_weights.sum())
        if wsum <= 0.0:
            raise ValueError(f"Invalid gaussian_mixture weights: sum={wsum}")
        gm_weights = gm_weights / wsum

        for k in range(gm_mu.shape[0]):
            mu_k = gm_mu[k]
            sigma_k = gm_sigma[k]

            if gaussian_output_space == "normalized":
                mu_mm, sig_mm = norm_to_mm(mu_k, sigma_k)
            elif gaussian_output_space == "mni_mm":
                mu_mm = np.asarray(mu_k, dtype=np.float32)
                sig_mm = np.asarray(sigma_k, dtype=np.float32)
            else:
                raise ValueError(
                    f"Unsupported gaussian_output_space={gaussian_output_space!r}. "
                    "Expected 'normalized' or 'mni_mm'."
                )

            mu_ijk = mm_to_vox(mu_mm, affine)
            sig_ijk = sigma_mm_to_vox(sig_mm, affine)
            comp = gaussian_prior_ijk((d, h, w), mu_ijk, sig_ijk, clamp_min_vox=clamp_min_sigma_vox)
            prior += float(gm_weights[k]) * comp

        m = float(prior.max())
        if m > 0:
            prior /= m
    else:
        if pred.get("mu") is None or pred.get("sigma") is None:
            raise ValueError("Missing both gaussian_mixture and coordinate-head prediction.")

        mu_norm = np.asarray(pred["mu"], dtype=np.float32)
        sigma = pred["sigma"]
        sigma_norm = (
            np.array([float(sigma)] * 3, dtype=np.float32)
            if np.isscalar(sigma)
            else np.asarray(sigma, dtype=np.float32)
        )

        mu_mm, sig_mm = norm_to_mm(mu_norm, sigma_norm)
        mu_ijk = mm_to_vox(mu_mm, affine)
        sig_ijk = sigma_mm_to_vox(sig_mm, affine)
        prior = gaussian_prior_ijk((d, h, w), mu_ijk, sig_ijk, clamp_min_vox=clamp_min_sigma_vox)

    return prior


def compute_lesion_contrast_metrics(
    predictions,
    mri_npy_dir,
    clamp_min_sigma_vox=1.0,
    gaussian_output_space="normalized",
    lesion_threshold=0.5,
):
    """
    Compute dataset-level lesion contrast metrics from predicted prior volumes.

    Metric definitions (computed over all valid voxels across valid subjects):
      - inside_mean: mean prediction value inside lesion mask
      - outside_mean: mean prediction value outside lesion mask (within brain mask)
      - contrast: inside_mean - outside_mean
    """
    inside_sum = 0.0
    outside_sum = 0.0
    inside_count = 0
    outside_count = 0
    mae_sum = 0.0
    mae_count = 0
    valid_cases = 0
    skipped = 0

    for patient_id, pred in predictions.items():
        npy_path = os.path.join(mri_npy_dir, f"{patient_id}_preproc.npz")
        if not os.path.exists(npy_path):
            skipped += 1
            continue

        try:
            npz = np.load(npy_path, allow_pickle=True)
            img = npz["image"]
            affine = npz["affine"].astype(np.float32)
            gt = npz["gt"] if "gt" in npz else None
            npz.close()
        except Exception:
            skipped += 1
            continue

        if gt is None:
            skipped += 1
            continue

        try:
            prior = _build_prior_volume_from_prediction(
                pred,
                img=img,
                affine=affine,
                clamp_min_sigma_vox=clamp_min_sigma_vox,
                gaussian_output_space=gaussian_output_space,
            )
        except Exception:
            skipped += 1
            continue

        brain_mask = (img > 1e-5).any(axis=0)
        lesion_mask = np.asarray(gt, dtype=np.float32) > float(lesion_threshold)

        inside = lesion_mask & brain_mask
        outside = (~lesion_mask) & brain_mask

        in_count = int(inside.sum())
        out_count = int(outside.sum())
        if in_count == 0 or out_count == 0:
            skipped += 1
            continue

        prior_masked = prior * brain_mask.astype(np.float32)
        inside_sum += float(prior_masked[inside].sum())
        outside_sum += float(prior_masked[outside].sum())
        inside_count += in_count
        outside_count += out_count

        gt_binary = lesion_mask.astype(np.float32)
        mae_sum += float(np.abs(prior_masked[brain_mask] - gt_binary[brain_mask]).sum())
        mae_count += int(brain_mask.sum())

        valid_cases += 1

    inside_mean = (inside_sum / inside_count) if inside_count > 0 else None
    outside_mean = (outside_sum / outside_count) if outside_count > 0 else None
    contrast = (inside_mean - outside_mean) if (inside_mean is not None and outside_mean is not None) else None
    voxel_mae = (mae_sum / mae_count) if mae_count > 0 else None

    return {
        "inside_mean": inside_mean, "outside_mean": outside_mean, "contrast": contrast,
        "voxel_mae": voxel_mae,
        "valid_cases": valid_cases, "skipped_cases": skipped,
        "inside_voxels": inside_count, "outside_voxels": outside_count, "brain_voxels": mae_count,
    }


def generate_prior_niftis(
    predictions,
    mri_npy_dir,
    output_dir,
    clamp_min_sigma_vox=1.0,
    gaussian_output_space="normalized",
):
    """
    Generate Gaussian prior NIfTI files from EEG coordinate predictions.

    For each patient ID in ``predictions``, loads the corresponding MRI npz file
    (expected at ``{mri_npy_dir}/{patient_id}_preproc.npz``) to obtain the affine and volume
    shape, then builds a 3-D Gaussian blob in voxel space and saves it as
    ``{output_dir}/{patient_id}_prior.nii.gz``.

    Parameters
    ----------
    predictions : dict
        Maps patient_id to either a legacy coordinate prediction
        {"mu": [x, y, z], "sigma": ...} or a Gaussian-mixture prediction
        under key "gaussian_mixture".
    mri_npy_dir : str
        Directory containing ``{patient_id}_preproc.npz`` files with ``"image"`` and
        ``"affine"`` arrays.
    output_dir : str
        Directory in which the generated NIfTI files are saved.
    clamp_min_sigma_vox : float
        Minimum Gaussian sigma in voxels (passed to gaussian_prior_ijk).
    gaussian_output_space : str
        Coordinate space of gaussian_mixture components: "normalized" or "mni_mm".
    """
    import nibabel as nib
    os.makedirs(output_dir, exist_ok=True)
    saved, skipped = 0, 0

    for patient_id, pred in predictions.items():
        npy_path = os.path.join(mri_npy_dir, f"{patient_id}_preproc.npz")
        if not os.path.exists(npy_path):
            print(f"  [skip] No MRI npz found for {patient_id} at {npy_path}")
            skipped += 1
            continue

        try:
            npz = np.load(npy_path, allow_pickle=True)
            img = npz["image"]          # (2, D, H, W)
            affine = npz["affine"].astype(np.float32)
            npz.close()
        except Exception as e:
            print(f"  [skip] Error loading {npy_path}: {e}")
            skipped += 1
            continue

        try:
            prior = _build_prior_volume_from_prediction(
                pred,
                img=img,
                affine=affine,
                clamp_min_sigma_vox=clamp_min_sigma_vox,
                gaussian_output_space=gaussian_output_space,
            )
        except Exception as e:
            print(f"  [skip] Failed prior build for {patient_id}: {e}")
            skipped += 1
            continue

        # Mask prior to brain region.
        # Use any-channel support; all-channel masking can become overly strict.
        img_mask = (img > 1e-5).any(axis=0).astype(np.float32)
        prior_unmasked_max = float(prior.max())
        prior = prior * img_mask
        if float(prior.max()) <= 0.0 and prior_unmasked_max > 0.0:
            print(
                f"  [warn] Prior became empty after masking for {patient_id}; "
                "saving unmasked prior instead."
            )

        nii_path = os.path.join(output_dir, f"{patient_id}_prior.nii.gz")
        nib.save(nib.Nifti1Image(prior, affine), nii_path)
        saved += 1

    print(f"  Saved {saved} prior NIfTIs to {output_dir} ({skipped} skipped)")


@torch.no_grad()
def generate_deconv_niftis(
    model,
    dataloader,
    case_ids,
    device,
    mri_npy_dir,
    output_dir,
    mask_to_brain=True,
    save_raw=False,
):
    """
    Generate deconv spatial-prior NIfTI files aligned to <patient_id>_preproc.npz.

    Saved volumes are in the exact preprocessed MRI grid and affine from
    ``{mri_npy_dir}/{patient_id}_preproc.npz``. This ensures voxel-perfect
    overlay with the stored MRI volumes (MNI space, cropped shape).

    By default, deconv priors are brain-masked. Set ``save_raw=True`` to
    additionally store the unmasked raw prior map.
    """
    import nibabel as nib

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    saved, skipped = 0, 0
    case_idx = 0

    for x, mask, _ in dataloader:
        x = x.to(device)
        mask = mask.to(device)

        outputs = model(x, mask=mask)
        deconv_spatial = outputs.get("deconv_spatial")
        if deconv_spatial is None:
            raise ValueError(
                "Deconv NIfTI export requested, but model output has no deconv_spatial. "
                "Enable --spatial_head deconv for this checkpoint/model."
            )

        prob_batch = deconv_spatial["prob"].detach().float().cpu()  # [B,1,d,h,w]
        batch_size = prob_batch.shape[0]

        for i in range(batch_size):
            if case_idx >= len(case_ids):
                break

            patient_id = case_ids[case_idx]
            case_idx += 1

            npz_path = os.path.join(mri_npy_dir, f"{patient_id}_preproc.npz")
            if not os.path.exists(npz_path):
                print(f"  [skip] No MRI npz found for {patient_id} at {npz_path}")
                skipped += 1
                continue

            try:
                npz = np.load(npz_path, allow_pickle=True)
                img = np.asarray(npz["image"], dtype=np.float32)  # [C,D,H,W]
                affine = np.asarray(npz["affine"], dtype=np.float32)
                npz.close()
            except Exception as e:
                print(f"  [skip] Error loading {npz_path}: {e}")
                skipped += 1
                continue

            if img.ndim != 4:
                print(f"  [skip] Invalid image shape for {patient_id}: expected [C,D,H,W], got {img.shape}")
                skipped += 1
                continue

            target_shape = tuple(int(s) for s in img.shape[1:])  # (D,H,W), expected 160x192x160
            pred = prob_batch[i : i + 1]  # [1,1,d,h,w]

            if tuple(pred.shape[-3:]) != target_shape:
                pred = F.interpolate(pred, size=target_shape, mode="trilinear", align_corners=False)

            vol_raw = np.clip(pred[0, 0].numpy().astype(np.float32), 0.0, 1.0)
            vol = vol_raw

            if mask_to_brain:
                brain_mask = (img > 1e-5).any(axis=0).astype(np.float32)
                vol = vol_raw * brain_mask

            nii_path = os.path.join(output_dir, f"{patient_id}_deconv_prior.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), nii_path)
            if save_raw:
                nii_raw_path = os.path.join(output_dir, f"{patient_id}_deconv_prior_raw.nii.gz")
                nib.save(nib.Nifti1Image(vol_raw, affine), nii_raw_path)
            saved += 1

    print(f"  Saved {saved} deconv prior NIfTIs to {output_dir} ({skipped} skipped)")


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_array(values: List[Any], dtype=None):
    return np.asarray(values if values else [], dtype=dtype)


def _combined_lobe_label(
    lobe_target: Optional[int],
    lobe_mask: float,
    hemi_target: Optional[int],
    hemi_mask: float,
) -> str:
    if lobe_mask <= 0.5 or lobe_target is None or lobe_target < 0 or lobe_target >= len(LOBE_CLASSES):
        lobe_label = "Unknown"
    else:
        lobe_label = str(LOBE_CLASSES[lobe_target]).capitalize()

    hemi_inv = {v: k for k, v in HEMI_LABEL_TO_INT.items()}
    if hemi_mask <= 0.5 or hemi_target is None or hemi_target not in hemi_inv:
        laterality = "Unknown"
    else:
        laterality = hemi_inv[hemi_target].capitalize()

    if laterality == "Unknown" or lobe_label == "Unknown":
        return "Unknown"
    return f"{laterality} {lobe_label}"


@torch.inference_mode()
def export_feature_vectors(
    model,
    split_to_dataset: Dict[str, MultiHeadTargetDataset],
    target_dict: Dict[str, Dict[str, torch.Tensor]],
    checkpoint_path: str,
    fold_index: int,
    feature_output_dir: Optional[str],
    feature_bag_multiplier: int,
    overwrite_feature_vectors: bool,
    training_bag_size: int,
    minimum_spike_count: int,
    inference_population: str,
):
    if feature_bag_multiplier < 1:
        raise ValueError(f"feature_bag_multiplier must be >= 1, got {feature_bag_multiplier}")
    if training_bag_size < 1:
        raise ValueError(f"training_bag_size must be >= 1, got {training_bag_size}")
    if minimum_spike_count < 1:
        raise ValueError(f"minimum_spike_count must be >= 1, got {minimum_spike_count}")

    if getattr(model, "pooling", None) != "mean":
        raise ValueError(
            "Feature export currently requires --pooling mean so pooled_embedding is the exact mean MIL pooling output. "
            f"Resolved pooling was {getattr(model, 'pooling', None)!r}."
        )

    max_feature_bag_size = int(training_bag_size) * int(feature_bag_multiplier)
    if max_feature_bag_size < minimum_spike_count:
        raise ValueError(
            "maximum feature bag size is below minimum_spike_count: "
            f"{max_feature_bag_size} < {minimum_spike_count}"
        )

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint_name = Path(checkpoint_path).name
    checkpoint_stem = Path(checkpoint_path).stem
    experiment_name = Path(checkpoint_path).resolve().parents[1].name if len(Path(checkpoint_path).resolve().parents) >= 2 else "experiment"
    checkpoint_tag = checkpoint_sha256[:12]

    feature_definition = (
        "Per-spike features are taken exactly immediately before MIL mean pooling (instance_embeddings). "
        "Per-patient pooled features are the direct output exactly immediately after MIL mean pooling (pooled_embedding). "
        "All vectors come from one shared selected checkpoint. Prediction-head logits and deconvolution outputs are not included."
    )

    embedding_dim_for_id = int(getattr(model, "emb_dim", -1))
    feature_space_payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "model_class": model.__class__.__name__,
        "feature_definition": feature_definition,
        "embedding_dim": embedding_dim_for_id,
    }
    feature_space_id = hashlib.sha256(
        json.dumps(feature_space_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    base_dir = Path(feature_output_dir) if feature_output_dir is not None else Path(checkpoint_path).resolve().parent / "feature_vectors"
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(checkpoint_path).stem.replace(" ", "_")
    if len(checkpoint_stem) > 48:
        checkpoint_stem = f"{checkpoint_stem[:32]}_{hashlib.sha1(checkpoint_stem.encode('utf-8')).hexdigest()[:8]}"
    file_base = f"features_{inference_population}_{checkpoint_stem}_{checkpoint_tag}"
    npz_path = base_dir / f"{file_base}.npz"
    json_path = base_dir / f"{file_base}.json"

    if (npz_path.exists() or json_path.exists()) and not overwrite_feature_vectors:
        raise FileExistsError(
            "Feature export already exists. Use --overwrite_feature_vectors to overwrite. "
            f"npz={npz_path}, json={json_path}"
        )

    model.eval()
    rng = np.random.default_rng()

    spike_features = []
    spike_subject_ids = []
    spike_fold = []
    spike_split = []
    spike_original_indices = []
    spike_bag_positions = []
    spike_lobe_labels_num = []
    spike_laterality_labels_num = []
    spike_lobe_labels = []
    spike_laterality_labels = []
    spike_combined_lobe_labels = []

    pooled_features = []
    pooled_subject_ids = []
    pooled_fold = []
    pooled_split = []
    pooled_n_selected_spikes = []
    pooled_lobe_labels_num = []
    pooled_laterality_labels_num = []
    pooled_lobe_labels = []
    pooled_laterality_labels = []
    pooled_combined_lobe_labels = []

    skipped_subjects = []
    seen_subjects = {}
    split_counts = {}

    embedding_dim = None

    for split_name, dataset in split_to_dataset.items():
        split_counts[split_name] = len(dataset.patient_ids)
        base = dataset.base_dataset
        for local_idx, patient_id in enumerate(dataset.patient_ids):
            if patient_id in seen_subjects:
                raise ValueError(
                    f"Subject {patient_id!r} appears in multiple splits: {seen_subjects[patient_id]!r} and {split_name!r}"
                )
            seen_subjects[patient_id] = split_name

            spikes_full = base.patients_data[local_idx]  # [N, C, L]
            n_spikes_total = int(spikes_full.shape[0])
            if n_spikes_total < minimum_spike_count:
                skipped_subjects.append({
                    "subject_id": patient_id,
                    "reason": f"n_spikes_total={n_spikes_total} < minimum_spike_count={minimum_spike_count}",
                    "split": split_name,
                })
                continue

            n_selected = min(n_spikes_total, max_feature_bag_size)
            if n_selected < minimum_spike_count:
                skipped_subjects.append({
                    "subject_id": patient_id,
                    "reason": f"n_selected={n_selected} < minimum_spike_count={minimum_spike_count}",
                    "split": split_name,
                })
                continue

            selected_idx = rng.choice(n_spikes_total, size=n_selected, replace=False)
            if np.unique(selected_idx).shape[0] != selected_idx.shape[0]:
                raise ValueError(f"Duplicate spike indices selected for subject {patient_id!r}")

            spikes = spikes_full[selected_idx]
            _, _, seg_len = spikes.shape
            crop_start = seg_len // 2 - base.window_size // 2
            crop_end = crop_start + base.window_size
            spikes = spikes[:, :, crop_start:crop_end]

            x = torch.tensor(spikes, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0)
            mask = torch.ones((1, n_selected), dtype=torch.float32, device=x.device)

            outputs = model(x, mask=mask, return_features=True)
            features = outputs.get("features", None)
            if features is None:
                raise RuntimeError("Model did not return features during feature export.")

            instance = features["instance_embeddings"]
            pooled = features["pooled_embedding"]

            if instance.ndim != 3:
                raise ValueError(f"Expected instance_embeddings with shape [B,N,D], got {tuple(instance.shape)}")
            if pooled.ndim != 2:
                raise ValueError(f"Expected pooled_embedding with shape [B,D], got {tuple(pooled.shape)}")
            if instance.shape[0] != 1 or pooled.shape[0] != 1:
                raise ValueError(
                    "Feature export uses one subject per forward pass; expected batch size 1, got "
                    f"instance={tuple(instance.shape)}, pooled={tuple(pooled.shape)}"
                )
            if instance.shape[1] != n_selected:
                raise ValueError(
                    f"instance_embeddings bag size mismatch for {patient_id!r}: "
                    f"expected {n_selected}, got {instance.shape[1]}"
                )
            if pooled.shape[1] != instance.shape[2]:
                raise ValueError(
                    f"Embedding dimension mismatch for {patient_id!r}: pooled D={pooled.shape[1]}, instance D={instance.shape[2]}"
                )

            pooled_expected = (instance * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            if not torch.allclose(pooled, pooled_expected, atol=1e-5, rtol=1e-4):
                raise ValueError(
                    f"Pooled embedding consistency check failed for subject {patient_id!r}; "
                    "pooled_embedding does not match masked mean of instance_embeddings."
                )

            instance_np = instance[0].detach().cpu().numpy().astype(np.float32)
            pooled_np = pooled[0].detach().cpu().numpy().astype(np.float32)

            if embedding_dim is None:
                embedding_dim = int(instance_np.shape[1])
            if int(instance_np.shape[1]) != embedding_dim or int(pooled_np.shape[0]) != embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensionality for {patient_id!r}: "
                    f"instance D={instance_np.shape[1]}, pooled D={pooled_np.shape[0]}, expected D={embedding_dim}"
                )

            t = target_dict[patient_id]
            lobe_mask = float(t["lobe_mask"].item())
            hemi_mask = float(t["hemi_mask"].item())
            lobe_num = int(t["lobe_target"].item()) if lobe_mask > 0.5 else -1
            hemi_num = int(t["hemi_target"].item()) if hemi_mask > 0.5 else -1

            lobe_label = LOBE_CLASSES[lobe_num] if (0 <= lobe_num < len(LOBE_CLASSES)) else "Unknown"
            hemi_label_inv = {v: k for k, v in HEMI_LABEL_TO_INT.items()}
            hemi_label = hemi_label_inv[hemi_num] if hemi_num in hemi_label_inv else "Unknown"
            combined_label = _combined_lobe_label(lobe_num, lobe_mask, hemi_num, hemi_mask)

            for bag_pos in range(n_selected):
                spike_features.append(instance_np[bag_pos])
                spike_subject_ids.append(patient_id)
                spike_fold.append(int(fold_index))
                spike_split.append(split_name)
                spike_original_indices.append(int(selected_idx[bag_pos]))
                spike_bag_positions.append(int(bag_pos))
                spike_lobe_labels_num.append(int(lobe_num))
                spike_laterality_labels_num.append(int(hemi_num))
                spike_lobe_labels.append(lobe_label)
                spike_laterality_labels.append(hemi_label)
                spike_combined_lobe_labels.append(combined_label)

                # Spike-level labels should match patient-level labels.
                if spike_lobe_labels_num[-1] != lobe_num or spike_laterality_labels_num[-1] != hemi_num:
                    raise ValueError(f"Spike-level label mismatch for subject {patient_id!r}")

            pooled_features.append(pooled_np)
            pooled_subject_ids.append(patient_id)
            pooled_fold.append(int(fold_index))
            pooled_split.append(split_name)
            pooled_n_selected_spikes.append(int(n_selected))
            pooled_lobe_labels_num.append(int(lobe_num))
            pooled_laterality_labels_num.append(int(hemi_num))
            pooled_lobe_labels.append(lobe_label)
            pooled_laterality_labels.append(hemi_label)
            pooled_combined_lobe_labels.append(combined_label)

    if len(spike_features) != int(sum(pooled_n_selected_spikes)):
        raise ValueError(
            "Spike row count mismatch: "
            f"len(spike_features)={len(spike_features)} vs sum(pooled_n_selected_spikes)={int(sum(pooled_n_selected_spikes))}"
        )
    if len(pooled_features) != len(pooled_subject_ids):
        raise ValueError("Pooled feature row count mismatch with patient count")

    npz_payload = {
        "spike_features": _safe_array(spike_features, dtype=np.float32),
        "spike_subject_ids": _safe_array(spike_subject_ids, dtype=str),
        "spike_fold": _safe_array(spike_fold, dtype=np.int32),
        "spike_split": _safe_array(spike_split, dtype=str),
        "spike_original_indices": _safe_array(spike_original_indices, dtype=np.int64),
        "spike_bag_positions": _safe_array(spike_bag_positions, dtype=np.int32),
        "spike_lobe_labels": _safe_array(spike_lobe_labels_num, dtype=np.int32),
        "spike_laterality_labels": _safe_array(spike_laterality_labels_num, dtype=np.int32),
        "spike_lobe_label_names": _safe_array(spike_lobe_labels, dtype=str),
        "spike_laterality_label_names": _safe_array(spike_laterality_labels, dtype=str),
        "spike_combined_lobe_labels": _safe_array(spike_combined_lobe_labels, dtype=str),
        "pooled_features": _safe_array(pooled_features, dtype=np.float32),
        "pooled_subject_ids": _safe_array(pooled_subject_ids, dtype=str),
        "pooled_fold": _safe_array(pooled_fold, dtype=np.int32),
        "pooled_split": _safe_array(pooled_split, dtype=str),
        "pooled_n_selected_spikes": _safe_array(pooled_n_selected_spikes, dtype=np.int32),
        "pooled_lobe_labels": _safe_array(pooled_lobe_labels_num, dtype=np.int32),
        "pooled_laterality_labels": _safe_array(pooled_laterality_labels_num, dtype=np.int32),
        "pooled_lobe_label_names": _safe_array(pooled_lobe_labels, dtype=str),
        "pooled_laterality_label_names": _safe_array(pooled_laterality_labels, dtype=str),
        "pooled_combined_lobe_labels": _safe_array(pooled_combined_lobe_labels, dtype=str),
    }

    np.savez_compressed(npz_path, **npz_payload)

    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script_name": Path(__file__).name,
        "experiment_name": experiment_name,
        "fold": int(fold_index),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_filename": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "model_class": model.__class__.__name__,
        "embedding_dim": int(embedding_dim if embedding_dim is not None else 0),
        "training_bag_size": int(training_bag_size),
        "feature_bag_multiplier": int(feature_bag_multiplier),
        "maximum_feature_bag_size": int(max_feature_bag_size),
        "minimum_spike_count": int(minimum_spike_count),
        "inference_population": inference_population,
        "number_of_patients_exported": int(len(pooled_subject_ids)),
        "number_of_patients_skipped": int(len(skipped_subjects)),
        "number_of_spikes_exported": int(len(spike_subject_ids)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "available_event_metadata": [],
        "feature_definition": feature_definition,
        "feature_space_id": feature_space_id,
        "skipped_subjects": skipped_subjects,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Feature export complete:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  fold: {fold_index}")
    print(f"  training bag size: {training_bag_size}")
    print(f"  export cap (max spikes per subject): {max_feature_bag_size}")
    print(f"  split counts: {split_counts}")
    print(f"  exported subjects: {len(pooled_subject_ids)}")
    print(f"  skipped subjects: {len(skipped_subjects)}")
    print(f"  total selected spikes: {len(spike_subject_ids)}")
    print(f"  npz: {npz_path}")
    print(f"  metadata: {json_path}")

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "feature_space_id": feature_space_id,
    }


def run_inference(
    checkpoint_path,
    json_split_path,
    fold_index,
    data_dir,
    json_targets_path,
    lobe_json_path=None,
    hemi_json_path=None,
    output_dir=None,
    in_channels=21,
    max_spikes=32,
    min_spikes_per_patient=64,
    batch_size=4,
    emb_dim=None,
    hidden=None,
    dropout=None,
    num_workers=0,
    encoder_type=None,
    pooling=None,
    spatial_head=None,
    num_gaussians=None,
    gaussian_coord_dim=None,
    gaussian_sigma_min=None,
    gaussian_sigma_max=None,
    gaussian_isotropic=None,
    gaussian_output_space=None,
    gaussian_make_heatmap=None,
    gaussian_heatmap_shape=None,
    deconv_output_shape=None,
    deconv_latent_shape=None,
    deconv_base_channels=None,
    deconv_dropout=None,
    test_mode=False,
    infer_test_set=False,
    infer_all_subjects=False,
    store_feature_vectors=False,
    feature_output_dir=None,
    feature_bag_multiplier=4,
    overwrite_feature_vectors=False,
    generate_niftis=False,
    mri_npy_dir=None,
    deconv_save_raw_prior_nifti=False,
):
    """
    Load a trained multi-head model and run inference.
    Default: train/val folds. Optionally includes test set or all fold subjects.
    Saves combined predictions and validation metrics to JSON files.

    Optionally generates per-patient Gaussian prior NIfTI files from the
    predicted (mu, sigma) pairs when ``generate_niftis=True``. In that case
    ``mri_npy_dir`` must point to the directory that contains the preprocessed
    ``{patient_id}_preproc.npz`` files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore head flags from saved args (fallback to True for old checkpoints).
    saved_args = checkpoint.get("args", {})
    use_coord_head = saved_args.get("use_coord_head", True)
    use_hemi_head = saved_args.get("use_hemi_head", True)
    use_lobe_head = saved_args.get("use_lobe_head", True)

    print(
        f"Active heads — coord: {use_coord_head}, "
        f"hemi: {use_hemi_head}, lobe: {use_lobe_head}"
    )
    resolved_output_space = saved_args.get("gaussian_output_space", "normalized" if gaussian_output_space is None else gaussian_output_space)
    resolved_sigma_min, resolved_sigma_max = resolve_gaussian_sigma_bounds(
        gaussian_output_space=resolved_output_space,
        gaussian_sigma_min=saved_args.get("gaussian_sigma_min", gaussian_sigma_min),
        gaussian_sigma_max=saved_args.get("gaussian_sigma_max", gaussian_sigma_max),
    )

    # Restore model architecture from checkpoint; explicit args override saved values.
    model_kwargs = {k: v for k, v in dict(
        emb_dim=saved_args.get("emb_dim", emb_dim),
        hidden=saved_args.get("hidden", hidden),
        dropout=saved_args.get("dropout", dropout),
        encoder_type=saved_args.get("encoder_type", encoder_type),
        pooling=saved_args.get("pooling", pooling),
        spatial_head=saved_args.get("spatial_head", "none" if spatial_head is None else spatial_head),
        num_gaussians=saved_args.get("num_gaussians", 3 if num_gaussians is None else num_gaussians),
        gaussian_coord_dim=saved_args.get("gaussian_coord_dim", 3 if gaussian_coord_dim is None else gaussian_coord_dim),
        gaussian_sigma_min=resolved_sigma_min,
        gaussian_sigma_max=resolved_sigma_max,
        gaussian_isotropic=saved_args.get("gaussian_isotropic", True if gaussian_isotropic is None else gaussian_isotropic),
        gaussian_output_space=saved_args.get("gaussian_output_space", "normalized" if gaussian_output_space is None else gaussian_output_space),
        gaussian_make_heatmap=saved_args.get("gaussian_make_heatmap", False if gaussian_make_heatmap is None else gaussian_make_heatmap),
        gaussian_heatmap_shape=saved_args.get("gaussian_heatmap_shape", gaussian_heatmap_shape),
        deconv_output_shape=saved_args.get("deconv_output_shape", deconv_output_shape),
        deconv_latent_shape=saved_args.get("deconv_latent_shape", deconv_latent_shape),
        deconv_base_channels=saved_args.get("deconv_base_channels", deconv_base_channels),
        deconv_dropout=saved_args.get("deconv_dropout", deconv_dropout),
    ).items() if v is not None}
    # Explicit non-None args always win over saved args.
    for k, v in dict(emb_dim=emb_dim, hidden=hidden, dropout=dropout,
                     encoder_type=encoder_type, pooling=pooling,
                     spatial_head=spatial_head,
                     num_gaussians=num_gaussians,
                     gaussian_coord_dim=gaussian_coord_dim,
                     gaussian_sigma_min=gaussian_sigma_min,
                     gaussian_sigma_max=gaussian_sigma_max,
                     gaussian_isotropic=gaussian_isotropic,
                     gaussian_output_space=gaussian_output_space,
                     gaussian_make_heatmap=gaussian_make_heatmap,
                     gaussian_heatmap_shape=gaussian_heatmap_shape,
                     deconv_output_shape=deconv_output_shape,
                     deconv_latent_shape=deconv_latent_shape,
                     deconv_base_channels=deconv_base_channels,
                     deconv_dropout=deconv_dropout).items():
        if v is not None:
            model_kwargs[k] = v
    print(f"Model kwargs (from checkpoint + overrides): {model_kwargs}")
    model = SpikeMILModel(
        in_channels=in_channels,
        n_hemi_classes=len(HEMI_LABEL_TO_INT),
        n_lobe_classes=len(LOBE_CLASSES),
        use_coord_head=use_coord_head,
        use_hemi_head=use_hemi_head,
        use_lobe_head=use_lobe_head,
        **model_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")

    if infer_all_subjects and infer_test_set:
        raise ValueError(
            "--infer_all_subjects and --infer_test_set cannot be used together. "
            "Use --infer_all_subjects to include all fold memberships in one run."
        )

    train_ids, val_ids = load_split(json_split_path, fold_index)
    test_ids = []
    with open(json_split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    fold_payload = payload["folds"]
    fold_key = f"fold_{fold_index}"
    if fold_key not in fold_payload:
        raise ValueError(f"{fold_key} not found in {json_split_path}.")
    fold_entry = fold_payload[fold_key]
    test_ids = fold_entry.get("test_ids", [])

    split_id_map: Dict[str, List[str]] = {
        "train": list(train_ids),
        "val": list(val_ids),
    }
    if infer_all_subjects:
        split_id_map["test"] = list(test_ids)
        inference_population = "all_subjects"
    elif infer_test_set:
        split_id_map["test"] = list(test_ids)
        inference_population = "train_val_test"
    else:
        inference_population = "train_val"

    target_dict = load_multitask_targets(
        json_targets_path,
        lobe_json_path=lobe_json_path,
        hemi_json_path=hemi_json_path,
    )

    split_found = {}
    for split_name, ids in split_id_map.items():
        split_found[split_name] = find_patient_files(
            data_dir, ids, target_dict, test_mode=test_mode
        )

    split_found_counts = {k: len(v[0]) for k, v in split_found.items()}
    print(f"Inference population: {inference_population}")
    print(f"Subjects found per split: {split_found_counts}")

    resolved_spatial_head = model_kwargs.get("spatial_head", saved_args.get("spatial_head", "none"))
    deconv_enabled_for_inference = bool(
        resolved_spatial_head == "deconv" and model_kwargs.get("deconv_output_shape") is not None
    )

    split_to_dataset: Dict[str, MultiHeadTargetDataset] = {}
    split_to_loader: Dict[str, DataLoader] = {}

    for split_name, (ids_found, files, targets) in split_found.items():
        if len(ids_found) == 0:
            print(f"Skipping split {split_name!r} because it has zero valid subjects.")
            continue
        base_ds = PatientMILSpikeDataset(
            ids_found,
            files,
            targets,
            max_spikes_per_bag=max_spikes,
            training=False,
            min_spikes_per_patient=min_spikes_per_patient,
        )
        ds = MultiHeadTargetDataset(
            base_ds,
            target_by_pid=target_dict,
            deconv_enabled=deconv_enabled_for_inference,
            deconv_output_shape=model_kwargs.get("deconv_output_shape"),
            deconv_target_blur_sigma=float(saved_args.get("deconv_target_blur_sigma", 0.0)),
            deconv_require_mni_alignment=deconv_enabled_for_inference,
            deconv_mask_npz_dir=mri_npy_dir if deconv_enabled_for_inference else None,
        )
        split_to_dataset[split_name] = ds
        split_to_loader[split_name] = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=mil_multitask_collate,
            pin_memory=True,
        )

    if "train" not in split_to_dataset or "val" not in split_to_dataset:
        raise ValueError(
            "Inference requires non-empty train and val split datasets after filtering. "
            f"Got splits: {sorted(split_to_dataset.keys())}"
        )

    train_dataset = split_to_dataset["train"]
    val_dataset = split_to_dataset["val"]
    train_loader = split_to_loader["train"]
    val_loader = split_to_loader["val"]
    test_dataset = split_to_dataset.get("test")
    test_loader = split_to_loader.get("test")

    split_to_preds: Dict[str, Dict[str, Any]] = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in split_to_loader:
            continue
        print(f"\nRunning inference on {split_name} set...")
        preds = infer_predictions(model, split_to_loader[split_name], split_to_dataset[split_name].patient_ids, device)
        split_to_preds[split_name] = preds
        print(f"Got predictions for {len(preds)} {split_name} cases")

    train_preds = split_to_preds.get("train", {})
    val_preds = split_to_preds.get("val", {})
    test_preds = split_to_preds.get("test", {})

    if store_feature_vectors:
        export_feature_vectors(
            model=model,
            split_to_dataset=split_to_dataset,
            target_dict=target_dict,
            checkpoint_path=checkpoint_path,
            fold_index=fold_index,
            feature_output_dir=feature_output_dir,
            feature_bag_multiplier=feature_bag_multiplier,
            overwrite_feature_vectors=overwrite_feature_vectors,
            training_bag_size=int(saved_args.get("max_spikes", max_spikes)),
            minimum_spike_count=min_spikes_per_patient,
            inference_population=inference_population,
        )

    # Per-case and aggregate validation metrics
    val_metrics = {}
    val_summary = {
        "num_cases": 0,
        "coord_euclidean_norm_mean": None, "coord_euclidean_mm_mean": None,
        "hemi_acc": None, "lobe_acc": None,
        "hemi_labeled_cases": 0, "lobe_labeled_cases": 0,
        "deconv_pred_mean": None,
        "deconv_pred_max": None,
        "deconv_dice": None,
        "deconv_mass_in_gt": None,
        "deconv_peak_distance": None,
        "deconv_topk_hit": None,
        "deconv_pred_mean_inside_brain": None,
        "deconv_effective_volume_voxels": None,
        "deconv_soft_bce": None,
        "deconv_coverage_loss": None,
        "deconv_mass_loss": None,
        "deconv_coverage_value": None,
        "deconv_mass_value": None,
        "deconv_labeled_cases": 0,
    }

    coord_euclidean_norm_vals = []
    coord_euclidean_mm_vals = []
    hemi_correct = []
    lobe_correct = []
    deconv_pred_mean_vals = []
    deconv_pred_max_vals = []
    deconv_dice_vals = []
    deconv_mass_in_gt_vals = []
    deconv_peak_distance_vals = []
    deconv_topk_hit_vals = []
    deconv_pred_mean_inside_brain_vals = []
    deconv_effective_volume_vals = []
    deconv_soft_bce_vals = []
    deconv_coverage_loss_vals = []
    deconv_mass_loss_vals = []
    deconv_coverage_value_vals = []
    deconv_mass_value_vals = []

    for case_id in val_dataset.patient_ids:
        if case_id not in val_preds:
            continue

        pred = val_preds[case_id]
        target = target_dict[case_id]

        case_metrics: dict = {}

        if use_coord_head and pred["mu"] is not None:
            pred_mu = torch.tensor(pred["mu"], dtype=torch.float32)
            gt_mu = target["mu"].float()
            euclidean_norm = torch.norm(pred_mu - gt_mu).item()
            diff_mm = (pred_mu - gt_mu) * MNI_EXTENT_MM
            euclidean_mm = torch.norm(diff_mm).item()
            coord_euclidean_norm_vals.append(euclidean_norm)
            coord_euclidean_mm_vals.append(euclidean_mm)
            case_metrics.update({
                "coord_euclidean_norm": euclidean_norm,
                "coord_euclidean_mm": euclidean_mm,
                "pred_mu": pred["mu"],
                "pred_sigma": pred["sigma"],
                "gt_mu": gt_mu.tolist(),
            })
        else:
            case_metrics.update({
                "coord_euclidean_norm": None, "coord_euclidean_mm": None,
                "pred_mu": None, "pred_sigma": None, "gt_mu": None,
            })

        hemi_is_labeled = bool(target["hemi_mask"].item() > 0.5)
        lobe_is_labeled = bool(target["lobe_mask"].item() > 0.5)

        case_hemi_acc = None
        case_lobe_acc = None

        if use_hemi_head and pred["hemi_pred"] is not None and hemi_is_labeled:
            case_hemi_acc = float(pred["hemi_pred"] == int(target["hemi_target"].item()))
            hemi_correct.append(case_hemi_acc)

        if use_lobe_head and pred["lobe_pred"] is not None and lobe_is_labeled:
            case_lobe_acc = float(pred["lobe_pred"] == int(target["lobe_target"].item()))
            lobe_correct.append(case_lobe_acc)

        case_metrics.update({
            "pred_hemi": pred["hemi_pred"],
            "gt_hemi": int(target["hemi_target"].item()) if (use_hemi_head and hemi_is_labeled) else None,
            "hemi_acc": case_hemi_acc,
            "pred_lobe": pred["lobe_pred"],
            "gt_lobe": int(target["lobe_target"].item()) if (use_lobe_head and lobe_is_labeled) else None,
            "lobe_acc": case_lobe_acc,
        })

        deconv_stats = pred.get("deconv_spatial_stats", None)
        if deconv_stats is not None:
            case_metrics["deconv_pred_mean"] = deconv_stats.get("pred_mean")
            case_metrics["deconv_pred_max"] = deconv_stats.get("pred_max")
            case_metrics["deconv_dice"] = deconv_stats.get("dice")
            case_metrics["deconv_mass_in_gt"] = deconv_stats.get("mass_in_gt")
            case_metrics["deconv_peak_distance"] = deconv_stats.get("peak_distance")
            case_metrics["deconv_topk_hit"] = deconv_stats.get("topk_hit")
            case_metrics["deconv_pred_mean_inside_brain"] = deconv_stats.get("pred_mean_inside_brain")
            case_metrics["deconv_effective_volume_voxels"] = deconv_stats.get("effective_volume_voxels")
            case_metrics["deconv_soft_bce"] = deconv_stats.get("soft_bce")
            case_metrics["deconv_coverage_loss"] = deconv_stats.get("coverage_loss")
            case_metrics["deconv_mass_loss"] = deconv_stats.get("mass_loss")
            case_metrics["deconv_coverage_value"] = deconv_stats.get("coverage_value")
            case_metrics["deconv_mass_value"] = deconv_stats.get("mass_value")

            if deconv_stats.get("pred_mean") is not None:
                deconv_pred_mean_vals.append(float(deconv_stats["pred_mean"]))
            if deconv_stats.get("pred_max") is not None:
                deconv_pred_max_vals.append(float(deconv_stats["pred_max"]))
            if deconv_stats.get("dice") is not None:
                deconv_dice_vals.append(float(deconv_stats["dice"]))
            if deconv_stats.get("mass_in_gt") is not None:
                deconv_mass_in_gt_vals.append(float(deconv_stats["mass_in_gt"]))
            if deconv_stats.get("peak_distance") is not None:
                deconv_peak_distance_vals.append(float(deconv_stats["peak_distance"]))
            if deconv_stats.get("topk_hit") is not None:
                deconv_topk_hit_vals.append(float(deconv_stats["topk_hit"]))
            if deconv_stats.get("pred_mean_inside_brain") is not None:
                deconv_pred_mean_inside_brain_vals.append(float(deconv_stats["pred_mean_inside_brain"]))
            if deconv_stats.get("effective_volume_voxels") is not None:
                deconv_effective_volume_vals.append(float(deconv_stats["effective_volume_voxels"]))
            if deconv_stats.get("soft_bce") is not None:
                deconv_soft_bce_vals.append(float(deconv_stats["soft_bce"]))
            if deconv_stats.get("coverage_loss") is not None:
                deconv_coverage_loss_vals.append(float(deconv_stats["coverage_loss"]))
            if deconv_stats.get("mass_loss") is not None:
                deconv_mass_loss_vals.append(float(deconv_stats["mass_loss"]))
            if deconv_stats.get("coverage_value") is not None:
                deconv_coverage_value_vals.append(float(deconv_stats["coverage_value"]))
            if deconv_stats.get("mass_value") is not None:
                deconv_mass_value_vals.append(float(deconv_stats["mass_value"]))
        else:
            case_metrics.update(
                {
                    "deconv_pred_mean": None,
                    "deconv_pred_max": None,
                    "deconv_dice": None,
                    "deconv_mass_in_gt": None,
                    "deconv_peak_distance": None,
                    "deconv_topk_hit": None,
                    "deconv_pred_mean_inside_brain": None,
                    "deconv_effective_volume_voxels": None,
                    "deconv_soft_bce": None,
                    "deconv_coverage_loss": None,
                    "deconv_mass_loss": None,
                    "deconv_coverage_value": None,
                    "deconv_mass_value": None,
                }
            )

        val_metrics[case_id] = case_metrics

    val_summary["num_cases"] = len(val_dataset.patient_ids)
    if coord_euclidean_norm_vals:
        val_summary["coord_euclidean_norm_mean"] = float(sum(coord_euclidean_norm_vals) / len(coord_euclidean_norm_vals))
        print(f"\nValidation coordinate error (mean Euclidean norm): {val_summary['coord_euclidean_norm_mean']:.4f}")
    if coord_euclidean_mm_vals:
        val_summary["coord_euclidean_mm_mean"] = float(sum(coord_euclidean_mm_vals) / len(coord_euclidean_mm_vals))
        print(f"\nValidation coordinate error (mean Euclidean distance): {val_summary['coord_euclidean_mm_mean']:.2f} mm")
    if hemi_correct:
        val_summary["hemi_acc"] = float(sum(hemi_correct) / len(hemi_correct))
        val_summary["hemi_labeled_cases"] = len(hemi_correct)
        print(f"\nValidation hemisphere accuracy: {val_summary['hemi_acc']:.3f} over {val_summary['hemi_labeled_cases']} labeled cases")
    if lobe_correct:
        val_summary["lobe_acc"] = float(sum(lobe_correct) / len(lobe_correct))
        val_summary["lobe_labeled_cases"] = len(lobe_correct)
        print(f"\nValidation lobe accuracy: {val_summary['lobe_acc']:.3f} over {val_summary['lobe_labeled_cases']} labeled cases")
    if deconv_pred_mean_vals:
        val_summary["deconv_pred_mean"] = float(sum(deconv_pred_mean_vals) / len(deconv_pred_mean_vals))
    if deconv_pred_max_vals:
        val_summary["deconv_pred_max"] = float(sum(deconv_pred_max_vals) / len(deconv_pred_max_vals))
    if deconv_dice_vals:
        val_summary["deconv_dice"] = float(sum(deconv_dice_vals) / len(deconv_dice_vals))
        val_summary["deconv_labeled_cases"] = len(deconv_dice_vals)
    if deconv_mass_in_gt_vals:
        val_summary["deconv_mass_in_gt"] = float(sum(deconv_mass_in_gt_vals) / len(deconv_mass_in_gt_vals))
    if deconv_peak_distance_vals:
        val_summary["deconv_peak_distance"] = float(sum(deconv_peak_distance_vals) / len(deconv_peak_distance_vals))
    if deconv_topk_hit_vals:
        val_summary["deconv_topk_hit"] = float(sum(deconv_topk_hit_vals) / len(deconv_topk_hit_vals))
    if deconv_pred_mean_inside_brain_vals:
        val_summary["deconv_pred_mean_inside_brain"] = float(sum(deconv_pred_mean_inside_brain_vals) / len(deconv_pred_mean_inside_brain_vals))
    if deconv_effective_volume_vals:
        val_summary["deconv_effective_volume_voxels"] = float(sum(deconv_effective_volume_vals) / len(deconv_effective_volume_vals))
    if deconv_soft_bce_vals:
        val_summary["deconv_soft_bce"] = float(sum(deconv_soft_bce_vals) / len(deconv_soft_bce_vals))
    if deconv_coverage_loss_vals:
        val_summary["deconv_coverage_loss"] = float(sum(deconv_coverage_loss_vals) / len(deconv_coverage_loss_vals))
    if deconv_mass_loss_vals:
        val_summary["deconv_mass_loss"] = float(sum(deconv_mass_loss_vals) / len(deconv_mass_loss_vals))
    if deconv_coverage_value_vals:
        val_summary["deconv_coverage_value"] = float(sum(deconv_coverage_value_vals) / len(deconv_coverage_value_vals))
    if deconv_mass_value_vals:
        val_summary["deconv_mass_value"] = float(sum(deconv_mass_value_vals) / len(deconv_mass_value_vals))
    if val_summary["deconv_labeled_cases"] > 0:
        print(
            "\nValidation deconv metrics: "
            f"Dice={val_summary['deconv_dice']:.4f}, "
            f"MassInGT={val_summary['deconv_mass_in_gt']:.4f}, "
            f"PeakDist={val_summary['deconv_peak_distance']:.2f}, "
            f"TopKHit={val_summary['deconv_topk_hit']:.3f}, "
            f"N={val_summary['deconv_labeled_cases']}"
        )

    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    inference_settings = {
        "checkpoint_path": checkpoint_path, "json_split_path": json_split_path, "fold_index": fold_index,
        "data_dir": data_dir, "json_targets_path": json_targets_path,
        "lobe_json_path": lobe_json_path, "hemi_json_path": hemi_json_path,
        "in_channels": in_channels, "max_spikes": max_spikes,
        "min_spikes_per_patient": min_spikes_per_patient, "batch_size": batch_size,
        "emb_dim": emb_dim, "hidden": hidden, "dropout": dropout, "num_workers": num_workers,
        "encoder_type": encoder_type, "pooling": pooling,
        "spatial_head": spatial_head, "num_gaussians": num_gaussians, "gaussian_coord_dim": gaussian_coord_dim,
        "gaussian_sigma_min": gaussian_sigma_min, "gaussian_sigma_max": gaussian_sigma_max,
        "gaussian_isotropic": gaussian_isotropic, "gaussian_output_space": gaussian_output_space,
        "gaussian_make_heatmap": gaussian_make_heatmap, "gaussian_heatmap_shape": gaussian_heatmap_shape,
        "deconv_output_shape": deconv_output_shape, "deconv_latent_shape": deconv_latent_shape,
        "deconv_base_channels": deconv_base_channels, "deconv_dropout": deconv_dropout,
        "test_mode": test_mode,
        "infer_test_set": bool(infer_test_set),
        "infer_all_subjects": bool(infer_all_subjects),
        "store_feature_vectors": bool(store_feature_vectors),
        "feature_output_dir": feature_output_dir,
        "feature_bag_multiplier": int(feature_bag_multiplier),
        "overwrite_feature_vectors": bool(overwrite_feature_vectors),
        "inference_population": inference_population,
        "generate_niftis": generate_niftis, "mri_npy_dir": mri_npy_dir,
        "deconv_save_raw_prior_nifti": deconv_save_raw_prior_nifti,
        "use_coord_head": use_coord_head, "use_hemi_head": use_hemi_head, "use_lobe_head": use_lobe_head,
        "resolved_model_kwargs": model_kwargs, "saved_train_args": saved_args,
    }
    inference_settings_path = os.path.join(output_dir, "inference_settings.json")
    with open(inference_settings_path, "w") as f:
        json.dump(inference_settings, f, indent=2)
    print(f"Saved inference settings to: {inference_settings_path}")

    output_path = os.path.join(output_dir, "predictions.json")
    predictions_payload = {"train": train_preds, "val": val_preds}
    if test_preds:
        predictions_payload["test"] = test_preds
    with open(output_path, "w") as f:
        json.dump(predictions_payload, f, indent=2)
    print(f"Saved predictions to: {output_path}")

    metrics_path = os.path.join(output_dir, "validation.json")
    with open(metrics_path, "w") as f:
        json.dump({"summary": val_summary, "cases": val_metrics}, f, indent=2)
    print(f"Saved validation metrics to: {metrics_path}")

    # Export per-subject deconv diagnostics for both train and validation splits.
    deconv_metrics_csv_path = os.path.join(output_dir, "deconv_inference_metrics.csv")
    csv_fields = [
        "split",
        "case_id",
        "pred_mean",
        "pred_max",
        "pred_mean_inside_brain",
        "effective_volume_voxels",
        "dice",
        "mass_in_gt",
        "peak_distance",
        "topk_hit",
        "target_soft_max",
        "target_soft_mean",
        "soft_bce",
        "coverage_loss",
        "mass_loss",
        "coverage_value",
        "mass_value",
    ]
    with open(deconv_metrics_csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=csv_fields)
        writer_csv.writeheader()
        split_items = [("train", train_preds), ("val", val_preds)]
        if test_preds:
            split_items.append(("test", test_preds))
        for split_name, split_preds in split_items:
            for case_id, pred in split_preds.items():
                stats = pred.get("deconv_spatial_stats") or {}
                row = {"split": split_name, "case_id": case_id}
                for key in csv_fields[2:]:
                    row[key] = stats.get(key)
                writer_csv.writerow(row)
    print(f"Saved deconv inference metrics CSV to: {deconv_metrics_csv_path}")

    # Optionally generate Gaussian prior NIfTI files.
    if generate_niftis:
        if mri_npy_dir is None:
            print(
                "\033[38;5;208mWarning: generate_niftis=True but mri_npy_dir is None. "
                "Skipping NIfTI generation.\033[0m"
            )
        else:
            if resolved_spatial_head == "deconv":
                print("\nGenerating deconv prior NIfTIs for training set...")
                generate_deconv_niftis(
                    model,
                    train_loader,
                    train_dataset.patient_ids,
                    device=device,
                    mri_npy_dir=mri_npy_dir,
                    output_dir=os.path.join(output_dir, "prior_niftis", "train"),
                    mask_to_brain=True,
                    save_raw=deconv_save_raw_prior_nifti,
                )
                print("Generating deconv prior NIfTIs for validation set...")
                generate_deconv_niftis(
                    model,
                    val_loader,
                    val_dataset.patient_ids,
                    device=device,
                    mri_npy_dir=mri_npy_dir,
                    output_dir=os.path.join(output_dir, "prior_niftis", "val"),
                    mask_to_brain=True,
                    save_raw=deconv_save_raw_prior_nifti,
                )
                if test_loader is not None and test_dataset is not None:
                    print("Generating deconv prior NIfTIs for test set...")
                    generate_deconv_niftis(
                        model,
                        test_loader,
                        test_dataset.patient_ids,
                        device=device,
                        mri_npy_dir=mri_npy_dir,
                        output_dir=os.path.join(output_dir, "prior_niftis", "test"),
                        mask_to_brain=True,
                        save_raw=deconv_save_raw_prior_nifti,
                    )
            else:
                print("\nGenerating prior NIfTIs for training set...")
                generate_prior_niftis(
                    train_preds,
                    mri_npy_dir=mri_npy_dir,
                    output_dir=os.path.join(output_dir, "prior_niftis", "train"),
                    gaussian_output_space=model_kwargs.get("gaussian_output_space", "normalized"),
                )
                print("Generating prior NIfTIs for validation set...")
                generate_prior_niftis(
                    val_preds,
                    mri_npy_dir=mri_npy_dir,
                    output_dir=os.path.join(output_dir, "prior_niftis", "val"),
                    gaussian_output_space=model_kwargs.get("gaussian_output_space", "normalized"),
                )
                if test_preds:
                    print("Generating prior NIfTIs for test set...")
                    generate_prior_niftis(
                        test_preds,
                        mri_npy_dir=mri_npy_dir,
                        output_dir=os.path.join(output_dir, "prior_niftis", "test"),
                        gaussian_output_space=model_kwargs.get("gaussian_output_space", "normalized"),
                    )

    return train_preds, val_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a multi-head MIL EEG spike model (coords + hemi + lobe)."
    )

    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to JSON file containing k-fold subject splits.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_spikes.npy files.")
    parser.add_argument("--targets_json", type=str, required=True,
                        help="JSON mapping patient_id -> coordinate targets.")
    parser.add_argument("--lobe_json", type=str, default=None,
                        help="JSON mapping patient_id -> lobe label. Omit to disable lobe head.")
    parser.add_argument("--hemi_json", type=str, default=None,
                        help="JSON mapping patient_id -> hemisphere label. Omit to disable hemi head.")

    parser.add_argument("--log_root", type=str, default="./runs",
                        help="Base directory for TensorBoard logs.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional checkpoint path for manual inference. Required when --skip_training is set.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold from the JSON splits to use.")
    parser.add_argument("--in_channels", type=int, default=21,
                        help="Number of EEG channels in input.")
    parser.add_argument("--max_spikes", type=int, default=32,
                        help="Max spikes sampled per patient per epoch.")
    parser.add_argument("--min_spikes_per_patient", type=int, default=32,
                        help="Minimum spikes required per patient.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Patients per batch.")
    parser.add_argument("--emb_dim", type=int, default=None,
                        help="Spike embedding dimension.")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Hidden size for shared trunk and heads.")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout probability.")
    parser.add_argument("--encoder_type", type=str, default="t_s_cnn",
                        choices=["t_s_cnn"],
                        help="Spike encoder backend (default: t_s_cnn — temporal 1D CNN with GNN spatial mixing).")
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["attention", "mean", "mean-std", "mean-max-topk"],
                        help="MIL pooling mode.")
    parser.add_argument("--lambda_coord", type=float, default=1.0,
                        help="Weight for coordinate regression loss. Set to 0.0 to disable coord head.")
    parser.add_argument("--lambda_hemi", type=float, default=0.15,
                        help="Weight for hemisphere classification loss. Set to 0.0 to disable hemi head.")
    parser.add_argument("--lambda_lobe", type=float, default=0.15,
                        help="Weight for lobe classification loss. Set to 0.0 to disable lobe head.")
    parser.add_argument("--spatial_head", type=str, default="deconv",
                        choices=["none", "coordinate", "gaussian_mixture", "deconv"],
                        help="Spatial prediction head (default: deconv — 3D deconvolutional prior map; the primary output head).")
    parser.add_argument("--num_gaussians", type=int, default=3,
                        help="Number of Gaussian components for gaussian_mixture head.")
    parser.add_argument("--gaussian_coord_dim", type=int, default=3,
                        help="Coordinate dimension for Gaussian centers (typically 3).")
    parser.add_argument("--gaussian_sigma_min", type=float, default=None,
                        help="Minimum sigma for Gaussian components. If omitted, defaults are unit-aware: "
                            "0.02 (normalized) or 2.0 (mni_mm).")
    parser.add_argument("--gaussian_sigma_max", type=float, default=None,
                        help="Maximum sigma for Gaussian components. If omitted, defaults are unit-aware: "
                            "0.25 (normalized) or 100.0 (mni_mm).")
    parser.add_argument("--gaussian_isotropic", action="store_true", default=True,
                        help="Use isotropic Gaussian components (sigma shape [K, 1]).")
    parser.add_argument("--gaussian_anisotropic", dest="gaussian_isotropic", action="store_false",
                        help="Use anisotropic Gaussian components (sigma shape [K, 3]).")
    parser.add_argument("--gaussian_output_space", type=str, default="normalized",
                        choices=["normalized", "mni_mm"],
                        help="Output space for Gaussian centers and sigmas.")
    parser.add_argument("--gaussian_make_heatmap", action="store_true",
                        help="If set, indicates Gaussian heatmap usage is desired (validation enforced).")
    parser.add_argument("--gaussian_heatmap_shape", type=int, nargs=3, default=None,
                        help="Optional heatmap shape (D H W) when using heatmap-based supervision.")
    parser.add_argument("--gaussian_loss_weight", type=float, default=1.0,
                        help="Weight for Gaussian-mixture spatial loss.")
    parser.add_argument("--gaussian_target", type=str, default="centroid",
                        choices=["mask", "centroid", "both"],
                        help="Gaussian supervision target type.")
    parser.add_argument("--gaussian_target_blur_sigma", type=float, default=5.0,
                        help="Target blur sigma for future mask/heatmap supervision.")
    parser.add_argument("--deconv_output_shape", type=int, nargs=3, default=[32, 40, 32],
                        help="Low-resolution 3D output shape for the deconv spatial head.")
    parser.add_argument("--deconv_latent_shape", type=int, nargs=3, default=[4, 5, 4],
                        help="Initial latent 3D grid shape before upsampling.")
    parser.add_argument("--deconv_base_channels", type=int, default=128,
                        help="Base number of channels in the deconv decoder.")
    parser.add_argument("--deconv_dropout", type=float, default=0.0,
                        help="Dropout used in deconv upsampling blocks.")
    parser.add_argument("--deconv_loss_weight", type=float, default=1.0,
                        help="Weight of the deconv spatial map loss.")
    parser.add_argument("--deconv_target_blur_sigma", type=float, default=3.0,
                        help="Gaussian blur sigma applied to binary GT masks at decoder resolution.")
    parser.add_argument("--deconv_entropy_weight", type=float, default=0.0,
                        help="Optional entropy reward to discourage collapsed/overly peaky deconv maps.")
    parser.add_argument("--deconv_use_brain_mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Use an MNI brain mask to constrain prediction and loss.")
    parser.add_argument("--deconv_brain_mask_path", type=str, default=None,
                        help="Path to MNI brain mask aligned to the deconv output grid or resampleable to it.")
    parser.add_argument("--deconv_loss", type=str, default="soft_bce_coverage",
                        choices=["dice", "bce", "dice_bce", "soft_bce", "soft_bce_coverage"],
                        help="Voxel-wise loss for the deconv spatial head.")
    parser.add_argument("--deconv_bce_weight", type=float, default=1.0,
                        help="Weight for the soft-BCE term used by --deconv_loss soft_bce_coverage.")
    parser.add_argument("--deconv_coverage_weight", type=float, default=0.1,
                        help="Weight for the coverage penalty used by --deconv_loss soft_bce_coverage.")
    parser.add_argument("--deconv_mass_weight", type=float, default=0.01,
                        help="Weight for the effective-volume mass penalty used by --deconv_loss soft_bce_coverage.")
    parser.add_argument("--deconv_mask_outside_brain", action=argparse.BooleanOptionalAction, default=False,
                        help="Force predictions outside the MNI brain mask to zero in optimization/metrics.")
    parser.add_argument("--deconv_tv_weight", type=float, default=0.0,
                        help="Optional total-variation smoothness penalty on predicted spatial prior.")
    parser.add_argument("--deconv_pos_weight", type=float, default=None,
                        help="Optional positive class weight for BCE/focal loss.")
    parser.add_argument("--deconv_outside_brain_penalty_weight", type=float, default=0.0,
                        help="Small penalty weight for predicted probability mass outside the brain mask.")
    parser.add_argument("--deconv_val_image_log_every", type=int, default=20,
                        help="Log deconv validation prediction/target slice image every N epochs.")
    parser.add_argument("--sigma_reg_lambda", type=float, default=5e-4,
                        help="Regularization weight on coord-head log(sigma).")
    parser.add_argument("--gaussian_sigma_reg_lambda", type=float, default=5e-4,
                        help="Regularization weight on log(sigma) for Gaussian-mixture head.")
    parser.add_argument("--harmonize_lobe_locations", action=argparse.BooleanOptionalAction, default=True,
                        help="Use weighted lobe labels from gt_lobe.json to harmonize train sampling.")
    parser.add_argument("--lobe_harmonization_factor", type=float, default=0.5,
                        help="Strength of lobe harmonization. 0 disables reweighting; 1 uses full rarity weighting.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-3,
                        help="Weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--test_mode", action="store_true",
                        help="If set, limit data for quick testing.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run only a smoke test (forward/backward on one batch).")
    parser.add_argument("--skip_inference", action="store_true",
                        help="If set, skips inference after training.")
    parser.add_argument("--skip_training", action="store_true",
                        help="If set, skip training and run inference from --checkpoint_path.")
    parser.add_argument("--infer_test_set", action="store_true",
                        help="If set, run inference on test set in addition to train/val (default: False).")
    parser.add_argument("--infer_all_subjects", action="store_true",
                        help="If set, run inference across train/val/test memberships for the selected fold.")
    parser.add_argument("--store_feature_vectors", action="store_true",
                        help="If set, export per-spike and per-patient feature vectors during inference.")
    parser.add_argument("--feature_output_dir", type=str, default=None,
                        help="Optional output directory for exported feature vectors. Defaults near the checkpoint.")
    parser.add_argument("--feature_bag_multiplier", type=int, default=4,
                        help="Feature export cap multiplier. Max export bag size = training_bag_size * multiplier.")
    parser.add_argument("--overwrite_feature_vectors", action="store_true",
                        help="If set, allows overwriting existing feature export artifacts.")
    parser.add_argument("--generate_niftis", action="store_true",
                        help="If set, generate Gaussian prior NIfTI files from inference predictions.")
    parser.add_argument("--mri_npy_dir", type=str, default=None,
                        help="Directory containing {patient_id}_preproc.npz MRI files (with 'image' and 'affine'). "
                             "Required when --generate_niftis is set.")
    parser.add_argument("--deconv_save_raw_prior_nifti", action=argparse.BooleanOptionalAction, default=False,
                        help="When exporting deconv NIfTIs, also save an unmasked raw prior as "
                             "<patient_id>_deconv_prior_raw.nii.gz.")
    parser.add_argument("--lesion_metric_every", type=int, default=50,
                        help="Compute lesion in-vs-out prior metric every N epochs (default: 50). "
                            "Set <= 0 to disable.")
    parser.add_argument("--lesion_mask_threshold", type=float, default=0.5,
                        help="Threshold for converting GT lesion mask to binary when computing lesion metrics.")

    # Early stopping arguments
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable early stopping (default: True). Use --no-early_stopping to disable.")
    parser.add_argument("--early_stopping_patience", type=int, default=200,
                        help="Epochs without improvement before stopping (default: 200).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum absolute improvement in smoothed val loss (default: 0.0).")
    parser.add_argument("--early_stopping_warmup", type=int, default=150,
                        help="Epochs before early stopping may trigger (default: 150).")
    parser.add_argument("--early_stopping_smoothing_window", type=int, default=10,
                        help="Rolling window size for smoothed val loss (default: 10).")
    parser.add_argument("--restore_best_checkpoint", action=argparse.BooleanOptionalAction, default=True,
                        help="Restore best smoothed-loss checkpoint before inference (default: True).")

    # Pretrained encoder arguments
    parser.add_argument("--pretrained_encoder_path", type=str, default=None,
                        help="Path to encoder_best.pt from spike-encoder pretraining. "
                             "If set, the encoder weights are loaded before training.")
    parser.add_argument("--freeze_encoder", type=str, default="none",
                        choices=["none", "full", "temporal"],
                        help="Freeze pretrained encoder weights: 'none' (no freeze), "
                             "'full' (entire encoder), or 'temporal' (temporal sub-modules only).")

    args = parser.parse_args()

    if args.skip_inference and args.store_feature_vectors:
        raise ValueError("--store_feature_vectors requires an active inference run; remove --skip_inference.")

    train_results = None
    checkpoint_for_inference = None
    inference_output_dir = None

    if args.skip_training:
        if args.checkpoint_path is None:
            raise ValueError("--skip_training requires --checkpoint_path.")
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        checkpoint_for_inference = args.checkpoint_path
        inference_output_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
        print("\nTraining skipped (--skip_training). Using checkpoint for inference:")
        print(f"  {checkpoint_for_inference}")
    else:
        train_results = train(
            json_split_path=args.splits_json, fold_index=args.fold, data_dir=args.data_dir,
            json_targets_path=args.targets_json, lobe_json_path=args.lobe_json, hemi_json_path=args.hemi_json,
            in_channels=args.in_channels, max_spikes=args.max_spikes,
            min_spikes_per_patient=args.min_spikes_per_patient, batch_size=args.batch_size,
            emb_dim=args.emb_dim, hidden=args.hidden, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, log_root=args.log_root,
            num_workers=args.num_workers, test_mode=args.test_mode,
            encoder_type=args.encoder_type, pooling=args.pooling,
            lambda_coord=args.lambda_coord, lambda_hemi=args.lambda_hemi, lambda_lobe=args.lambda_lobe,
            spatial_head=args.spatial_head, num_gaussians=args.num_gaussians,
            gaussian_coord_dim=args.gaussian_coord_dim,
            gaussian_sigma_min=args.gaussian_sigma_min, gaussian_sigma_max=args.gaussian_sigma_max,
            gaussian_isotropic=args.gaussian_isotropic, gaussian_output_space=args.gaussian_output_space,
            gaussian_make_heatmap=args.gaussian_make_heatmap,
            gaussian_heatmap_shape=tuple(args.gaussian_heatmap_shape) if args.gaussian_heatmap_shape is not None else None,
            gaussian_loss_weight=args.gaussian_loss_weight, gaussian_target=args.gaussian_target,
            gaussian_target_blur_sigma=args.gaussian_target_blur_sigma,
            deconv_output_shape=tuple(args.deconv_output_shape), deconv_latent_shape=tuple(args.deconv_latent_shape),
            deconv_base_channels=args.deconv_base_channels, deconv_dropout=args.deconv_dropout,
            deconv_loss_weight=args.deconv_loss_weight, deconv_target_blur_sigma=args.deconv_target_blur_sigma,
            deconv_bce_weight=args.deconv_bce_weight,
            deconv_coverage_weight=args.deconv_coverage_weight,
            deconv_mass_weight=args.deconv_mass_weight,
            deconv_entropy_weight=args.deconv_entropy_weight,
            deconv_use_brain_mask=args.deconv_use_brain_mask, deconv_brain_mask_path=args.deconv_brain_mask_path,
            deconv_loss=args.deconv_loss, deconv_mask_outside_brain=args.deconv_mask_outside_brain,
            deconv_tv_weight=args.deconv_tv_weight, deconv_pos_weight=args.deconv_pos_weight,
            deconv_outside_brain_penalty_weight=args.deconv_outside_brain_penalty_weight,
            deconv_val_image_log_every=args.deconv_val_image_log_every,
            sigma_reg_lambda=args.sigma_reg_lambda, gaussian_sigma_reg_lambda=args.gaussian_sigma_reg_lambda,
            harmonize_lobe_locations=args.harmonize_lobe_locations,
            lobe_harmonization_factor=args.lobe_harmonization_factor,
            lesion_metric_every=args.lesion_metric_every, mri_npy_dir=args.mri_npy_dir,
            lesion_mask_threshold=args.lesion_mask_threshold, dry_run=args.dry_run,
            early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta, early_stopping_warmup=args.early_stopping_warmup,
            early_stopping_smoothing_window=args.early_stopping_smoothing_window,
            restore_best_checkpoint=args.restore_best_checkpoint,
            pretrained_encoder_path=args.pretrained_encoder_path, freeze_encoder=args.freeze_encoder,
        )

        if train_results is not None:
            checkpoint_for_inference = train_results["best_checkpoint"]
            inference_output_dir = train_results["log_dir"]

        if args.checkpoint_path is not None:
            if not os.path.exists(args.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
            checkpoint_for_inference = args.checkpoint_path
            if inference_output_dir is None:
                inference_output_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
            print("\nUsing explicit --checkpoint_path for inference override:")
            print(f"  {checkpoint_for_inference}")

    if args.skip_inference:
        print("\nInference skipped (--skip_inference flag set).")
    else:
        if checkpoint_for_inference is None or not os.path.exists(checkpoint_for_inference):
            raise FileNotFoundError(
                "No valid checkpoint available for inference. "
                "Provide --checkpoint_path with --skip_training or run training first."
            )

        print("\n" + "=" * 80)
        print("RUNNING INFERENCE WITH CHECKPOINT")
        print("=" * 80)
        run_inference(
            checkpoint_path=checkpoint_for_inference,
            json_split_path=args.splits_json, fold_index=args.fold, data_dir=args.data_dir,
            json_targets_path=args.targets_json, lobe_json_path=args.lobe_json, hemi_json_path=args.hemi_json,
            output_dir=inference_output_dir,
            in_channels=args.in_channels, max_spikes=args.max_spikes * 4,
            min_spikes_per_patient=args.min_spikes_per_patient, batch_size=args.batch_size,
            emb_dim=args.emb_dim, hidden=args.hidden, dropout=args.dropout, num_workers=args.num_workers,
            encoder_type=args.encoder_type, pooling=args.pooling,
            spatial_head=args.spatial_head, num_gaussians=args.num_gaussians,
            gaussian_coord_dim=args.gaussian_coord_dim,
            gaussian_sigma_min=args.gaussian_sigma_min, gaussian_sigma_max=args.gaussian_sigma_max,
            gaussian_isotropic=args.gaussian_isotropic, gaussian_output_space=args.gaussian_output_space,
            gaussian_make_heatmap=args.gaussian_make_heatmap,
            gaussian_heatmap_shape=tuple(args.gaussian_heatmap_shape) if args.gaussian_heatmap_shape is not None else None,
            deconv_output_shape=tuple(args.deconv_output_shape), deconv_latent_shape=tuple(args.deconv_latent_shape),
            deconv_base_channels=args.deconv_base_channels, deconv_dropout=args.deconv_dropout,
            test_mode=args.test_mode, infer_test_set=args.infer_test_set,
            infer_all_subjects=args.infer_all_subjects,
            store_feature_vectors=args.store_feature_vectors,
            feature_output_dir=args.feature_output_dir,
            feature_bag_multiplier=args.feature_bag_multiplier,
            overwrite_feature_vectors=args.overwrite_feature_vectors,
            generate_niftis=args.generate_niftis, mri_npy_dir=args.mri_npy_dir,
            deconv_save_raw_prior_nifti=args.deconv_save_raw_prior_nifti,
        )
