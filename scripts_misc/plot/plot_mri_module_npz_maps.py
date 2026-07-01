"""
Visualize MRI module prediction maps against MRI preprocessing outputs.

For each subject ID with a prediction map file in ``--pred_dir``, this script loads:
- prediction map: ``{subject_id}.npz`` or ``{subject_id}.nii.gz``
- normalized T1 image: ``{subject_id}_T1w_norm.nii.gz``
- normalized GT mask: ``{subject_id}_gt_norm.nii.gz``

It creates a tri-planar (axial/coronal/sagittal) figure centered on the GT mask
center of mass, overlays prediction map on top of T1, then overlays GT contour,
and saves:
1) one PNG per subject
2) one combined grid PNG containing all per-subject plots

When ``--enable_no_gt_multislice`` is used, subjects with missing/empty GT masks
are rendered as a 3x12 multi-slice panel (transversal/sagittal/coronal) with one
shared colormap legend.

Notes
-----
- Map source is selected with ``--map_source`` (``npz`` or ``nii``).
- NPZ map key can be provided with ``--npz_key``. If omitted, the script tries to
  auto-select a sensible array.
- Use ``--test_set`` when ``--pred_dir`` contains ``fold_<idx>`` subdirectories
    with test-set predictions from each fold model. In this mode, all fold predictions
    are plotted, and optional combined maps are loaded from ``--combined_pred_dir``.

Usage
-----
python scripts_other/plot/plot_mri_module_npz_maps.py \
    --pred_dir /path/to/mri_module_outputs \
    --mri_dir /path/to/preprocessed_mri \
    --map_source npz \
    --output_dir /path/to/output
"""

import argparse
import itertools
import json
import math
import os
import re
from collections import deque
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from tqdm import tqdm


T1_SUFFIX = "_T1w_norm.nii.gz"
GT_SUFFIX = "_gt_norm.nii.gz"
FLAIR_SUFFIXES = ("_FLAIR_norm.nii.gz", "_flair_norm.nii.gz")
NII_SUFFIX = [".nii", ".nii.gz"]
MRI_DIR_DEFAULT = r"l:\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\mri"
K_FOLD_SPLITS_PATH = r"l:\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\k_fold_splits.json"
SUBJECT_ID_REGEX = re.compile(r"(RESP\d+)", re.IGNORECASE)


def _extract_subject_id(name: str) -> Optional[str]:
    """Extract subject ID using RESP####... pattern from a filename or stem."""
    match = SUBJECT_ID_REGEX.search(str(name))
    if match is None:
        return None
    return match.group(1).upper()


def _safe_percentile(values: np.ndarray, q: float, fallback: float) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return fallback
    return float(np.percentile(vals, q))


def _visual_style(white_background: bool) -> Tuple[str, str, mcolors.Colormap]:
    bg = "white" if white_background else "black"
    fg = "black" if white_background else "white"
    gray_cmap = plt.cm.get_cmap("gray").copy()
    gray_cmap.set_bad(color=bg)
    return bg, fg, gray_cmap


def _mask_zero_background(slice2d: np.ndarray, white_background: bool) -> np.ndarray:
    arr = np.asarray(slice2d, dtype=np.float32)
    if not white_background:
        return arr

    def _largest_connected_component_2d(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_coords: List[Tuple[int, int]] = []
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        ys, xs = np.where(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(coords) > len(best_coords):
                best_coords = coords

        out = np.zeros_like(mask, dtype=bool)
        if best_coords:
            yy, xx = zip(*best_coords)
            out[np.asarray(yy, dtype=int), np.asarray(xx, dtype=int)] = True
        return out

    def _dilate_mask_2d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = np.asarray(mask, dtype=bool)
        for _ in range(max(0, int(iterations))):
            padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
            out = (
                padded[0:-2, 0:-2] | padded[0:-2, 1:-1] | padded[0:-2, 2:] |
                padded[1:-1, 0:-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
                padded[2:, 0:-2] | padded[2:, 1:-1] | padded[2:, 2:]
            )
        return out

    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr

    vals = arr[finite]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        out = arr.copy()
        out[finite] = max(vmax * 1.5, 1.0)
        return out

    value_range = vmax - vmin
    bins = int(np.clip(np.sqrt(vals.size), 64, 512))
    hist, edges = np.histogram(vals, bins=bins, range=(vmin, vmax))
    if hist.size == 0:
        return arr

    # Background is expected in lower intensities and to be relatively abundant.
    low_band_max = vmin + 0.35 * value_range
    low_candidates = np.where(edges[:-1] <= low_band_max)[0]
    if low_candidates.size > 0:
        peak_idx = int(low_candidates[np.argmax(hist[low_candidates])])
    else:
        peak_idx = int(np.argmax(hist))

    peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    bin_width = max(float(edges[1] - edges[0]), 1e-6)
    band_half = max(1.5 * bin_width, 0.01 * value_range)

    band_mask = finite & (arr >= (peak_center - band_half)) & (arr <= (peak_center + band_half))
    if int(np.count_nonzero(band_mask)) < max(16, int(0.002 * vals.size)):
        # Fallback: near-min values are most likely background.
        band_mask = finite & (arr <= (vmin + max(3.0 * bin_width, 0.02 * value_range)))

    largest_bg = _largest_connected_component_2d(band_mask)
    if np.any(largest_bg):
        dilated_bg = _dilate_mask_2d(largest_bg, iterations=2)
        wider_half = max(15.0 * bin_width, 0.12 * value_range)
        refined_bg = dilated_bg & finite & (arr >= (peak_center - wider_half)) & (arr <= (peak_center + wider_half))
        if np.any(refined_bg):
            band_mask = refined_bg
        else:
            band_mask = largest_bg

    out = arr.copy()
    out[band_mask] = float(vmax * 1.5)
    return out


def _gt_center(gt: np.ndarray) -> Optional[Tuple[int, int, int]]:
    coords = np.argwhere(gt > 0)
    if coords.size == 0:
        return None
    center = np.round(coords.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def _clamp_idx(idx: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(idx, size - 1))


def _slice_triplet(volume: np.ndarray, center: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    x, y, z = center
    x = _clamp_idx(x, volume.shape[0])
    y = _clamp_idx(y, volume.shape[1])
    z = _clamp_idx(z, volume.shape[2])

    return {
        "Axial": np.flipud(np.rot90(volume[:, :, z])),
        "Coronal": np.flipud(np.rot90(volume[:, y, :])),
        "Sagittal": np.flipud(np.rot90(volume[x, :, :])),
    }


def _slice_indices(size: int, n_slices: int) -> List[int]:
    if size <= 0:
        return [0 for _ in range(max(1, n_slices))]
    n = max(1, int(n_slices))
    if size == 1:
        return [0 for _ in range(n)]
    vals = np.linspace(0, size - 1, num=n)
    return [_clamp_idx(int(round(v)), size) for v in vals]


def _slice_indices_around_center(size: int, n_slices: int, center: int, window_fraction: float = 0.3) -> List[int]:
    if size <= 0:
        return [0 for _ in range(max(1, n_slices))]
    n = max(1, int(n_slices))
    if size == 1:
        return [0 for _ in range(n)]

    center = _clamp_idx(int(center), size)
    half_width = max(1, int(round(size * float(window_fraction) / 2.0)))
    start = max(0, center - half_width)
    end = min(size - 1, center + half_width)
    if end <= start:
        start, end = 0, size - 1

    vals = np.linspace(start, end, num=n)
    return [_clamp_idx(int(round(v)), size) for v in vals]


def _crop_volume_to_bbox(
    volume: np.ndarray,
    bbox_min: Tuple[int, int, int],
    bbox_max: Tuple[int, int, int],
    pad: int,
) -> Tuple[np.ndarray, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    mins = np.asarray(bbox_min, dtype=int)
    maxs = np.asarray(bbox_max, dtype=int)
    pad = max(0, int(pad))
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.asarray(volume.shape, dtype=int))
    cropped = volume[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]
    return cropped, (tuple(int(v) for v in mins), tuple(int(v) for v in maxs))


def _find_hotspot_clusters(map3d: np.ndarray, threshold: float = 0.3) -> List[Dict[str, object]]:
    mask = np.asarray(map3d, dtype=np.float32) > float(threshold)
    if not np.any(mask):
        return []

    visited = np.zeros(mask.shape, dtype=bool)
    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    clusters: List[Dict[str, object]] = []
    shape = mask.shape

    for seed in np.argwhere(mask):
        sx, sy, sz = (int(seed[0]), int(seed[1]), int(seed[2]))
        if visited[sx, sy, sz]:
            continue

        queue = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = True
        coords: List[Tuple[int, int, int]] = []
        values: List[float] = []

        while queue:
            x, y, z = queue.popleft()
            coords.append((x, y, z))
            values.append(float(map3d[x, y, z]))

            for dx, dy, dz in neighbors:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < 0 or ny < 0 or nz < 0:
                    continue
                if nx >= shape[0] or ny >= shape[1] or nz >= shape[2]:
                    continue
                if visited[nx, ny, nz] or not mask[nx, ny, nz]:
                    continue
                visited[nx, ny, nz] = True
                queue.append((nx, ny, nz))

        if not coords:
            continue

        coords_arr = np.asarray(coords, dtype=np.int32)
        bbox_min = tuple(int(v) for v in coords_arr.min(axis=0))
        bbox_max = tuple(int(v) + 1 for v in coords_arr.max(axis=0))
        clusters.append(
            {
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "voxel_count": int(coords_arr.shape[0]),
                "max_value": float(np.max(values)),
                "mean_value": float(np.mean(values)),
            }
        )

    clusters.sort(key=lambda item: (int(item["voxel_count"]), float(item["max_value"])), reverse=True)
    return clusters


def _index_subject_npz(pred_dir: str, recursive: bool = False) -> Dict[str, str]:
    pattern = os.path.join(pred_dir, "**", "*.npz") if recursive else os.path.join(pred_dir, "*.npz")
    matches = sorted(glob(pattern, recursive=recursive))
    mapping: Dict[str, str] = {}
    for path in matches:
        sid = _extract_subject_id(os.path.basename(path))
        if sid and sid not in mapping:
            mapping[sid] = path
    return mapping


def _index_subject_nii(pred_dir: str, recursive: bool = False) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for suffix in NII_SUFFIX:
        pattern = os.path.join(pred_dir, "**", f"*{suffix}") if recursive else os.path.join(pred_dir, f"*{suffix}")
        matches = sorted(glob(pattern, recursive=recursive))
        for path in matches:
            name = os.path.basename(path)
            if not name.endswith(suffix):
                continue
            sid = _extract_subject_id(name)
            # Avoid accidentally indexing MRI reference files if present in pred_dir.
            if sid is None:
                continue
            if sid and sid not in mapping:
                mapping[sid] = path
    return mapping


def _find_fold_dirs(pred_dir: str) -> Dict[str, str]:
    fold_dirs: Dict[str, str] = {}
    try:
        children = sorted(os.listdir(pred_dir))
    except OSError:
        return fold_dirs

    for name in children:
        path = os.path.join(pred_dir, name)
        if not os.path.isdir(path):
            continue
        if re.fullmatch(r"fold_\d+", name):
            fold_dirs[name] = path
    return fold_dirs


def _load_kfold_val_ids(json_path: str) -> Dict[str, set]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fold_payload = payload["folds"]

    val_ids_by_fold: Dict[str, set] = {}
    for fold_name, fold_data in fold_payload.items():
        if not isinstance(fold_data, dict):
            continue
        val_ids = fold_data.get("val_ids", [])
        if not isinstance(val_ids, list):
            continue
        val_ids_by_fold[fold_name] = {str(v) for v in val_ids}
    return val_ids_by_fold


def _map_npz_to_mri_space(
    map3d: np.ndarray,
    map_affine: np.ndarray,
    target_nii: nib.Nifti1Image,
) -> np.ndarray:
    src_nii = nib.Nifti1Image(np.asarray(map3d, dtype=np.float32), np.asarray(map_affine, dtype=np.float32))
    # Use linear interpolation to preserve smooth probability-like maps.
    resampled = resample_from_to(src_nii, target_nii, order=1)
    return np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)


def _resample_mask_to_target(mask_nii: nib.Nifti1Image, target_nii: nib.Nifti1Image) -> np.ndarray:
    # Nearest-neighbor preserves binary mask semantics.
    resampled = resample_from_to(mask_nii, target_nii, order=0)
    mask = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
    return mask > 0


def _register_map_to_mask_translation(
    map3d: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
) -> Tuple[np.ndarray, Tuple[float, float, float], float]:
    try:
        import importlib

        sitk = importlib.import_module("SimpleITK")
    except Exception as exc:
        raise ImportError(
            "SimpleITK is required for registration-based NPZ alignment. "
            "Install it with: pip install SimpleITK"
        ) from exc

    mask_bool = mask > 0
    if not np.any(mask_bool):
        return map3d, (0.0, 0.0, 0.0), float("nan")

    moving_np = _to_probability_like(map3d).astype(np.float32)
    fixed_np = mask_bool.astype(np.float32)

    fixed_img = sitk.GetImageFromArray(fixed_np)
    moving_img = sitk.GetImageFromArray(moving_np)
    fixed_img.SetSpacing((1.0, 1.0, 1.0))
    moving_img.SetSpacing((1.0, 1.0, 1.0))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=250,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(sitk.TranslationTransform(3), inPlace=False)

    final_tx = reg.Execute(fixed_img, moving_img)
    shift = np.asarray(final_tx.GetParameters(), dtype=np.float32)
    max_shift = max(0, int(max_shift))
    shift_clamped = np.clip(shift, -max_shift, max_shift)

    tx_apply = sitk.TranslationTransform(3)
    tx_apply.SetParameters(tuple(float(v) for v in shift_clamped.tolist()))

    aligned_img = sitk.Resample(
        moving_img,
        fixed_img,
        tx_apply,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    aligned = sitk.GetArrayFromImage(aligned_img).astype(np.float32)
    score = float(np.mean(aligned[mask_bool])) if np.any(mask_bool) else float("nan")
    return aligned, (float(shift_clamped[0]), float(shift_clamped[1]), float(shift_clamped[2])), score


def _index_files_recursive(root_dir: str, suffix: str) -> Dict[str, str]:
    pattern = os.path.join(root_dir, "**", f"*{suffix}")
    matches = sorted(glob(pattern, recursive=True))
    mapping: Dict[str, str] = {}
    for path in matches:
        name = os.path.basename(path)
        if not name.endswith(suffix):
            continue
        sid = _extract_subject_id(name)
        if sid and sid not in mapping:
            mapping[sid] = path
    return mapping


def _index_files_recursive_multi_suffix(root_dir: str, suffixes: Tuple[str, ...]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for suffix in suffixes:
        current = _index_files_recursive(root_dir, suffix)
        for sid, path in current.items():
            if sid not in mapping:
                mapping[sid] = path
    return mapping


def _score_key(name: str) -> int:
    key = name.lower()
    priority = [
        "prob",
        "probs",
        "foreground",
        "foreground_prob",
        "prediction",
        "pred",
        "map",
        "heatmap",
        "logit",
        "logits",
        "arr_0",
    ]
    for i, token in enumerate(priority):
        if token in key:
            return i
    return 10_000


def _to_probability_like(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[finite]
    vmin = float(vals.min())
    vmax = float(vals.max())

    # Already probability-like.
    if vmin >= 0.0 and vmax <= 1.0:
        out = arr.copy()
        out[~finite] = 0.0
        return out

    # Looks like logits -> sigmoid.
    if vmin >= -20.0 and vmax <= 20.0:
        out = 1.0 / (1.0 + np.exp(-arr))
        out[~finite] = 0.0
        return out.astype(np.float32)

    # Fallback: robust min-max normalization.
    lo = _safe_percentile(vals, 1.0, fallback=vmin)
    hi = _safe_percentile(vals, 99.0, fallback=vmax)
    if hi <= lo:
        hi = lo + 1e-6
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _overlay_alpha(map_slice: np.ndarray, alpha_constant: float, high_end: float = 0.3) -> np.ndarray:
    """Scale overlay alpha by map strength with saturation above ``high_end``."""
    high_end = max(float(high_end), 1e-6)
    alpha_map = np.clip(np.asarray(map_slice, dtype=np.float32) / high_end, 0.0, 1.0)
    alpha_constant = float(np.clip(alpha_constant, 0.0, 1.0))
    return alpha_constant * alpha_map


def _plot_raw_scan_triplet(
    subject_id: str,
    modality_name: str,
    scan: np.ndarray,
    gt: np.ndarray,
    out_path: str,
    white_background: bool = False,
) -> bool:
    center = _gt_center(gt)
    if center is None:
        return False

    scan_slices = _slice_triplet(scan, center)
    gt_slices = _slice_triplet(gt, center)

    scan_nonzero = scan[scan > 0]
    scan_vmin = _safe_percentile(scan_nonzero, 1, fallback=0.0)
    scan_vmax = _safe_percentile(scan_nonzero, 99, fallback=1.0)
    if scan_vmax <= scan_vmin:
        scan_vmax = scan_vmin + 1.0

    bg_color, fg_color, gray_cmap = _visual_style(white_background)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), facecolor=bg_color)
    plane_order = ["Axial", "Coronal", "Sagittal"]

    for ax, plane in zip(axes, plane_order):
        ax.set_facecolor(bg_color)
        scan_sl = scan_slices[plane]
        gt_sl = gt_slices[plane]

        ax.imshow(
            _mask_zero_background(scan_sl, white_background),
            cmap=gray_cmap,
            vmin=scan_vmin,
            vmax=scan_vmax,
            origin="lower",
        )
        if np.any(gt_sl > 0):
            ax.contour(gt_sl, levels=[0.5], colors=["#ff4040"], linewidths=1.0)
        ax.axis("off")

    x, y, z = center
    fig.text(
        0.006,
        0.012,
        f"{subject_id} | {modality_name} | c=({x},{y},{z})",
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)
    return True


def _plot_raw_scan_multislice(
    subject_id: str,
    modality_name: str,
    scan: np.ndarray,
    gt: np.ndarray,
    out_path: str,
    n_slices: int,
    focus_center: Optional[Tuple[int, int, int]] = None,
    focus_window_fraction: float = 0.3,
    white_background: bool = False,
) -> bool:
    scan_nonzero = scan[scan > 0]
    scan_vmin = _safe_percentile(scan_nonzero, 1, fallback=0.0)
    scan_vmax = _safe_percentile(scan_nonzero, 99, fallback=1.0)
    if scan_vmax <= scan_vmin:
        scan_vmax = scan_vmin + 1.0

    bg_color, fg_color, gray_cmap = _visual_style(white_background)
    n_slices = max(1, int(n_slices))
    if focus_center is None:
        axial_idx = _slice_indices(scan.shape[2], n_slices)
        sagittal_idx = _slice_indices(scan.shape[0], n_slices)
        coronal_idx = _slice_indices(scan.shape[1], n_slices)
    else:
        fx, fy, fz = focus_center
        axial_idx = _slice_indices_around_center(scan.shape[2], n_slices, fz, window_fraction=focus_window_fraction)
        sagittal_idx = _slice_indices_around_center(scan.shape[0], n_slices, fx, window_fraction=focus_window_fraction)
        coronal_idx = _slice_indices_around_center(scan.shape[1], n_slices, fy, window_fraction=focus_window_fraction)

    fig, axes = plt.subplots(
        3,
        n_slices,
        figsize=(max(9.0, n_slices * 1.1), 3.6),
        facecolor=bg_color,
    )
    if axes.ndim != 2:
        axes = np.asarray(axes).reshape(3, n_slices)

    planes = [
        ("Transversal", axial_idx),
        ("Sagittal", sagittal_idx),
        ("Coronal", coronal_idx),
    ]

    for row, (plane_name, idx_list) in enumerate(planes):
        for col, sl_idx in enumerate(idx_list):
            ax = axes[row, col]
            ax.set_facecolor(bg_color)

            if plane_name == "Transversal":
                scan_sl = np.flipud(np.rot90(scan[:, :, sl_idx]))
                gt_sl = np.flipud(np.rot90(gt[:, :, sl_idx]))
            elif plane_name == "Sagittal":
                scan_sl = np.flipud(np.rot90(scan[sl_idx, :, :]))
                gt_sl = np.flipud(np.rot90(gt[sl_idx, :, :]))
            else:
                scan_sl = np.flipud(np.rot90(scan[:, sl_idx, :]))
                gt_sl = np.flipud(np.rot90(gt[:, sl_idx, :]))

            ax.imshow(
                _mask_zero_background(scan_sl, white_background),
                cmap=gray_cmap,
                vmin=scan_vmin,
                vmax=scan_vmax,
                origin="lower",
            )
            if np.any(gt_sl > 0):
                ax.contour(gt_sl, levels=[0.5], colors=["#ff4040"], linewidths=0.8)
            ax.axis("off")

            if col == 0:
                ax.text(
                    0.02,
                    0.98,
                    plane_name,
                    transform=ax.transAxes,
                    color=fg_color,
                    fontsize=7,
                    ha="left",
                    va="top",
                )

    fig.text(
        0.006,
        0.01,
        f"{subject_id} | {modality_name} | 3x{n_slices}",
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.01, hspace=0.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)
    return True


def _score_map_similarity(candidate: np.ndarray, reference: np.ndarray) -> float:
    cand = _to_probability_like(candidate)
    ref = _to_probability_like(reference)
    valid = np.isfinite(cand) & np.isfinite(ref)
    if not np.any(valid):
        return float("-inf")
    return float(np.mean(cand[valid] * ref[valid]))


def _reorient_npz_to_reference(
    map3d: np.ndarray,
    ref3d: np.ndarray,
) -> Tuple[np.ndarray, str, float]:
    src = np.asarray(map3d, dtype=np.float32)
    ref = np.asarray(ref3d, dtype=np.float32)

    best_map = src
    best_desc = "identity"
    best_score = _score_map_similarity(src, ref) if src.shape == ref.shape else float("-inf")

    # Search axis permutations and flips to match NPZ array memory layout to NIfTI layout.
    for perm in itertools.permutations((0, 1, 2)):
        permuted = np.transpose(src, axes=perm)
        if permuted.shape != ref.shape:
            continue

        for flip_mask in range(8):
            candidate = permuted
            flip_axes: List[int] = []
            for axis in range(3):
                if (flip_mask >> axis) & 1:
                    candidate = np.flip(candidate, axis=axis)
                    flip_axes.append(axis)

            score = _score_map_similarity(candidate, ref)
            if score > best_score:
                best_score = score
                best_map = np.asarray(candidate, dtype=np.float32)
                best_desc = f"perm={perm}, flips={tuple(flip_axes)}"

    return best_map, best_desc, best_score


def _select_3d_map(arr: np.ndarray, channel: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D map, got shape {arr.shape}")

    # Heuristic for (C, D, H, W) maps.
    if channel is None:
        ch = 1 if arr.shape[0] >= 2 else 0
    else:
        ch = int(np.clip(channel, 0, arr.shape[0] - 1))
    return np.asarray(arr[ch], dtype=np.float32)


def _load_pred_map_from_npz(
    npz_path: str,
    reference_shape: Optional[Tuple[int, int, int]],
    npz_key: Optional[str],
    channel: Optional[int],
) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as payload:
        keys = list(payload.keys())
        if not keys:
            raise ValueError("NPZ has no arrays")

        if npz_key is not None:
            if npz_key not in payload:
                raise KeyError(f"Requested npz_key '{npz_key}' not found. Available keys: {keys}")
            candidate = _select_3d_map(payload[npz_key], channel=channel)
            if reference_shape is not None and candidate.shape != reference_shape:
                raise ValueError(
                    f"Key '{npz_key}' has shape {candidate.shape}, expected {reference_shape}"
                )
            return _to_probability_like(candidate)

        # Auto-pick key by shape match first, then name priority.
        candidates: List[Tuple[int, str, np.ndarray]] = []
        for key in keys:
            try:
                arr3d = _select_3d_map(payload[key], channel=channel)
            except Exception:
                continue
            if reference_shape is not None and tuple(arr3d.shape) != reference_shape:
                continue
            candidates.append((_score_key(key), key, arr3d))

        if not candidates:
            raise ValueError(
                f"No 3D/4D array in {npz_path} matched expected shape {reference_shape}. "
                f"Available keys: {keys}"
            )

        candidates.sort(key=lambda t: (t[0], t[1]))
        return _to_probability_like(candidates[0][2])


def _plot_single_subject(
    subject_id: str,
    t1: np.ndarray,
    pred_map: np.ndarray,
    gt: np.ndarray,
    out_path: str,
    map_alpha: float,
    map_cmap: str,
    white_background: bool = False,
) -> bool:
    center = _gt_center(gt)
    if center is None:
        return False

    pred_map = np.where(t1 > _safe_percentile(t1, 10, fallback=0.0), pred_map, 0.0).astype(np.float32)
    pred_map = np.clip(pred_map, 0.0, 1.0)

    t1_slices = _slice_triplet(t1, center)
    map_slices = _slice_triplet(pred_map, center)
    gt_slices = _slice_triplet(gt, center)

    t1_nonzero = t1[t1 > 0]
    t1_vmin = _safe_percentile(t1_nonzero, 1, fallback=0.0)
    t1_vmax = _safe_percentile(t1_nonzero, 99, fallback=1.0)
    if t1_vmax <= t1_vmin:
        t1_vmax = t1_vmin + 1.0

    bg_color, fg_color, gray_cmap = _visual_style(white_background)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(9.45, 3.0),
        facecolor=bg_color,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.06]},
    )
    plane_order = ["Axial", "Coronal", "Sagittal"]
    slice_axes = axes[:3]
    cax = axes[3]

    for ax, plane in zip(slice_axes, plane_order):
        ax.set_facecolor(bg_color)
        t1_sl = t1_slices[plane]
        map_sl = map_slices[plane]
        gt_sl = gt_slices[plane]

        ax.imshow(
            _mask_zero_background(t1_sl, white_background),
            cmap=gray_cmap,
            vmin=t1_vmin,
            vmax=t1_vmax,
            origin="lower",
        )
        ax.imshow(
            map_sl,
            cmap=map_cmap,
            alpha=_overlay_alpha(map_sl, map_alpha),
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        if np.any(gt_sl > 0):
            ax.contour(gt_sl, levels=[0.5], colors=["#ff4040"], linewidths=1.0)

        ax.axis("off")

    cax.set_facecolor(bg_color)
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=map_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Map", color=fg_color, fontsize=7)
    cbar.ax.tick_params(colors=fg_color, labelsize=7, length=2)
    cbar.outline.set_edgecolor(fg_color)

    x, y, z = center
    fig.text(
        0.006,
        0.012,
        f"{subject_id} | c=({x},{y},{z})",
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)
    return True


def _plot_single_subject_no_gt_multislice(
    subject_id: str,
    t1: np.ndarray,
    pred_map: np.ndarray,
    out_path: str,
    map_alpha: float,
    map_cmap: str,
    n_slices: int = 12,
    footer_text: Optional[str] = None,
    focus_center: Optional[Tuple[int, int, int]] = None,
    focus_window_fraction: float = 0.3,
    cluster_bbox: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
    white_background: bool = False,
) -> bool:
    pred_map = np.where(t1 > _safe_percentile(t1, 10, fallback=0.0), pred_map, 0.0).astype(np.float32)
    pred_map = np.clip(pred_map, 0.0, 1.0)

    t1_nonzero = t1[t1 > 0]
    t1_vmin = _safe_percentile(t1_nonzero, 1, fallback=0.0)
    t1_vmax = _safe_percentile(t1_nonzero, 99, fallback=1.0)
    if t1_vmax <= t1_vmin:
        t1_vmax = t1_vmin + 1.0

    bg_color, fg_color, gray_cmap = _visual_style(white_background)
    n_slices = max(1, int(n_slices))
    if focus_center is None:
        axial_idx = _slice_indices(t1.shape[2], n_slices)
        sagittal_idx = _slice_indices(t1.shape[0], n_slices)
        coronal_idx = _slice_indices(t1.shape[1], n_slices)
    else:
        fx, fy, fz = focus_center
        axial_idx = _slice_indices_around_center(t1.shape[2], n_slices, fz, window_fraction=focus_window_fraction)
        sagittal_idx = _slice_indices_around_center(t1.shape[0], n_slices, fx, window_fraction=focus_window_fraction)
        coronal_idx = _slice_indices_around_center(t1.shape[1], n_slices, fy, window_fraction=focus_window_fraction)

    fig, axes = plt.subplots(
        3,
        n_slices + 1,
        figsize=(max(9.0, n_slices * 1.18), 3.9),
        facecolor=bg_color,
        gridspec_kw={"width_ratios": [1.0] * n_slices + [0.16]},
    )

    if axes.ndim != 2:
        axes = np.asarray(axes).reshape(3, n_slices + 1)

    planes = [
        ("Transversal", axial_idx),
        ("Sagittal", sagittal_idx),
        ("Coronal", coronal_idx),
    ]

    # Pre-build cluster bbox mask once to avoid repeated allocation inside the loop.
    cluster_mask: Optional[np.ndarray] = None
    if cluster_bbox is not None:
        bbox_min_c, bbox_max_c = cluster_bbox
        cluster_mask = np.zeros(t1.shape, dtype=np.float32)
        cluster_mask[
            bbox_min_c[0] : bbox_max_c[0],
            bbox_min_c[1] : bbox_max_c[1],
            bbox_min_c[2] : bbox_max_c[2],
        ] = 1.0

    for row, (plane_name, idx_list) in enumerate(planes):
        for col, sl_idx in enumerate(idx_list):
            ax = axes[row, col]
            ax.set_facecolor(bg_color)

            if plane_name == "Transversal":
                t1_sl = np.flipud(np.rot90(t1[:, :, sl_idx]))
                map_sl = np.flipud(np.rot90(pred_map[:, :, sl_idx]))
                cluster_sl = np.flipud(np.rot90(cluster_mask[:, :, sl_idx])) if cluster_mask is not None else None
            elif plane_name == "Sagittal":
                t1_sl = np.flipud(np.rot90(t1[sl_idx, :, :]))
                map_sl = np.flipud(np.rot90(pred_map[sl_idx, :, :]))
                cluster_sl = np.flipud(np.rot90(cluster_mask[sl_idx, :, :])) if cluster_mask is not None else None
            else:
                t1_sl = np.flipud(np.rot90(t1[:, sl_idx, :]))
                map_sl = np.flipud(np.rot90(pred_map[:, sl_idx, :]))
                cluster_sl = np.flipud(np.rot90(cluster_mask[:, sl_idx, :])) if cluster_mask is not None else None

            ax.imshow(
                _mask_zero_background(t1_sl, white_background),
                cmap=gray_cmap,
                vmin=t1_vmin,
                vmax=t1_vmax,
                origin="lower",
            )
            ax.imshow(
                map_sl,
                cmap=map_cmap,
                alpha=_overlay_alpha(map_sl, map_alpha),
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )
            if cluster_sl is not None and np.any(cluster_sl > 0):
                ax.contour(cluster_sl, levels=[0.5], colors=["#ff2020"], linewidths=0.5, alpha=0.35)
            ax.axis("off")

            if col == 0:
                ax.text(
                    0.02,
                    0.98,
                    plane_name,
                    transform=ax.transAxes,
                    color=fg_color,
                    fontsize=7,
                    ha="left",
                    va="top",
                )

        axes[row, -1].axis("off")

    fig.subplots_adjust(left=0.0, right=0.995, bottom=0.0, top=1.0, wspace=0.01, hspace=0.01)

    cax_col = axes[:, -1]
    top_bbox = cax_col[0].get_position()
    bottom_bbox = cax_col[-1].get_position()
    for ax_cb in cax_col:
        ax_cb.axis("off")

    cax0 = fig.add_axes(
        [
            top_bbox.x0,
            bottom_bbox.y0,
            top_bbox.width,
            top_bbox.y1 - bottom_bbox.y0,
        ]
    )
    cax0.set_facecolor(bg_color)
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=map_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax0)
    cbar.set_label("Map", color=fg_color, fontsize=7)
    cbar.ax.tick_params(colors=fg_color, labelsize=7, length=2)
    cbar.outline.set_edgecolor(fg_color)

    fig.text(
        0.006,
        0.01,
        f"{subject_id} | {footer_text or 'multislice'} | 3x{n_slices}",
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)
    return True


def _build_grid_image(image_paths: List[str], out_path: str, ncols: int, white_background: bool = False) -> None:
    if not image_paths:
        return

    ncols = max(1, ncols)
    nrows = int(math.ceil(len(image_paths) / ncols))

    bg_color = "white" if white_background else "black"
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.02, nrows * 1.02),
        facecolor=bg_color,
    )

    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, ax in enumerate(axes_arr.ravel()):
        ax.set_facecolor(bg_color)
        if i >= len(image_paths):
            ax.axis("off")
            continue

        img_path = image_paths[i]
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MRI module NPZ maps on T1w norm with GT-centered tri-planar views."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing prediction maps named <subject_id>.npz or <subject_id>.nii.gz.",
    )
    parser.add_argument(
        "--mri_dir",
        default=MRI_DIR_DEFAULT,
        help="Directory containing *_T1w_norm.nii.gz, *_FLAIR_norm.nii.gz and *_gt_norm.nii.gz files (recursive search).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root directory. Default: <pred_dir>/mri_npz_visualizations.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of columns in the combined grid image.",
    )
    parser.add_argument(
        "--map_alpha",
        type=float,
        default=0.6,
        help="Alpha for map overlay in [0,1].",
    )
    parser.add_argument(
        "--map_cmap",
        type=str,
        default="cool",
        help="Matplotlib colormap for map overlay.",
    )
    parser.add_argument(
        "--map_source",
        type=str,
        choices=["npz", "nii"],
        default="npz",
        help="Prediction map source format. Use 'nii' to plot .nii.gz maps instead of .npz.",
    )
    parser.add_argument(
        "--npz_key",
        type=str,
        default=None,
        help="Optional explicit NPZ key to use for map extraction.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Optional channel index for 4D arrays (C,D,H,W). Default prefers foreground channel 1.",
    )
    parser.add_argument(
        "--align_max_shift",
        type=int,
        default=128,
        help="Max translation clamp (voxels) per axis after NPZ registration to same-subject NIfTI mask.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Enable SimpleITK translation registration after affine mapping (disabled by default).",
    )
    parser.add_argument(
        "--disable_npz_match_nii_orientation",
        action="store_true",
        help=(
            "Disable axis permutation/flip matching of NPZ array to paired NIfTI map "
            "before affine mapping. Enabled by default."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of subjects to render.",
    )
    parser.add_argument(
        "--enable_no_gt_multislice",
        action="store_true",
        help=(
            "If set, all subjects are rendered as a 3xN multi-slice panel "
            "(Transversal/Sagittal/Coronal) with one shared colorbar, regardless of GT availability."
        ),
    )
    parser.add_argument(
        "--enable_hotspot_clusters",
        action="store_true",
        help=(
            "If set together with --enable_no_gt_multislice, also render cropped multislice "
            "plots around each connected hotspot cluster in the prediction map above 0.3."
        ),
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help=(
            "Enable test-set fold discovery mode. Expects fold_<idx> subdirectories under "
            "--pred_dir and plots all fold predictions without k-fold val-id filtering."
        ),
    )
    parser.add_argument(
        "--combined_pred_dir",
        type=str,
        default=None,
        help=(
            "Optional directory with pre-computed combined prediction maps (e.g. ensemble "
            "means across folds). Defaults to <pred_dir>/combined_preds "
            "when --test_set is enabled."
        ),
    )
    parser.add_argument(
        "--enable_plot_raw_scans",
        action="store_true",
        help=(
            "Also save raw T1/FLAIR images with GT contours in per_subject/raw_scans "
            "using the same slice selection as the subject map plots."
        ),
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Render figures with a white background and white zero-valued scan voxels.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.pred_dir, f"mri_{args.map_source}_visualizations")
    per_subject_dir = os.path.join(output_dir, "per_subject")
    os.makedirs(per_subject_dir, exist_ok=True)
    raw_scan_dir = os.path.join(per_subject_dir, "raw_scans") if args.enable_plot_raw_scans else None
    if raw_scan_dir is not None:
        os.makedirs(raw_scan_dir, exist_ok=True)

    fold_dirs = _find_fold_dirs(args.pred_dir)
    use_fold_mode = len(fold_dirs) > 0

    pred_items: List[Dict[str, Optional[str]]] = []

    if args.test_set:
        if not use_fold_mode:
            raise FileNotFoundError(
                f"--test_set enabled but no fold_<idx> directories found under: {args.pred_dir}"
            )

        print(f"[test_set] Detected fold directories: {list(sorted(fold_dirs.keys()))}")
        for fold_name, fold_dir in sorted(fold_dirs.items()):
            if args.map_source == "npz":
                fold_pred_map = _index_subject_npz(fold_dir, recursive=True)
                fold_nii_map = _index_subject_nii(fold_dir, recursive=True)
            else:
                fold_pred_map = _index_subject_nii(fold_dir, recursive=True)
                fold_nii_map = {}

            for subject_id in sorted(fold_pred_map.keys()):
                pred_items.append(
                    {
                        "subject_id": subject_id,
                        "display_id": f"{fold_name}_{subject_id}",
                        "pred_path": fold_pred_map[subject_id],
                        "pred_nii_path": fold_nii_map.get(subject_id),
                    }
                )

        combined_pred_dir = args.combined_pred_dir or os.path.join(
            args.pred_dir,
            "combined_preds",
        )
        if os.path.isdir(combined_pred_dir):
            if args.map_source == "npz":
                combined_map = _index_subject_npz(combined_pred_dir, recursive=False)
                combined_nii_map = _index_subject_nii(combined_pred_dir, recursive=False)
            else:
                combined_map = _index_subject_nii(combined_pred_dir, recursive=False)
                combined_nii_map = {}

            for subject_id in sorted(combined_map.keys()):
                pred_items.append(
                    {
                        "subject_id": subject_id,
                        "display_id": f"combined_map_{subject_id}",
                        "pred_path": combined_map[subject_id],
                        "pred_nii_path": combined_nii_map.get(subject_id),
                    }
                )
            print(
                f"[test_set] Added {len(combined_map)} combined maps from: {combined_pred_dir}"
            )
        else:
            tqdm.write(
                f"[test_set] Combined map directory not found (skipping): {combined_pred_dir}"
            )
    elif use_fold_mode:
        if not os.path.isfile(K_FOLD_SPLITS_PATH):
            raise FileNotFoundError(f"K-fold splits JSON not found: {K_FOLD_SPLITS_PATH}")

        val_ids_by_fold = _load_kfold_val_ids(K_FOLD_SPLITS_PATH)
        print(f"Detected fold directories: {list(fold_dirs.keys())}")
        for fold_name, fold_dir in sorted(fold_dirs.items()):
            val_ids = val_ids_by_fold.get(fold_name, set())
            if not val_ids:
                tqdm.write(f"[skip-fold] {fold_name}: missing or empty val_ids in {K_FOLD_SPLITS_PATH}")
                continue

            if args.map_source == "npz":
                fold_pred_map = _index_subject_npz(fold_dir, recursive=True)
                fold_nii_map = _index_subject_nii(fold_dir, recursive=True)
            else:
                fold_pred_map = _index_subject_nii(fold_dir, recursive=True)
                fold_nii_map = {}

            for subject_id in sorted(set(fold_pred_map.keys()) & val_ids):
                pred_items.append(
                    {
                        "subject_id": subject_id,
                        "display_id": f"{fold_name}_{subject_id}",
                        "pred_path": fold_pred_map[subject_id],
                        "pred_nii_path": fold_nii_map.get(subject_id),
                    }
                )
    else:
        if args.map_source == "npz":
            pred_map = _index_subject_npz(args.pred_dir, recursive=True)
        else:
            pred_map = _index_subject_nii(args.pred_dir, recursive=True)
        pred_nii_map = _index_subject_nii(args.pred_dir, recursive=True) if args.map_source == "npz" else {}

        for subject_id in sorted(pred_map.keys()):
            pred_items.append(
                {
                    "subject_id": subject_id,
                    "display_id": subject_id,
                    "pred_path": pred_map[subject_id],
                    "pred_nii_path": pred_nii_map.get(subject_id),
                }
            )

    t1_map = _index_files_recursive(args.mri_dir, T1_SUFFIX)
    flair_map = _index_files_recursive_multi_suffix(args.mri_dir, FLAIR_SUFFIXES)
    gt_map = _index_files_recursive(args.mri_dir, GT_SUFFIX)

    if args.limit is not None:
        pred_items = pred_items[: max(args.limit, 0)]

    if not pred_items:
        ext_msg = ".npz" if args.map_source == "npz" else ".nii.gz"
        raise FileNotFoundError(f"No {ext_msg} prediction files found under: {args.pred_dir}")

    saved_paths: List[str] = []
    cluster_paths: List[str] = []
    skipped_missing = 0
    skipped_empty_gt = 0
    skipped_map_load = 0
    skipped_missing_affine_nii = 0
    applied_manual_alignment = 0
    rendered_multislice = 0
    raw_scan_images = 0
    raw_scan_skipped = 0

    for item in tqdm(pred_items, desc="Rendering subjects", unit="subject"):
        subject_id = str(item["subject_id"])
        display_id = str(item["display_id"])
        pred_path = str(item["pred_path"])
        pred_nii_path = item.get("pred_nii_path")
        t1_path = t1_map.get(subject_id)
        gt_path = gt_map.get(subject_id)

        if t1_path is None:
            skipped_missing += 1
            continue

        t1_img = nib.load(t1_path)
        t1 = np.asarray(t1_img.get_fdata(dtype=np.float32), dtype=np.float32)
        gt = np.zeros_like(t1, dtype=np.float32)
        has_gt = gt_path is not None
        if has_gt:
            gt_img = nib.load(str(gt_path))
            gt = np.asarray(gt_img.get_fdata(dtype=np.float32), dtype=np.float32)

        try:
            if args.map_source == "npz":
                map3d_raw = _load_pred_map_from_npz(
                    pred_path,
                    reference_shape=None,
                    npz_key=args.npz_key,
                    channel=args.channel,
                )
                if pred_nii_path is None:
                    tqdm.write(
                        f"[skip] {subject_id}: no same-subject .nii.gz found in pred_dir "
                        "for affine-based NPZ mapping"
                    )
                    skipped_missing_affine_nii += 1
                    continue

                pred_nii_img = nib.load(pred_nii_path)

                if not args.disable_npz_match_nii_orientation:
                    pred_nii_data = np.asarray(pred_nii_img.get_fdata(dtype=np.float32), dtype=np.float32)
                    map3d_raw, transform_desc, transform_score = _reorient_npz_to_reference(
                        map3d_raw,
                        pred_nii_data,
                    )
                    tqdm.write(
                        f"[orient] {subject_id}: {transform_desc}, similarity={transform_score:.6f}"
                    )
                map3d = _map_npz_to_mri_space(
                    map3d=map3d_raw,
                    map_affine=pred_nii_img.affine,
                    target_nii=t1_img,
                )
                if args.register:
                    nii_mask_t1 = _resample_mask_to_target(pred_nii_img, t1_img)
                    map3d, best_shift, best_score = _register_map_to_mask_translation(
                        map3d=map3d,
                        mask=nii_mask_t1,
                        max_shift=max(0, int(args.align_max_shift)),
                    )
                    applied_manual_alignment += 1
                    tqdm.write(
                        f"[align] {subject_id}: reg_shift={best_shift}, "
                        f"mean(map[nii>0])={best_score:.5f}"
                    )
                map3d = _to_probability_like(map3d)
            else:
                map3d = np.asarray(nib.load(pred_path).get_fdata(dtype=np.float32), dtype=np.float32)
                if map3d.ndim != 3:
                    raise ValueError(f"Expected 3D NIfTI map, got shape {map3d.shape}")
                map3d = _to_probability_like(map3d)
        except Exception as exc:
            tqdm.write(f"[skip] {subject_id}: could not load map from {pred_path}: {exc}")
            skipped_map_load += 1
            continue

        if map3d.shape != t1.shape or gt.shape != t1.shape:
            tqdm.write(
                f"[skip] {subject_id}: shape mismatch "
                f"map={map3d.shape}, t1={t1.shape}, gt={gt.shape}"
            )
            skipped_missing += 1
            continue

        out_path = os.path.join(per_subject_dir, f"{display_id}.png")
        if args.enable_no_gt_multislice:
            ok = _plot_single_subject_no_gt_multislice(
                subject_id=display_id,
                t1=t1,
                pred_map=map3d,
                out_path=out_path,
                map_alpha=float(np.clip(args.map_alpha, 0.0, 1.0)),
                map_cmap=args.map_cmap,
                n_slices=16,
                footer_text="multislice",
                white_background=args.white_background,
            )
            if ok:
                rendered_multislice += 1

            if ok and args.enable_hotspot_clusters:
                hotspot_clusters = _find_hotspot_clusters(map3d, threshold=0.3)
                if hotspot_clusters:
                    for cluster_idx, cluster in enumerate(hotspot_clusters, start=1):
                        bbox_min = np.asarray(cluster["bbox_min"], dtype=int)
                        bbox_max = np.asarray(cluster["bbox_max"], dtype=int)
                        cluster_center = tuple(int(round((lo + hi - 1) / 2.0)) for lo, hi in zip(bbox_min, bbox_max))
                        cluster_name = f"{display_id}_cluster{cluster_idx}"
                        cluster_out_path = os.path.join(per_subject_dir, f"{cluster_name}.png")
                        cluster_ok = _plot_single_subject_no_gt_multislice(
                            subject_id=cluster_name,
                            t1=t1,
                            pred_map=map3d,
                            out_path=cluster_out_path,
                            map_alpha=float(np.clip(args.map_alpha, 0.0, 1.0)),
                            map_cmap=args.map_cmap,
                            n_slices=16,
                            footer_text=f"cluster{cluster_idx} > 0.3 | vox={int(cluster['voxel_count'])}",
                            focus_center=cluster_center,
                            focus_window_fraction=0.18,
                            cluster_bbox=(tuple(int(v) for v in bbox_min), tuple(int(v) for v in bbox_max)),
                            white_background=args.white_background,
                        )
                        if cluster_ok:
                            cluster_paths.append(cluster_out_path)
        else:
            ok = _plot_single_subject(
                subject_id=display_id,
                t1=t1,
                pred_map=map3d,
                gt=gt,
                out_path=out_path,
                map_alpha=float(np.clip(args.map_alpha, 0.0, 1.0)),
                map_cmap=args.map_cmap,
                white_background=args.white_background,
            )

        if ok:
            saved_paths.append(out_path)

            if raw_scan_dir is not None:
                t1_raw_out = os.path.join(raw_scan_dir, f"{display_id}_T1w.png")
                if args.enable_no_gt_multislice:
                    t1_raw_ok = _plot_raw_scan_multislice(
                        subject_id=display_id,
                        modality_name="T1w",
                        scan=t1,
                        gt=gt,
                        out_path=t1_raw_out,
                        n_slices=16,
                        white_background=args.white_background,
                    )
                else:
                    t1_raw_ok = _plot_raw_scan_triplet(
                        subject_id=display_id,
                        modality_name="T1w",
                        scan=t1,
                        gt=gt,
                        out_path=t1_raw_out,
                        white_background=args.white_background,
                    )
                if t1_raw_ok:
                    raw_scan_images += 1
                else:
                    raw_scan_skipped += 1

                flair_path = flair_map.get(subject_id)
                if flair_path is not None:
                    try:
                        flair = np.asarray(nib.load(flair_path).get_fdata(dtype=np.float32), dtype=np.float32)
                    except Exception:
                        flair = None

                    if flair is None or flair.shape != t1.shape:
                        raw_scan_skipped += 1
                    else:
                        flair_raw_out = os.path.join(raw_scan_dir, f"{display_id}_FLAIR.png")
                        if args.enable_no_gt_multislice:
                            flair_raw_ok = _plot_raw_scan_multislice(
                                subject_id=display_id,
                                modality_name="FLAIR",
                                scan=flair,
                                gt=gt,
                                out_path=flair_raw_out,
                                n_slices=16,
                                white_background=args.white_background,
                            )
                        else:
                            flair_raw_ok = _plot_raw_scan_triplet(
                                subject_id=display_id,
                                modality_name="FLAIR",
                                scan=flair,
                                gt=gt,
                                out_path=flair_raw_out,
                                white_background=args.white_background,
                            )
                        if flair_raw_ok:
                            raw_scan_images += 1
                        else:
                            raw_scan_skipped += 1
        else:
            skipped_empty_gt += 1

    if not saved_paths:
        raise RuntimeError(
            "No per-subject images were generated. "
            "Check if GT masks are non-empty and NPZ keys/shapes match T1/GT."
        )

    grid_path = os.path.join(output_dir, "all_subjects_grid.png")
    _build_grid_image(saved_paths, grid_path, ncols=args.grid_cols, white_background=args.white_background)

    print("\nDone.")
    print(f"  Per-subject images: {len(saved_paths)}")
    if args.enable_hotspot_clusters:
        print(f"  Cluster images: {len(cluster_paths)}")
    print(f"  Grid image: {grid_path}")
    print(f"  Skipped (missing T1/GT or shape mismatch): {skipped_missing}")
    print(f"  Skipped (empty GT): {skipped_empty_gt}")
    print(f"  Skipped (could not load NPZ map): {skipped_map_load}")
    if args.enable_no_gt_multislice:
        print(f"  Rendered (multi-slice): {rendered_multislice}")
    if args.map_source == "npz":
        print(f"  Registration enabled: {args.register}")
        if args.register:
            print(f"  Manual NPZ alignments applied: {applied_manual_alignment}")
        print(f"  Skipped (missing same-subject NIfTI for affine): {skipped_missing_affine_nii}")
    if raw_scan_dir is not None:
        print(f"  Raw scan images: {raw_scan_images}")
        print(f"  Raw scan skips (missing FLAIR/GT center/shape): {raw_scan_skipped}")


if __name__ == "__main__":
    main()
