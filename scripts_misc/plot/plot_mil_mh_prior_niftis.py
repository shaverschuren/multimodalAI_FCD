"""
Visualize EEG MIL multi-head prior NIfTI outputs against MRI preprocessing outputs.

For each patient with a prior NIfTI file, this script loads:
- prior map: commonly ``{patient_id}_deconv_prior.nii.gz`` or ``{patient_id}_prior.nii.gz``
- normalized T1 image: ``{patient_id}_T1w_norm.nii.gz``
- normalized GT mask: ``{patient_id}_gt_norm.nii.gz``

It creates a tri-planar (axial/coronal/sagittal) figure centered on the GT mask
center of mass, overlays prior on top of T1, then overlays GT contour, and saves:
1) one PNG per patient
2) one combined grid PNG containing all per-patient plots

Usage
-----
python scripts_other/plot/plot_mil_mh_prior_niftis.py \
    --prior_dir /path/to/prior_niftis \
    --mri_dir /path/to/preprocessed_mri \
    --output_dir /path/to/output
"""

import argparse
import json
import math
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import nibabel as nib
import numpy as np
from tqdm import tqdm


DEFAULT_PRIOR_SUFFIXES = "_deconv_prior.nii.gz,_prior.nii.gz"
T1_SUFFIX = "_T1w_norm.nii.gz"
GT_SUFFIX = "_gt_norm.nii.gz"
FLAIR_SUFFIXES = ("_FLAIR_norm.nii.gz", "_flair_norm.nii.gz")
MRI_DEFAULT_DIR = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri"

def _extract_pid(path: str, suffix: str) -> Optional[str]:
    name = os.path.basename(path)
    if not name.endswith(suffix):
        return None
    name_without_suffix = name[: -len(suffix)]
    match = re.match(r'(RESP\d+)', name_without_suffix)
    if match:
        return match.group(1)
    return name_without_suffix


def _index_files(root_dir: str, suffix: str, must_contain_dir: Optional[str] = None) -> Dict[str, str]:
    pattern = os.path.join(root_dir, "**", f"*{suffix}")
    matches = sorted(glob(pattern, recursive=True))
    mapping: Dict[str, str] = {}
    for path in matches:
        if must_contain_dir is not None:
            path_parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
            if must_contain_dir.lower() not in path_parts:
                continue
        pid = _extract_pid(path, suffix)
        if pid is None:
            continue
        # Keep first match for deterministic behavior if duplicates exist.
        if pid not in mapping:
            mapping[pid] = path
    return mapping


def _index_files_multi_suffix(
    root_dir: str,
    suffixes: Tuple[str, ...],
    must_contain_dir: Optional[str] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for suffix in suffixes:
        current = _index_files(root_dir, suffix, must_contain_dir=must_contain_dir)
        for pid, path in current.items():
            if pid not in mapping:
                mapping[pid] = path
    return mapping


def _index_prior_files(
    root_dir: str,
    suffixes: List[str],
    must_contain_dir: Optional[str] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    normalized = [s.strip() for s in suffixes if s.strip()]
    for suffix in normalized:
        current = _index_files(root_dir, suffix, must_contain_dir=must_contain_dir)
        for pid, path in current.items():
            if pid not in mapping:
                mapping[pid] = path
    return mapping


def _load_validation_case_metrics(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load per-case metrics from all validation.json files under val folders."""
    pattern = os.path.join(root_dir, "**", "validation.json")
    files = sorted(glob(pattern, recursive=True))
    cases_by_pid: Dict[str, Dict[str, Any]] = {}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[warn] Could not read {path}: {exc}")
            continue

        case_entries = payload.get("cases", payload)
        if not isinstance(case_entries, dict):
            continue

        for patient_id, entry in case_entries.items():
            if patient_id not in cases_by_pid and isinstance(entry, dict):
                cases_by_pid[patient_id] = entry

    return cases_by_pid


def _load_split_case_metrics(root_dir: str, split_name: str) -> Dict[str, Dict[str, Any]]:
    """Load per-case metrics from all <split_name>.json files recursively."""
    metrics_file = f"{split_name}.json"
    pattern = os.path.join(root_dir, "**", metrics_file)
    files = sorted(glob(pattern, recursive=True))
    cases_by_pid: Dict[str, Dict[str, Any]] = {}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[warn] Could not read {path}: {exc}")
            continue

        case_entries = payload.get("cases", payload)
        if not isinstance(case_entries, dict):
            continue

        for patient_id, entry in case_entries.items():
            if patient_id not in cases_by_pid and isinstance(entry, dict):
                cases_by_pid[patient_id] = entry

    return cases_by_pid


def _load_split_case_metrics_with_fallback(
    root_dir: str,
    split_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Load per-case metrics from the first split JSON that exists."""
    for split_name in split_names:
        cases = _load_split_case_metrics(root_dir, split_name=split_name)
        if cases:
            return cases
    return {}


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


def _overlay_alpha(map_slice: np.ndarray, alpha_constant: float, high_end: float = 0.3) -> np.ndarray:
    """Scale overlay alpha by map strength with saturation above ``high_end``."""
    high_end = max(float(high_end), 1e-6)
    alpha_map = np.clip(np.asarray(map_slice, dtype=np.float32) / high_end, 0.0, 1.0)
    alpha_constant = float(np.clip(alpha_constant, 0.0, 1.0))
    return alpha_constant * alpha_map


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

    # Rotate to improve display orientation consistency.
    return {
        "Axial": np.flipud(np.rot90(volume[:, :, z])),
        "Coronal": np.flipud(np.rot90(volume[:, y, :])),
        "Sagittal": np.flipud(np.rot90(volume[x, :, :])),
    }


def _fmt_vec(values: Any, decimals: int = 3) -> str:
    if values is None:
        return "n/a"
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return "n/a"
    if arr.size == 0:
        return "n/a"
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr.tolist()) + "]"


def _plot_single_patient(
    patient_id: str,
    t1: np.ndarray,
    prior: np.ndarray,
    gt: np.ndarray,
    out_path: str,
    prior_alpha: float,
    prior_cmap: str,
    reported_metrics: Optional[Dict[str, Any]] = None,
    white_background: bool = False,
) -> bool:
    center = _gt_center(gt)
    if center is None:
        return False

    # Restrict prior to voxels supported by the normalized T1 image.
    prior = np.where(t1 > _safe_percentile(t1, 10, fallback=0.0), prior, 0.0).astype(np.float32)
    prior = np.clip(prior, 0.0, 1.0).astype(np.float32)

    t1_slices = _slice_triplet(t1, center)
    prior_slices = _slice_triplet(prior, center)
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
        prior_sl = prior_slices[plane]
        gt_sl = gt_slices[plane]

        ax.imshow(
            _mask_zero_background(t1_sl, white_background),
            cmap=gray_cmap,
            vmin=t1_vmin,
            vmax=t1_vmax,
            origin="lower",
        )
        ax.imshow(
            prior_sl,
            cmap=prior_cmap,
            alpha=_overlay_alpha(prior_sl, prior_alpha),
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        if np.any(gt_sl > 0):
            ax.contour(gt_sl, levels=[0.5], colors=["#ff4040"], linewidths=1.0)

        ax.axis("off")

    cax.set_facecolor(bg_color)
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=prior_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Prior", color=fg_color, fontsize=7)
    cbar.ax.tick_params(colors=fg_color, labelsize=7, length=2)
    cbar.outline.set_edgecolor(fg_color)

    x, y, z = center
    metrics = reported_metrics or {}
    pred_mu = metrics.get("pred_mu", metrics.get("mu"))
    pred_sigma = metrics.get("pred_sigma", metrics.get("sigma"))
    err_mm = metrics.get("coord_euclidean_mm", metrics.get("euclidean_mm"))
    err_text = f"{float(err_mm):.2f} mm" if isinstance(err_mm, (int, float)) else "n/a"

    metadata_text = (
        f"{patient_id} | c=({x},{y},{z}) | "
        f"mu={_fmt_vec(pred_mu)} | sigma={_fmt_vec(pred_sigma)} | err={err_text}"
    )
    fig.text(
        0.006,
        0.012,
        metadata_text,
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)
    return True


def _plot_raw_scan_triplet(
    patient_id: str,
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
        f"{patient_id} | {modality_name} | c=({x},{y},{z})",
        color=fg_color,
        fontsize=6.7,
        va="bottom",
        ha="left",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
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
        patient_id = os.path.basename(img_path).replace(".png", "")

        ax.imshow(image)
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.0, facecolor=bg_color)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot prior NIfTI overlays on T1w norm with GT-centered tri-planar views."
    )
    parser.add_argument(
        "--prior_dir",
        required=True,
        help="Directory containing prior NIfTI files (recursive search).",
    )
    parser.add_argument(
        "--mri_dir",
        required=False,
        default=MRI_DEFAULT_DIR,
        help="Directory containing *_T1w_norm.nii.gz, *_FLAIR_norm.nii.gz and *_gt_norm.nii.gz files (recursive search).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root directory. Default: <prior_dir>/prior_visualizations.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of columns in the combined grid image.",
    )
    parser.add_argument(
        "--prior_alpha",
        type=float,
        default=0.45,
        help="Alpha for prior overlay in [0,1].",
    )
    parser.add_argument(
        "--prior_cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for prior overlay.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of patients to render.",
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help=(
            "If set, restrict prior files to the test split folder and prefer "
            "test.json metrics when present, falling back to validation.json."
        ),
    )
    parser.add_argument(
        "--prior_suffixes",
        type=str,
        default=DEFAULT_PRIOR_SUFFIXES,
        help=(
            "Comma-separated prior filename suffixes to match. "
            "Default: '_deconv_prior.nii.gz,_prior.nii.gz'."
        ),
    )
    parser.add_argument(
        "--split_dir_token",
        type=str,
        default=None,
        help=(
            "Optional directory-name token filter when indexing prior files "
            "(e.g. 'test' or 'val')."
        ),
    )
    parser.add_argument(
        "--enable_plot_raw_scans",
        action="store_true",
        help=(
            "Also save raw T1/FLAIR images with GT contours in per_subject/raw_scans "
            "using the same slices as the patient overlay plots."
        ),
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Render figures with a white background and white zero-valued scan voxels.",
    )
    args = parser.parse_args()

    split_name = "test" if args.test_set else "validation"
    suffixes = [s for s in args.prior_suffixes.split(",") if s.strip()]
    split_dir_token = args.split_dir_token
    if args.test_set and split_dir_token is None:
        split_dir_token = "test"

    output_dir = args.output_dir or os.path.join(args.prior_dir, "prior_visualizations")
    per_patient_dir = os.path.join(output_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)
    raw_scan_dir = os.path.join(output_dir, "per_subject", "raw_scans") if args.enable_plot_raw_scans else None
    if raw_scan_dir is not None:
        os.makedirs(raw_scan_dir, exist_ok=True)

    prior_map = _index_prior_files(
        args.prior_dir,
        suffixes=suffixes,
        must_contain_dir=split_dir_token,
    )
    split_case_metrics = _load_split_case_metrics_with_fallback(
        args.prior_dir,
        split_names=[split_name, "validation"] if args.test_set else [split_name],
    )
    t1_map = _index_files(args.mri_dir, T1_SUFFIX)
    flair_map = _index_files_multi_suffix(args.mri_dir, FLAIR_SUFFIXES)
    gt_map = _index_files(args.mri_dir, GT_SUFFIX)

    all_patient_ids = sorted(prior_map.keys())
    if args.limit is not None:
        all_patient_ids = all_patient_ids[: max(args.limit, 0)]

    if not all_patient_ids:
        raise FileNotFoundError(
            f"No prior NIfTI files found under: {args.prior_dir}"
        )

    saved_paths: List[str] = []
    skipped_missing = 0
    skipped_empty_gt = 0
    raw_scan_images = 0
    raw_scan_skipped = 0

    for patient_id in tqdm(all_patient_ids, desc="Rendering patients", unit="patient"):
        prior_path = prior_map[patient_id]
        t1_path = t1_map.get(patient_id)
        gt_path = gt_map.get(patient_id)

        if t1_path is None or gt_path is None:
            skipped_missing += 1
            continue

        prior = np.asarray(nib.load(prior_path).get_fdata(dtype=np.float32), dtype=np.float32)
        t1 = np.asarray(nib.load(t1_path).get_fdata(dtype=np.float32), dtype=np.float32)
        gt = np.asarray(nib.load(gt_path).get_fdata(dtype=np.float32), dtype=np.float32)

        if prior.shape != t1.shape or gt.shape != t1.shape:
            print(
                f"[skip] {patient_id}: shape mismatch "
                f"prior={prior.shape}, t1={t1.shape}, gt={gt.shape}"
            )
            skipped_missing += 1
            continue

        out_path = os.path.join(per_patient_dir, f"{patient_id}.png")
        ok = _plot_single_patient(
            patient_id=patient_id,
            t1=t1,
            prior=prior,
            gt=gt,
            out_path=out_path,
            prior_alpha=float(np.clip(args.prior_alpha, 0.0, 1.0)),
            prior_cmap=args.prior_cmap,
            reported_metrics=split_case_metrics.get(patient_id),
            white_background=args.white_background,
        )

        if ok:
            saved_paths.append(out_path)

            if raw_scan_dir is not None:
                t1_raw_out = os.path.join(raw_scan_dir, f"{patient_id}_T1w.png")
                t1_raw_ok = _plot_raw_scan_triplet(
                    patient_id=patient_id,
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

                flair_path = flair_map.get(patient_id)
                if flair_path is not None:
                    try:
                        flair = np.asarray(nib.load(flair_path).get_fdata(dtype=np.float32), dtype=np.float32)
                    except Exception:
                        flair = None

                    if flair is None or flair.shape != t1.shape:
                        raw_scan_skipped += 1
                    else:
                        flair_raw_out = os.path.join(raw_scan_dir, f"{patient_id}_FLAIR.png")
                        flair_raw_ok = _plot_raw_scan_triplet(
                            patient_id=patient_id,
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
            "No per-patient images were generated. "
            "Check if GT masks are non-empty and files are matched by patient ID."
        )

    grid_path = os.path.join(output_dir, "all_patients_grid.png")
    _build_grid_image(saved_paths, grid_path, ncols=args.grid_cols, white_background=args.white_background)

    print("\nDone.")
    print(f"  Per-patient images: {len(saved_paths)}")
    print(f"  Grid image: {grid_path}")
    print(f"  Skipped (missing T1/GT or shape mismatch): {skipped_missing}")
    print(f"  Skipped (empty GT): {skipped_empty_gt}")
    if raw_scan_dir is not None:
        print(f"  Raw scan images: {raw_scan_images}")
        print(f"  Raw scan skips (missing FLAIR/GT center/shape): {raw_scan_skipped}")


if __name__ == "__main__":
    main()
