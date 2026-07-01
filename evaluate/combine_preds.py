"""
Combine fold-wise predictions into one per-subject map.

Supported inputs:
- NPZ predictions (nnUNet-style probabilities)
- NIfTI predictions (e.g., EEG prior maps such as *_deconv_prior.nii.gz)

Expected layout:
    <pred_dir>/fold_0/
    <pred_dir>/fold_1/
    ...

For every subject present in >= --min-folds folds, this script computes a
voxel-wise ensemble mean (arithmetic or geometric) and writes combined outputs.

Usage
-----
python evaluate/combine_preds.py \
        --pred_dir /path/to/fold_predictions_root \
        --input_format npz \
        --output_dir /path/to/combined_preds

python evaluate/combine_preds.py \
        --pred_dir /path/to/fold_predictions_root \
        --input_format nii \
    --nii_suffixes _deconv_prior.nii.gz,_prior.nii.gz \
        --output_dir /path/to/combined_priors
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

SUBJECT_ID_REGEX = re.compile(r"(RESP\d+)", re.IGNORECASE)
NII_SUFFIX = (".nii.gz", ".nii")


def _extract_subject_id(name: str) -> Optional[str]:
    match = SUBJECT_ID_REGEX.search(str(name))
    if match is None:
        return None
    return match.group(1).upper()


def _find_fold_dirs(pred_dir: str) -> Dict[str, str]:
    fold_dirs: Dict[str, str] = {}
    try:
        children = sorted(os.listdir(pred_dir))
    except OSError:
        return fold_dirs

    for name in children:
        path = os.path.join(pred_dir, name)
        if os.path.isdir(path) and re.fullmatch(r"fold_\d+", name):
            fold_dirs[name] = path
    return fold_dirs


def _index_subject_npz(pred_dir: str, recursive: bool = False) -> Dict[str, str]:
    pattern = os.path.join(pred_dir, "**", "*.npz") if recursive else os.path.join(pred_dir, "*.npz")
    mapping: Dict[str, str] = {}
    for path in sorted(glob(pattern, recursive=recursive)):
        sid = _extract_subject_id(os.path.basename(path))
        if sid and sid not in mapping:
            mapping[sid] = path
    return mapping


def _index_subject_nii(
    pred_dir: str,
    recursive: bool = False,
    required_suffixes: Optional[List[str]] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if required_suffixes is not None:
        suffixes = tuple(s.strip() for s in required_suffixes if s.strip())
    else:
        suffixes = NII_SUFFIX

    for suffix in suffixes:
        pattern = os.path.join(pred_dir, "**", f"*{suffix}") if recursive else os.path.join(pred_dir, f"*{suffix}")
        for path in sorted(glob(pattern, recursive=recursive)):
            sid = _extract_subject_id(os.path.basename(path))
            if sid and sid not in mapping:
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


def _select_3d_map(arr: np.ndarray, channel: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D map, got shape {arr.shape}")

    if channel is None:
        ch = 1 if arr.shape[0] >= 2 else 0
    else:
        ch = int(np.clip(channel, 0, arr.shape[0] - 1))
    return np.asarray(arr[ch], dtype=np.float32)


def _to_probability_like(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[finite]
    vmin = float(vals.min())
    vmax = float(vals.max())

    if vmin >= 0.0 and vmax <= 1.0:
        out = arr.copy()
        out[~finite] = 0.0
        return out

    if vmin >= -20.0 and vmax <= 20.0:
        out = 1.0 / (1.0 + np.exp(-arr))
        out[~finite] = 0.0
        return out.astype(np.float32)

    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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


def _load_pred_map_from_npz(npz_path: str, npz_key: Optional[str], channel: Optional[int]) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as payload:
        keys = list(payload.keys())
        if not keys:
            raise ValueError("NPZ has no arrays")

        if npz_key is not None:
            if npz_key not in payload:
                raise KeyError(f"Requested npz_key '{npz_key}' not found. Available keys: {keys}")
            return _to_probability_like(_select_3d_map(payload[npz_key], channel=channel))

        candidates: List[Tuple[int, str, np.ndarray]] = []
        for key in keys:
            try:
                arr3d = _select_3d_map(payload[key], channel=channel)
            except Exception:
                continue
            candidates.append((_score_key(key), key, arr3d))

        if not candidates:
            raise ValueError(f"No 3D/4D array found in {npz_path}. Available keys: {keys}")

        candidates.sort(key=lambda t: (t[0], t[1]))
        return _to_probability_like(candidates[0][2])


def _geometric_mean(stack: np.ndarray, eps: float) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    stack = np.clip(stack, float(eps), 1.0)
    return np.exp(np.mean(np.log(stack), axis=0)).astype(np.float32)


def _arithmetic_mean(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    return np.mean(stack, axis=0).astype(np.float32)


def _infer_subject_nii_suffix(filename: str, subject_id: str) -> Optional[str]:
    base = os.path.basename(filename)
    sid = subject_id.upper()
    base_upper = base.upper()
    if base_upper.startswith(sid):
        suffix = base[len(subject_id) :]
        if suffix and (suffix.endswith(".nii") or suffix.endswith(".nii.gz")):
            return suffix
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine fold-wise predictions by voxel-wise ensemble (arithmetic or geometric mean).")
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Root directory containing fold_<idx> subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output directory for combined maps. "
            "Default: <pred_dir>/combined_preds"
        ),
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["npz", "nii"],
        default="npz",
        help="Input map format to combine. Use 'nii' for EEG prior NIfTIs.",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        choices=["arithmetic", "geometric"],
        default="arithmetic",
        help="Voxel-wise ensemble method: 'arithmetic' (default) or 'geometric' mean.",
    )
    parser.add_argument(
        "--min_folds",
        type=int,
        default=2,
        help="Minimum number of folds required to combine a subject (default: 2).",
    )
    parser.add_argument(
        "--npz_key",
        type=str,
        default=None,
        help="Optional explicit NPZ key to load.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Optional channel index for 4D arrays in NPZ files.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Lower clipping bound before log in geometric mean (default: 1e-6).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search inside each fold directory for prediction files.",
    )
    parser.add_argument(
        "--nii_suffixes",
        type=str,
        default="_deconv_prior.nii.gz,_prior.nii.gz",
        help=(
            "Optional comma-separated required suffixes for NIfTI inputs "
            "(e.g. _deconv_prior.nii.gz,_prior.nii.gz). "
            "Only used when --input_format nii."
        ),
    )
    parser.add_argument(
        "--disable_npz_match_nii_orientation",
        action="store_true",
        help=(
            "Disable axis permutation/flip matching of NPZ array to paired NIfTI map "
            "before geometric-mean combination."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_dir = args.pred_dir
    output_dir = args.output_dir or os.path.join(pred_dir, "combined_preds")
    os.makedirs(output_dir, exist_ok=True)

    fold_dirs = _find_fold_dirs(pred_dir)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_<idx> directories found under: {pred_dir}")

    recursive_search = bool(args.recursive or args.input_format == "nii")

    fold_maps: Dict[str, Dict[str, str]] = {}
    fold_ref_nii: Dict[str, Dict[str, str]] = {}
    for fold_name, fold_dir in sorted(fold_dirs.items()):
        if args.input_format == "npz":
            fold_maps[fold_name] = _index_subject_npz(fold_dir, recursive=recursive_search)
            fold_ref_nii[fold_name] = _index_subject_nii(fold_dir, recursive=recursive_search)
        else:
            nii_suffixes = [s for s in args.nii_suffixes.split(",") if s.strip()]
            fold_maps[fold_name] = _index_subject_nii(
                fold_dir,
                recursive=recursive_search,
                required_suffixes=nii_suffixes,
            )
            fold_ref_nii[fold_name] = {}

    all_subjects = sorted({sid for mapping in fold_maps.values() for sid in mapping.keys()})

    combined_count = 0
    skipped_count = 0
    manifest: List[Dict[str, object]] = []

    print(
        f"Found {len(all_subjects)} unique subjects across {len(fold_dirs)} folds "
        f"for input_format={args.input_format}."
    )
    for sid in tqdm(all_subjects, desc="Combining subjects", unit="subject"):
        fold_entries: List[Tuple[str, str, Optional[str]]] = []
        for fold_name in sorted(fold_dirs.keys()):
            map_path = fold_maps[fold_name].get(sid)
            if map_path is None:
                continue
            nii_path = fold_ref_nii.get(fold_name, {}).get(sid)
            fold_entries.append((fold_name, map_path, nii_path))

        if len(fold_entries) < max(1, int(args.min_folds)):
            skipped_count += 1
            manifest.append(
                {
                    "subject_id": sid,
                    "status": "skipped",
                    "reason": f"available_folds={len(fold_entries)} < min_folds={args.min_folds}",
                }
            )
            continue

        maps: List[np.ndarray] = []
        shapes: List[Tuple[int, int, int]] = []
        load_error: Optional[str] = None
        orient_logs: List[Dict[str, object]] = []
        template_nii_img: Optional[nib.Nifti1Image] = None
        template_nii_suffix: Optional[str] = None
        for fold_name, map_path, nii_path in fold_entries:
            try:
                if args.input_format == "npz":
                    arr = _load_pred_map_from_npz(map_path, npz_key=args.npz_key, channel=args.channel)
                    if not args.disable_npz_match_nii_orientation and nii_path is not None:
                        ref_data = np.asarray(nib.load(str(nii_path)).get_fdata(dtype=np.float32), dtype=np.float32)
                        arr, transform_desc, transform_score = _reorient_npz_to_reference(arr, ref_data)
                        orient_logs.append(
                            {
                                "fold": fold_name,
                                "nii_path": nii_path,
                                "transform": transform_desc,
                                "similarity": float(transform_score),
                            }
                        )
                        if template_nii_img is None:
                            template_nii_img = nib.load(str(nii_path))
                else:
                    nii_img = nib.load(str(map_path))
                    arr = np.asarray(nii_img.get_fdata(dtype=np.float32), dtype=np.float32)
                    arr = _to_probability_like(arr)
                    if template_nii_img is None:
                        template_nii_img = nii_img
                        template_nii_suffix = _infer_subject_nii_suffix(os.path.basename(map_path), sid)
                    else:
                        if not np.allclose(nii_img.affine, template_nii_img.affine, atol=1e-4):
                            raise ValueError("affine mismatch across fold NIfTI priors")
            except Exception as exc:
                load_error = f"{fold_name}: {exc}"
                break
            maps.append(arr)
            shapes.append(tuple(int(v) for v in arr.shape))

        if load_error is not None:
            skipped_count += 1
            manifest.append(
                {
                    "subject_id": sid,
                    "status": "skipped",
                    "reason": f"failed_loading_fold_map ({load_error})",
                }
            )
            continue

        if len(set(shapes)) != 1:
            skipped_count += 1
            manifest.append(
                {
                    "subject_id": sid,
                    "status": "skipped",
                    "reason": f"shape_mismatch_across_folds ({shapes})",
                }
            )
            continue

        stack = np.stack(maps, axis=0)
        if args.ensemble_method == "geometric":
            combined = _geometric_mean(stack, eps=max(float(args.epsilon), 1e-12))
        else:
            combined = _arithmetic_mean(stack)

        out_npz_path: Optional[str] = None
        out_nii_path: Optional[str] = None
        wrote_nii = False

        if args.input_format == "npz":
            out_npz_path = os.path.join(output_dir, f"{sid}.npz")
            np.savez_compressed(out_npz_path, prob=combined.astype(np.float32))

            if template_nii_img is None:
                template_nii_path = next((entry[2] for entry in fold_entries if entry[2] is not None), None)
                if template_nii_path is not None:
                    template_nii_img = nib.load(str(template_nii_path))

            out_nii_path = os.path.join(output_dir, f"{sid}.nii.gz")
            if template_nii_img is not None and tuple(template_nii_img.shape) == tuple(combined.shape):
                header = template_nii_img.header.copy()
                header.set_data_dtype(np.float32)
                out_img = nib.Nifti1Image(combined.astype(np.float32), template_nii_img.affine, header=header)
                nib.save(out_img, out_nii_path)
                wrote_nii = True
        else:
            out_name = f"{sid}.nii.gz"
            if template_nii_suffix is not None:
                out_name = f"{sid}{template_nii_suffix}"
            out_nii_path = os.path.join(output_dir, out_name)
            if template_nii_img is not None and tuple(template_nii_img.shape) == tuple(combined.shape):
                header = template_nii_img.header.copy()
                header.set_data_dtype(np.float32)
                out_img = nib.Nifti1Image(combined.astype(np.float32), template_nii_img.affine, header=header)
                nib.save(out_img, out_nii_path)
                wrote_nii = True

        combined_count += 1
        manifest.append(
            {
                "subject_id": sid,
                "status": "combined",
                "input_format": args.input_format,
                "ensemble_method": args.ensemble_method,
                "n_folds": len(fold_entries),
                "folds": [entry[0] for entry in fold_entries],
                "npz_orientation_alignment": orient_logs,
                "npz_output": out_npz_path,
                "nii_output": out_nii_path if wrote_nii else None,
            }
        )

    manifest_path = os.path.join(output_dir, "combine_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done combining fold predictions.")
    print(f"  Output directory: {output_dir}")
    print(f"  Combined subjects: {combined_count}")
    print(f"  Skipped subjects: {skipped_count}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
