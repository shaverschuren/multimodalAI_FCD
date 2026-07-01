"""
training.multimodal.multimodal_late_fusion.py

Combine MRI and EEG prediction maps into multimodal prediction maps via late
fusion: voxel-wise multiplication followed by min-max renormalisation to [0, 1].

Validation mode (default)
-------------------------
Fusion is performed per fold over the validation split only, so each subject
appears exactly once (as part of the fold where it was held out).

Test-set mode
-------------
When ``--test_set`` is enabled, the script first looks for direct ``fold_*``
directories under the MRI and EEG roots and recursively indexes test outputs
inside those folders. If no fold directories are present, it falls back to the
older combined-prediction layout (typically ``combined_preds``) for backwards
compatibility. Test subjects are resolved from ``k_fold_splits.json``
(``test_ids`` per fold) in this mode.

Input directory structures
--------------------------
MRI run directory (--mri_run_dir)
    Two layouts are supported:

    A) nnUNet / multimodal-UNet style (e.g. UNet_with_prior_training.py):
       <mri_run_dir>/<run_name>/pred_niftis/val/<sid>_pred_prob.nii.gz
       where <run_name> may contain a fold index such as ``multimodal_fold3_…``

    B) MRI-module NPZ style (eeg_spike_mil_regression_training.py):
       <mri_run_dir>/fold_<N>/<sid>.npz  (foreground probability stored under
       key ``prob``, ``probs``, ``prediction``, etc.)

    The fold index is inferred from the directory path (``fold_<N>`` token).

EEG run directory (--eeg_run_dir)
    <eeg_run_dir>/<any_sub_path>/val/<sid>_prior.nii.gz
    OR (flat):
    <eeg_run_dir>/<any_sub_path>/<sid>_prior.nii.gz

    All ``*_prior.nii.gz`` files found recursively are indexed.  When a
    subject appears more than once, the path that contains ``/val/`` in the
    directory tree is preferred; ties are broken alphabetically.

In test mode, EEG priors are read from direct ``fold_*`` directories when they
exist; otherwise they are read from ``<eeg_run_dir>/<eeg_combined_subdir>`` if
that directory exists (default: ``combined_preds``), otherwise from
``<eeg_run_dir>`` directly.

Output directory structure (--output_dir)
-----------------------------------------
The fused maps are written so that they are usable by all three analysis
scripts without modification:

    <output_dir>/
        latefusion_fold<N>_<YYYYMMDD-HHMMSS>/
            pred_niftis/
                val/
                    <sid>_pred_prob.nii.gz

This mirrors the ``multimodal_fold…/pred_niftis/val/`` layout expected by
``plot_multimodal_nii_maps.py`` and ``evaluate_maps.py`` (flat recursive
discovery).  A run-level ``fusion_meta.json`` is also written that records
the input paths, fold assignments and per-subject status.

Usage
-----
python training/multimodal/multimodal_late_fusion.py \\
    --mri_run_dir  /path/to/mri/runs \\
    --eeg_run_dir  /path/to/eeg/runs \\
    --fold_json    /path/to/k_fold_splits.json \\
    --output_dir   /path/to/late_fusion_output \\
    [--mri_npz_key prob] \\
    [--folds 0 1 2 3 4]

Author: Sjors Verschuren / Copilot
Date: June 2026
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import warnings
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from util.config import get_data_root


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MRI_PRED_SUFFIX = ".nii.gz"
EEG_PRIOR_SUFFIX = "_prior.nii.gz"
SUBJECT_ID_REGEX = re.compile(r"(RESP\d+)", re.IGNORECASE)

_data_root = get_data_root()
MRI_RUN_DIR_DEFAULT = str(_data_root / "results" / "runs" / "mri") if _data_root else None
EEG_RUN_DIR_DEFAULT = str(_data_root / "results" / "runs" / "eeg_new") if _data_root else None
OUTPUT_DIR_DEFAULT = str(_data_root / "results" / "runs" / "multimodal_late_fusion_new") if _data_root else None
FOLD_JSON_DEFAULT = str(_data_root / "preprocessing" / "k_fold_splits.json") if _data_root else None
FOLDS_DEFAULT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EEG_PRIOR_SUFFIXES_DEFAULT = "_deconv_prior.nii.gz,_prior.nii.gz"

# NPZ key candidates in priority order (mirrors evaluate_maps.py)
NPZ_KEYS_PRIORITY: Tuple[str, ...] = (
    "probabilities",
    "prob",
    "probs",
    "prediction",
    "pred",
    "softmax",
    "map",
    "prob_map",
    "foreground",
    "logits",
    "arr_0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subject_id(name: str) -> Optional[str]:
    match = SUBJECT_ID_REGEX.search(str(name))
    return match.group(1).upper() if match else None


def _extract_fold_name_from_path(path: str) -> Optional[str]:
    for part in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"fold_\d+", part):
            return part
    return None


def _load_fold_splits(json_path: str) -> Dict[str, List[str]]:
    """Return {fold_name: [val_subject_id, …]} from k_fold_splits.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fold_payload = payload["folds"]

    result: Dict[str, List[str]] = {}
    for fold_name, fold_data in fold_payload.items():
        if not isinstance(fold_data, dict):
            continue
        val_ids = fold_data.get("val_ids", [])
        if isinstance(val_ids, list):
            result[str(fold_name)] = [str(v).upper() for v in val_ids]
    return result


def _load_fold_test_splits(json_path: str) -> Dict[str, List[str]]:
    """Return {fold_name: [test_subject_id, …]} from k_fold_splits.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fold_payload = payload["folds"]

    result: Dict[str, List[str]] = {}
    for fold_name, fold_data in fold_payload.items():
        if not isinstance(fold_data, dict):
            continue
        test_ids = fold_data.get("test_ids", [])
        if isinstance(test_ids, list):
            result[str(fold_name)] = [str(v).upper() for v in test_ids]
    return result


# ---------------------------------------------------------------------------
# MRI prediction discovery
# ---------------------------------------------------------------------------


def _find_fold_dirs(pred_dir: str) -> Dict[str, str]:
    """Return {fold_name: path} for direct children matching ``fold_\\d+``."""
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


def _path_parts_lower(path: str) -> List[str]:
    return [part.lower() for part in os.path.normpath(path).split(os.sep)]


def _is_test_split_path(path: str) -> bool:
    parts = _path_parts_lower(path)
    return "test" in parts and "train" not in parts and "val" not in parts


def _index_recursive_by_suffixes(
    directory: str,
    suffixes: List[str],
    require_test_path: bool = False,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for suffix in [s.strip() for s in suffixes if s.strip()]:
        pattern = os.path.join(directory, "**", f"*{suffix}")
        for path in sorted(glob(pattern, recursive=True)):
            if require_test_path and not _is_test_split_path(path):
                continue
            sid = _extract_subject_id(os.path.basename(path))
            if sid and sid not in mapping:
                mapping[sid] = path
    return mapping


def _index_dir_nii(directory: str) -> Dict[str, str]:
    """Non-recursively index NIfTI prediction files in *directory*."""
    mapping: Dict[str, str] = {}
    for suffix in (".nii.gz", ".nii"):
        for path in sorted(glob(os.path.join(directory, f"*{suffix}"))):
            sid = _extract_subject_id(os.path.basename(path))
            if sid and sid not in mapping:
                mapping[sid] = path
    return mapping


def _index_dir_npz(directory: str) -> Dict[str, str]:
    """Non-recursively index NPZ prediction files in *directory*."""
    mapping: Dict[str, str] = {}
    for path in sorted(glob(os.path.join(directory, "*.npz"))):
        sid = _extract_subject_id(os.path.basename(path))
        if sid and sid not in mapping:
            mapping[sid] = path
    return mapping


def _index_dir_nii_by_suffixes(directory: str, suffixes: List[str]) -> Dict[str, str]:
    """Recursively index NIfTI files ending in one of *suffixes* under *directory*."""
    mapping: Dict[str, str] = {}
    for suffix in [s.strip() for s in suffixes if s.strip()]:
        pattern = os.path.join(directory, "**", f"*{suffix}")
        for path in sorted(glob(pattern, recursive=True)):
            sid = _extract_subject_id(os.path.basename(path))
            if sid and sid not in mapping:
                mapping[sid] = path
    return mapping


def _index_mri_predictions(
    root_dir: str,
    fold_splits: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
    """
    Index MRI prediction maps, mirroring the fold-aware logic of
    ``plot_mri_module_npz_maps.py``.

    Fold mode (preferred)
    ---------------------
    When ``fold_\\d+`` subdirectories are found directly under *root_dir*,
    files are indexed **non-recursively** within each fold directory and
    filtered to that fold's validation subjects (from *fold_splits*).
    NPZ takes priority over NIfTI so that the raw array is used for fusion;
    the paired NIfTI (if present in the same dir) is tracked separately to
    supply the affine and orientation reference for NPZ alignment.

    Flat / recursive fallback
    -------------------------
    When no fold directories are found, all NIfTI and NPZ files are
    collected recursively; paths containing ``train`` (without ``val``) are
    skipped.

    Returns
    -------
    pred_map : Dict[str, str]
        sid -> prediction path (NPZ preferred over NIfTI).
    nii_map : Dict[str, str]
        sid -> paired NIfTI path (for NPZ orientation; empty when only NIfTI preds exist).
    fold_mode : bool
        True when fold directories were detected.
    """
    fold_dirs = _find_fold_dirs(root_dir)

    if fold_dirs:
        pred_map: Dict[str, str] = {}
        nii_map: Dict[str, str] = {}
        for fold_name, fold_dir in sorted(fold_dirs.items()):
            val_ids = set(fold_splits.get(fold_name, []))
            if not val_ids:
                tqdm.write(
                    f"  [mri-index] {fold_name}: no val_ids in fold_splits — skipping fold dir."
                )
                continue

            fold_nii = _index_recursive_by_suffixes(
                fold_dir,
                [MRI_PRED_SUFFIX],
                require_test_path=False,
            )
            fold_npz = _index_recursive_by_suffixes(
                fold_dir,
                [".npz"],
                require_test_path=False,
            )
            # NPZ preferred as prediction; NIfTI tracked as orientation reference.
            fold_pred: Dict[str, str] = {**fold_nii, **fold_npz}  # NPZ wins

            for sid in sorted(val_ids & set(fold_pred)):
                if sid not in pred_map:
                    pred_map[sid] = fold_pred[sid]
                    if sid in fold_nii:
                        nii_map[sid] = fold_nii[sid]
                else:
                    tqdm.write(
                        f"  [mri-index] Duplicate val subject '{sid}' across folds — "
                        f"keeping first match."
                    )
        return pred_map, nii_map, True

    # Flat / recursive fallback.
    all_val_ids: set = {sid for ids in fold_splits.values() for sid in ids}
    pred_map = {}
    nii_map = {}

    # Collect NPZ first (preferred), then NIfTI as fallback.
    for pattern in [
        os.path.join(root_dir, "**", "*.npz"),
        os.path.join(root_dir, "**", f"*{MRI_PRED_SUFFIX}"),
    ]:
        for path in sorted(glob(pattern, recursive=True)):
            parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
            if "train" in parts and "val" not in parts:
                continue
            sid = _extract_subject_id(os.path.basename(path))
            if sid and sid in all_val_ids and sid not in pred_map:
                pred_map[sid] = path

    # Collect NIfTI paths as orientation references for NPZ subjects.
    for path in sorted(glob(os.path.join(root_dir, "**", f"*{MRI_PRED_SUFFIX}"), recursive=True)):
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        if "train" in parts and "val" not in parts:
            continue
        sid = _extract_subject_id(os.path.basename(path))
        if sid and sid not in nii_map:
            nii_map[sid] = path

    return pred_map, nii_map, False


def _index_test_mri_predictions(
    root_dir: str,
    combined_subdir: str,
    fold_test_splits: Dict[str, List[str]],
) -> Tuple[Dict[str, str], Dict[str, str], str, bool]:
    """
    Index MRI test predictions either from direct fold_* directories or from
    a combined-predictions layout.

    Returns (pred_map, nii_map, source_dir, fold_mode).
    """
    fold_dirs = _find_fold_dirs(root_dir)

    if fold_dirs:
        pred_map: Dict[str, str] = {}
        nii_map: Dict[str, str] = {}
        for fold_name, fold_dir in sorted(fold_dirs.items()):
            test_ids = set(fold_test_splits.get(fold_name, []))
            if not test_ids:
                tqdm.write(
                    f"  [mri-test] {fold_name}: no test_ids in fold_splits — skipping fold dir."
                )
                continue

            fold_nii = _index_recursive_by_suffixes(
                fold_dir,
                [MRI_PRED_SUFFIX],
                require_test_path=False,
            )
            fold_npz = _index_recursive_by_suffixes(
                fold_dir,
                [".npz"],
                require_test_path=False,
            )
            fold_pred: Dict[str, str] = {**fold_nii, **fold_npz}

            for sid in sorted(test_ids & set(fold_pred.keys())):
                path = fold_pred[sid]
                if sid in pred_map:
                    tqdm.write(
                        f"  [mri-test] Duplicate subject '{sid}' across folds — keeping first match."
                    )
                    continue
                pred_map[sid] = path
                if sid in fold_nii:
                    nii_map[sid] = fold_nii[sid]

        return pred_map, nii_map, root_dir, True

    preferred_dir = os.path.join(root_dir, combined_subdir)
    source_dir = preferred_dir if os.path.isdir(preferred_dir) else root_dir
    all_test_ids = {sid for ids in fold_test_splits.values() for sid in ids}
    nii_all = _index_recursive_by_suffixes(source_dir, [MRI_PRED_SUFFIX])
    npz_all = _index_recursive_by_suffixes(source_dir, [".npz"])
    nii_map = {sid: path for sid, path in nii_all.items() if sid in all_test_ids}
    npz_map = {sid: path for sid, path in npz_all.items() if sid in all_test_ids}
    pred_map = {**nii_map, **npz_map}
    return pred_map, nii_map, source_dir, False


def _score_map_similarity(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Mean element-wise product of two probability-like maps (higher = better match)."""
    c = np.asarray(candidate, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    valid = np.isfinite(c) & np.isfinite(r)
    if not np.any(valid):
        return float("-inf")
    return float(np.mean(c[valid] * r[valid]))


def _reorient_npz_to_reference(
    map3d: np.ndarray,
    ref3d: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    Brute-force search over all axis permutations and flips to find the
    orientation of *map3d* that best matches *ref3d*.

    Mirrors ``_reorient_npz_to_reference`` in plot_mri_module_npz_maps.py.
    Returns (reoriented_map, transform_description).
    """
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
                best_desc = f"perm={perm}, flips={tuple(flip_axes)}, score={best_score:.6f}"

    return best_map, best_desc


def _map_npz_to_nii_space(
    map3d: np.ndarray,
    map_affine: np.ndarray,
    target_nii: "nib.Nifti1Image",
) -> np.ndarray:
    """Resample *map3d* (with *map_affine*) into the space of *target_nii*."""
    from nibabel.processing import resample_from_to  # type: ignore

    src_nii = nib.Nifti1Image(np.asarray(map3d, dtype=np.float32), map_affine)
    resampled = resample_from_to(src_nii, target_nii, order=1)
    return np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)


def _load_mri_map(
    path: str,
    npz_key: Optional[str] = None,
    paired_nii_path: Optional[str] = None,
    sid: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load MRI prediction map from NIfTI or NPZ.

    For NPZ files, applies the same two-step alignment used by
    ``plot_mri_module_npz_maps.py``:
      1. Axis-permutation + flip search against the paired NIfTI data
         (``_reorient_npz_to_reference``).
      2. Affine-based resampling into the paired NIfTI space
         (``_map_npz_to_nii_space``).

    *paired_nii_path* is required for NPZ alignment. If absent, the raw NPZ
    array is returned with its stored affine (or None), which may be
    misaligned.

    Returns (map_float32, affine_or_None).
    """
    name = os.path.basename(path).lower()

    if name.endswith(".nii.gz") or name.endswith(".nii"):
        img = nib.load(path)
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        affine = np.asarray(img.affine, dtype=np.float64)
        return _pick_3d_map(data), affine

    if name.endswith(".npz"):
        keys_to_try = []
        if npz_key:
            keys_to_try.append(npz_key)
        keys_to_try.extend(NPZ_KEYS_PRIORITY)

        with np.load(path, allow_pickle=True) as npz:
            available = list(npz.keys())
            arr = None
            seen: set = set()
            for k in [k for k in keys_to_try if not (k in seen or seen.add(k))]:
                if k in npz:
                    candidate = np.asarray(npz[k])
                    if candidate.ndim >= 3:
                        arr = candidate
                        break
            if arr is None:
                raise ValueError(
                    f"No suitable 3-D array in {path}. Available keys: {available}"
                )
            stored_affine = npz["affine"] if "affine" in npz else None
            if stored_affine is not None:
                stored_affine = np.asarray(stored_affine, dtype=np.float64)

        map3d = _pick_3d_map(arr).astype(np.float32)

        if paired_nii_path is None:
            tqdm.write(
                f"  [align] {sid}: no paired NIfTI found — using raw NPZ array "
                "(may be misaligned)."
            )
            return map3d, stored_affine

        pred_nii_img = nib.load(paired_nii_path)
        pred_nii_data = np.asarray(pred_nii_img.get_fdata(dtype=np.float32), dtype=np.float32)

        # Step 1: fix axis/flip layout mismatch.
        map3d, transform_desc = _reorient_npz_to_reference(map3d, pred_nii_data)
        tqdm.write(f"  [orient] {sid}: {transform_desc}")

        # Step 2: resample into NIfTI space using the stored affine.
        affine_for_resample = stored_affine if stored_affine is not None else pred_nii_img.affine
        map3d = _map_npz_to_nii_space(map3d, affine_for_resample, pred_nii_img)

        return map3d, np.asarray(pred_nii_img.affine, dtype=np.float64)

    raise ValueError(f"Unsupported MRI map format: {path}")


def _pick_3d_map(arr: np.ndarray) -> np.ndarray:
    """Select the foreground probability volume from a multi-dim array."""
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # [C, D, H, W] with small C: take foreground channel (index 1 or 0)
        if arr.shape[0] <= 4 and all(s > 4 for s in arr.shape[1:]):
            return arr[1] if arr.shape[0] > 1 else arr[0]
        # [D, H, W, C] with small C
        if arr.shape[-1] <= 4 and all(s > 4 for s in arr.shape[:-1]):
            return arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
        axis = int(np.argmin(arr.shape))
        return arr.take(indices=min(1, arr.shape[axis] - 1), axis=axis)
    if arr.ndim == 5:
        return _pick_3d_map(arr[0])
    raise ValueError(f"Unsupported array shape: {arr.shape}")


# ---------------------------------------------------------------------------
# EEG prior discovery
# ---------------------------------------------------------------------------


def _index_eeg_priors(root_dir: str) -> Dict[str, str]:
    """
    Recursively find *_prior.nii.gz files under root_dir.

    When a subject appears in both a ``val`` sub-path and a non-val path,
    the val path is preferred (mirrors training / analysis conventions).
    """
    pattern = os.path.join(root_dir, "**", f"*{EEG_PRIOR_SUFFIX}")
    candidates: Dict[str, List[str]] = {}  # sid -> [paths]

    for path in sorted(glob(pattern, recursive=True)):
        # Skip training split files.
        parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
        if "train" in parts and "val" not in parts:
            continue
        sid = _extract_subject_id(os.path.basename(path))
        if sid:
            candidates.setdefault(sid, []).append(path)

    mapping: Dict[str, str] = {}
    for sid, paths in candidates.items():
        # Prefer paths containing a /val/ directory component.
        val_paths = [p for p in paths if "val" in [x.lower() for x in os.path.normpath(p).split(os.sep)]]
        mapping[sid] = (val_paths or paths)[0]

    return mapping


def _index_test_eeg_priors(
    root_dir: str,
    combined_subdir: str,
    prior_suffixes: List[str],
    fold_test_splits: Dict[str, List[str]],
) -> Tuple[Dict[str, str], str, bool]:
    """
    Index EEG priors for test mode from direct fold_* directories when present,
    otherwise from a combined-prediction layout.
    """
    fold_dirs = _find_fold_dirs(root_dir)
    if fold_dirs:
        mapping: Dict[str, str] = {}
        for fold_name, fold_dir in sorted(fold_dirs.items()):
            test_ids = set(fold_test_splits.get(fold_name, []))
            if not test_ids:
                tqdm.write(
                    f"  [eeg-test] {fold_name}: no test_ids in fold_splits — skipping fold dir."
                )
                continue

            fold_map = _index_recursive_by_suffixes(
                fold_dir,
                prior_suffixes,
                require_test_path=False,
            )
            for sid in sorted(test_ids & set(fold_map.keys())):
                path = fold_map[sid]
                if sid in mapping:
                    tqdm.write(
                        f"  [eeg-test] Duplicate subject '{sid}' across folds — keeping first match."
                    )
                    continue
                mapping[sid] = path
        return mapping, root_dir, True

    preferred_dir = os.path.join(root_dir, combined_subdir)
    source_dir = preferred_dir if os.path.isdir(preferred_dir) else root_dir
    all_test_ids = {sid for ids in fold_test_splits.values() for sid in ids}
    mapping_all = _index_dir_nii_by_suffixes(source_dir, prior_suffixes)
    mapping = {sid: path for sid, path in mapping_all.items() if sid in all_test_ids}
    return mapping, source_dir, False


def _load_eeg_map(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load EEG prior map from NIfTI. Returns (map_float32, affine)."""
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    return data, affine


# ---------------------------------------------------------------------------
# Spatial resampling
# ---------------------------------------------------------------------------


def _resample_to_target(
    src_data: np.ndarray,
    src_affine: Optional[np.ndarray],
    tgt_data: np.ndarray,
    tgt_affine: Optional[np.ndarray],
) -> np.ndarray:
    """
    Resample src_data into the space of tgt_data using linear interpolation.

    Falls back to simple shape-based check if no affines are available; in
    that case raises if shapes differ.
    """
    if src_data.shape == tgt_data.shape:
        return src_data

    if src_affine is None or tgt_affine is None:
        raise ValueError(
            f"Shape mismatch ({src_data.shape} vs {tgt_data.shape}) and "
            "at least one map lacks an affine — cannot resample."
        )

    from nibabel.processing import resample_from_to  # type: ignore

    src_nii = nib.Nifti1Image(src_data.astype(np.float32), src_affine)
    tgt_nii = nib.Nifti1Image(tgt_data.astype(np.float32), tgt_affine)
    resampled = resample_from_to(src_nii, tgt_nii, order=1)
    return np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _to_unit_range(arr: np.ndarray) -> np.ndarray:
    """Map array values to [0, 1] using global min-max normalisation."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    vmin = float(arr[finite].min())
    vmax = float(arr[finite].max())
    if vmax <= vmin:
        return np.zeros_like(arr)
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def fuse_maps(
    mri_map: np.ndarray,
    eeg_map: np.ndarray,
) -> np.ndarray:
    """
    Late fusion: voxel-wise geometric mean of two probability maps

    Both maps are first normalised to [0, 1] TODO: Maybe drop this?
    """
    mri_norm = _to_unit_range(mri_map)
    eeg_norm = _to_unit_range(eeg_map)
    geometric_mean = np.sqrt(mri_norm * eeg_norm)  # Geometric mean fusion
    return _to_unit_range(geometric_mean)


# ---------------------------------------------------------------------------
# Per-fold fusion
# ---------------------------------------------------------------------------


def run_fold(
    fold_name: str,
    val_ids: List[str],
    mri_map: Dict[str, str],
    mri_nii_map: Dict[str, str],
    eeg_map: Dict[str, str],
    output_fold_dir: str,
    npz_key: Optional[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Fuse MRI and EEG maps for all validation subjects of one fold.

    Returns (saved_ids, skipped_mri, skipped_eeg).
    """
    val_dir = os.path.join(output_fold_dir, "pred_niftis", "val")
    os.makedirs(val_dir, exist_ok=True)

    saved: List[str] = []
    skip_mri: List[str] = []
    skip_eeg: List[str] = []

    for sid in tqdm(sorted(val_ids), desc=fold_name, leave=False):
        mri_path = mri_map.get(sid)
        eeg_path = eeg_map.get(sid)

        if mri_path is None:
            skip_mri.append(sid)
            continue
        if eeg_path is None:
            skip_eeg.append(sid)
            continue

        paired_nii = mri_nii_map.get(sid)

        try:
            mri_data, mri_affine = _load_mri_map(
                mri_path, npz_key=npz_key, paired_nii_path=paired_nii, sid=sid
            )
        except Exception as exc:
            tqdm.write(f"  [{fold_name}] Cannot load MRI map for {sid}: {exc}")
            skip_mri.append(sid)
            continue

        try:
            eeg_data, eeg_affine = _load_eeg_map(eeg_path)
        except Exception as exc:
            tqdm.write(f"  [{fold_name}] Cannot load EEG map for {sid}: {exc}")
            skip_eeg.append(sid)
            continue

        # Align EEG map to MRI space if shapes differ.
        if eeg_data.shape != mri_data.shape:
            try:
                eeg_data = _resample_to_target(eeg_data, eeg_affine, mri_data, mri_affine)
            except Exception as exc:
                tqdm.write(f"  [{fold_name}] Resampling failed for {sid}: {exc}")
                skip_eeg.append(sid)
                continue

        fused = fuse_maps(mri_data, eeg_data)

        # Use MRI affine for output (authoritative spatial reference).
        out_affine = mri_affine if mri_affine is not None else eeg_affine
        if out_affine is None:
            out_affine = np.eye(4)

        out_path = os.path.join(val_dir, f"{sid}{MRI_PRED_SUFFIX}")
        nib.save(nib.Nifti1Image(fused, out_affine), out_path)
        saved.append(sid)

    return saved, skip_mri, skip_eeg


def run_test_set(
    subject_ids: List[str],
    mri_map: Dict[str, str],
    mri_nii_map: Dict[str, str],
    eeg_map: Dict[str, str],
    output_dir: str,
    npz_key: Optional[str],
    subject_to_fold: Optional[Dict[str, str]] = None,
    fallback_output_subdir: str = "combined_preds",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Fuse MRI and EEG maps for an explicit subject list in test-set mode.

    Outputs are written under ``<output_dir>/<fold_name>/test`` when
    *subject_to_fold* provides a fold for the subject. If no fold is known,
    outputs fall back to ``<output_dir>/<fallback_output_subdir>``.
    """
    os.makedirs(output_dir, exist_ok=True)

    saved: List[str] = []
    skip_mri: List[str] = []
    skip_eeg: List[str] = []

    for sid in tqdm(sorted(subject_ids), desc="test_set", leave=False):
        mri_path = mri_map.get(sid)
        eeg_path = eeg_map.get(sid)

        if mri_path is None:
            skip_mri.append(sid)
            continue
        if eeg_path is None:
            skip_eeg.append(sid)
            continue

        paired_nii = mri_nii_map.get(sid)

        try:
            mri_data, mri_affine = _load_mri_map(
                mri_path, npz_key=npz_key, paired_nii_path=paired_nii, sid=sid
            )
        except Exception as exc:
            tqdm.write(f"  [test_set] Cannot load MRI map for {sid}: {exc}")
            skip_mri.append(sid)
            continue

        try:
            eeg_data, eeg_affine = _load_eeg_map(eeg_path)
        except Exception as exc:
            tqdm.write(f"  [test_set] Cannot load EEG map for {sid}: {exc}")
            skip_eeg.append(sid)
            continue

        if eeg_data.shape != mri_data.shape:
            try:
                eeg_data = _resample_to_target(eeg_data, eeg_affine, mri_data, mri_affine)
            except Exception as exc:
                tqdm.write(f"  [test_set] Resampling failed for {sid}: {exc}")
                skip_eeg.append(sid)
                continue

        fused = fuse_maps(mri_data, eeg_data)
        out_affine = mri_affine if mri_affine is not None else eeg_affine
        if out_affine is None:
            out_affine = np.eye(4)

        fold_name = None if subject_to_fold is None else subject_to_fold.get(sid)
        if fold_name is not None:
            out_dir = os.path.join(output_dir, fold_name, "test")
        else:
            out_dir = os.path.join(output_dir, fallback_output_subdir)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{sid}{MRI_PRED_SUFFIX}")
        nib.save(nib.Nifti1Image(fused, out_affine), out_path)
        saved.append(sid)

    return saved, skip_mri, skip_eeg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Late-fusion of MRI and EEG prediction maps: "
            "voxel-wise multiplication + min-max renormalisation."
        )
    )
    parser.add_argument(
        "--mri_run_dir",
        default=MRI_RUN_DIR_DEFAULT,
        help=(
            "Root directory of the MRI runs. "
            "Expected to contain *_pred_prob.nii.gz or *.npz prediction files, "
            "discoverable recursively."
        ),
    )
    parser.add_argument(
        "--eeg_run_dir",
        default=EEG_RUN_DIR_DEFAULT,
        help=(
            "Root directory of the EEG runs. "
            "Expected to contain *_prior.nii.gz files discoverable recursively."
        ),
    )
    parser.add_argument(
        "--fold_json",
        default=FOLD_JSON_DEFAULT,
        help="Path to k_fold_splits.json used during training.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory to write late-fusion NIfTI maps.",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=FOLDS_DEFAULT,
        help="Fold indices to process (e.g. --folds 0 1 2). Default: all folds in fold_json.",
    )
    parser.add_argument(
        "--mri_npz_key",
        default=None,
        help=(
            "NPZ key for the MRI foreground probability array "
            "(only used when MRI maps are .npz files)."
        ),
    )
    parser.add_argument(
        "--run_tag",
        default=None,
        help=(
            "Optional tag inserted into the run directory name, "
            "e.g. 'mri_v2_eeg_mh'. Defaults to 'latefusion'."
        ),
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help=(
            "Enable test-set mode: fuse combined MRI and EEG predictions directly "
            "using fold_json test_ids."
        ),
    )
    parser.add_argument(
        "--mri_combined_subdir",
        default="combined_preds",
        help=(
            "Subdirectory under --mri_run_dir containing combined test maps. "
            "Used only when --test_set is enabled."
        ),
    )
    parser.add_argument(
        "--eeg_combined_subdir",
        default="combined_preds",
        help=(
            "Subdirectory under --eeg_run_dir containing combined test priors. "
            "Used only when --test_set is enabled."
        ),
    )
    parser.add_argument(
        "--eeg_prior_suffixes",
        default=EEG_PRIOR_SUFFIXES_DEFAULT,
        help=(
            "Comma-separated EEG prior filename suffixes used for indexing in test mode. "
            "Default: '_deconv_prior.nii.gz,_prior.nii.gz'."
        ),
    )
    parser.add_argument(
        "--test_output_subdir",
        default="combined_preds",
        help=(
            "Legacy fallback output subdirectory used only for test subjects "
            "without a resolvable fold_<idx> token in their source path."
        ),
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isdir(args.mri_run_dir):
        sys.exit(f"ERROR: --mri_run_dir does not exist: {args.mri_run_dir}")
    if not os.path.isdir(args.eeg_run_dir):
        sys.exit(f"ERROR: --eeg_run_dir does not exist: {args.eeg_run_dir}")
    if not os.path.isfile(args.fold_json):
        sys.exit(f"ERROR: --fold_json does not exist: {args.fold_json}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_set:
        # --------------------------------------------------------------
        # Test-set mode: fuse directly from test predictions
        # --------------------------------------------------------------
        fold_test_splits = _load_fold_test_splits(args.fold_json)
        if not fold_test_splits:
            sys.exit(f"ERROR: No valid fold test splits found in {args.fold_json}")

        requested_folds = args.folds
        if requested_folds is not None:
            fold_test_splits = {
                k: v for k, v in fold_test_splits.items()
                if any(k == f"fold_{i}" for i in requested_folds)
            }
            if not fold_test_splits:
                sys.exit(
                    f"ERROR: None of the requested folds {requested_folds} found in fold_json."
                )

        subject_to_fold: Dict[str, str] = {}
        for fold_name, sids in fold_test_splits.items():
            for sid in sids:
                subject_to_fold[sid] = fold_name

        requested_test_ids = set(subject_to_fold.keys())
        tqdm.write(f"Using {len(requested_test_ids)} test subjects from fold_json test_ids.")

        prior_suffixes = [s.strip() for s in str(args.eeg_prior_suffixes).split(",") if s.strip()]

        tqdm.write("Indexing MRI test maps...")
        mri_index, mri_nii_index, mri_source_dir, mri_fold_mode = _index_test_mri_predictions(
            args.mri_run_dir,
            args.mri_combined_subdir,
            fold_test_splits,
        )
        npz_count = sum(1 for p in mri_index.values() if p.lower().endswith(".npz"))
        nii_count = len(mri_index) - npz_count
        if mri_fold_mode:
            tqdm.write(
                f"  Found {len(mri_index)} MRI test maps ({nii_count} NIfTI, {npz_count} NPZ) "
                f"from direct fold_* directories; {len(mri_nii_index)} paired NIfTI refs."
            )
        else:
            tqdm.write(
                f"  Found {len(mri_index)} MRI test maps ({nii_count} NIfTI, {npz_count} NPZ) "
                f"from {mri_source_dir}; {len(mri_nii_index)} paired NIfTI refs."
            )

        tqdm.write("Indexing EEG test prior maps...")
        eeg_index, eeg_source_dir, eeg_fold_mode = _index_test_eeg_priors(
            args.eeg_run_dir,
            args.eeg_combined_subdir,
            prior_suffixes,
            fold_test_splits,
        )
        if eeg_fold_mode:
            tqdm.write(f"  Found {len(eeg_index)} EEG test priors from direct fold_* directories.")
        else:
            tqdm.write(f"  Found {len(eeg_index)} EEG test priors from {eeg_source_dir}.")

        shared_ids = sorted(set(mri_index.keys()) & set(eeg_index.keys()) & requested_test_ids)
        if not shared_ids:
            sys.exit("ERROR: No overlapping subject IDs between MRI and EEG test maps.")

        # Keep fold assignment from JSON as source of truth; path-derived fold is fallback.
        if mri_fold_mode or eeg_fold_mode:
            for sid in shared_ids:
                if sid in subject_to_fold:
                    continue
                fold_name = _extract_fold_name_from_path(mri_index.get(sid, ""))
                if fold_name is None:
                    fold_name = _extract_fold_name_from_path(eeg_index.get(sid, ""))
                if fold_name is not None:
                    subject_to_fold[sid] = fold_name

        test_out_dir = args.output_dir
        tqdm.write(f"\nTest-set fusion: {len(shared_ids)} overlapping subjects → {test_out_dir}")

        saved, skip_mri, skip_eeg = run_test_set(
            subject_ids=shared_ids,
            mri_map=mri_index,
            mri_nii_map=mri_nii_index,
            eeg_map=eeg_index,
            output_dir=test_out_dir,
            npz_key=args.mri_npz_key,
            subject_to_fold=subject_to_fold,
            fallback_output_subdir=args.test_output_subdir,
        )

        datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        meta = {
            "mode": "test_set",
            "mri_run_dir": str(args.mri_run_dir),
            "eeg_run_dir": str(args.eeg_run_dir),
            "mri_source_dir": mri_source_dir,
            "eeg_source_dir": eeg_source_dir,
            "output_dir": str(args.output_dir),
            "output_test_dir": test_out_dir,
            "test_output_subdir": str(args.test_output_subdir),
            "mri_combined_subdir": str(args.mri_combined_subdir),
            "eeg_combined_subdir": str(args.eeg_combined_subdir),
            "eeg_prior_suffixes": prior_suffixes,
            "created_at": datestr,
            "subject_to_fold": subject_to_fold,
            "overlap_subject_ids": shared_ids,
            "saved": saved,
            "skipped_no_mri": skip_mri,
            "skipped_no_eeg": skip_eeg,
        }
        meta_path = os.path.join(args.output_dir, f"fusion_meta_test_{datestr}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        tqdm.write(
            f"\n{'=' * 60}\n"
            f"Late fusion complete (test_set mode).\n"
            f"  Total saved : {len(saved)}\n"
            f"  No MRI map  : {len(skip_mri)}\n"
            f"  No EEG map  : {len(skip_eeg)}\n"
            f"  Metadata    : {meta_path}\n"
            f"  Output      : {test_out_dir}\n"
            f"{'=' * 60}"
        )
        return

    # ------------------------------------------------------------------
    # Validation-fold mode (original behavior)
    # ------------------------------------------------------------------
    fold_splits = _load_fold_splits(args.fold_json)
    if not fold_splits:
        sys.exit(f"ERROR: No valid fold splits found in {args.fold_json}")

    requested_folds = args.folds
    if requested_folds is not None:
        fold_splits = {
            k: v for k, v in fold_splits.items()
            if any(k == f"fold_{i}" for i in requested_folds)
        }
        if not fold_splits:
            sys.exit(
                f"ERROR: None of the requested folds {requested_folds} found in fold_json."
            )

    tqdm.write("Indexing MRI prediction maps...")
    mri_index, mri_nii_index, mri_fold_mode = _index_mri_predictions(args.mri_run_dir, fold_splits)
    mode_label = "fold-dir mode" if mri_fold_mode else "recursive fallback mode"
    npz_count = sum(1 for p in mri_index.values() if p.lower().endswith(".npz"))
    nii_count = len(mri_index) - npz_count
    tqdm.write(
        f"  Found {len(mri_index)} MRI maps ({nii_count} NIfTI, {npz_count} NPZ) "
        f"[{mode_label}]; {len(mri_nii_index)} paired NIfTI orientation refs."
    )

    tqdm.write("Indexing EEG prior maps...")
    eeg_index = _index_eeg_priors(args.eeg_run_dir)
    tqdm.write(f"  Found {len(eeg_index)} EEG prior maps.")

    # ------------------------------------------------------------------
    # Run fusion per fold
    # ------------------------------------------------------------------
    run_tag = args.run_tag or "latefusion"
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_meta: Dict[str, object] = {
        "mri_run_dir": str(args.mri_run_dir),
        "eeg_run_dir": str(args.eeg_run_dir),
        "fold_json": str(args.fold_json),
        "output_dir": str(args.output_dir),
        "run_tag": run_tag,
        "created_at": datestr,
        "folds": {},
    }

    total_saved = 0
    total_skip_mri = 0
    total_skip_eeg = 0

    fold_items = sorted(fold_splits.items())
    fold_bar = tqdm(fold_items, desc="Folds", unit="fold")
    for fold_name, val_ids in fold_bar:
        # Extract fold index from fold_name (e.g. "fold_3" -> 3)
        fold_idx_m = re.search(r"(\d+)", fold_name)
        fold_idx = fold_idx_m.group(1) if fold_idx_m else fold_name

        run_dir_name = f"{run_tag}_fold{fold_idx}_{datestr}"
        output_fold_dir = os.path.join(args.output_dir, run_dir_name)

        fold_bar.set_description(f"Fold {fold_idx}")
        tqdm.write(f"\nFold {fold_idx}: {len(val_ids)} validation subjects → {run_dir_name}")

        saved, skip_mri, skip_eeg = run_fold(
            fold_name=fold_name,
            val_ids=val_ids,
            mri_map=mri_index,
            mri_nii_map=mri_nii_index,
            eeg_map=eeg_index,
            output_fold_dir=output_fold_dir,
            npz_key=args.mri_npz_key,
        )

        total_saved += len(saved)
        total_skip_mri += len(skip_mri)
        total_skip_eeg += len(skip_eeg)

        fold_meta = {
            "val_ids": val_ids,
            "saved": saved,
            "skipped_no_mri": skip_mri,
            "skipped_no_eeg": skip_eeg,
            "output_dir": output_fold_dir,
        }
        all_meta["folds"][fold_name] = fold_meta  # type: ignore[index]

        tqdm.write(
            f"  Saved: {len(saved)}  |  "
            f"Skipped (no MRI): {len(skip_mri)}  |  "
            f"Skipped (no EEG): {len(skip_eeg)}"
        )
        if skip_mri:
            tqdm.write(f"  Missing MRI: {', '.join(sorted(skip_mri))}")
        if skip_eeg:
            tqdm.write(f"  Missing EEG: {', '.join(sorted(skip_eeg))}")

    # ------------------------------------------------------------------
    # Write run-level metadata
    # ------------------------------------------------------------------
    meta_path = os.path.join(args.output_dir, f"fusion_meta_{datestr}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    tqdm.write(
        f"\n{'=' * 60}\n"
        f"Late fusion complete.\n"
        f"  Total saved : {total_saved}\n"
        f"  No MRI map  : {total_skip_mri}\n"
        f"  No EEG map  : {total_skip_eeg}\n"
        f"  Metadata    : {meta_path}\n"
        f"  Output      : {args.output_dir}\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
