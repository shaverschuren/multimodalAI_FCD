"""
Evaluate 3D soft FCD prediction maps against binary ground-truth masks.

This script evaluates 3D soft FCD prediction maps against binary ground-truth masks at voxel,
cluster, and subject levels. Fixed metrics use threshold 0.5 by default. Cluster detection uses
predicted connected components and GT connected components, with bounding-box Dice > 0.22 as the
default detection criterion. Pinpointing is defined as a predicted cluster center of mass falling
inside the GT mask. The script also exports PR curves and detection-rate versus
false-positive-cluster-burden data for later plotting.

Notes on metric definitions
----------------------------
- Voxel Dice: overlap at the voxel level between binarised prediction and GT mask.
  High voxel Dice means the predicted region closely matches the GT lesion shape.
- Cluster bounding-box DSC (boxDSC): Dice Similarity Coefficient computed on the 3-D
  bounding boxes of a predicted cluster and a GT component. Used as a coarse detection
  criterion inspired by Kersting et al., 2025. (https://doi.org/10.1111/epi.18240).
  A boxDSC > 0.22 (default) means the boxes overlap substantially enough to count as a detection. 
- Pinpointing: a predicted cluster is a pinpointing if its center of mass falls anywhere
  inside the GT mask. Also inspired by Kersting et al., 2025. (https://doi.org/10.1111/epi.18240)
- Subject-level detection rate: fraction of FCD cases (non-empty GT) for which at least
  one predicted cluster satisfies the detection (or pinpointing) criterion for any GT
  component.

Usage
-----
python evaluate/evaluate_maps.py \\
    --pred_dir /path/to/predictions \\
    --gt_dir /path/to/gt_masks \\
    --out_json /path/to/evaluation_metrics.json \\
    --threshold 0.5 \\
    --box_dsc_threshold 0.22 \\
    --threshold_min 0.01 \\
    --threshold_max 0.99 \\
    --threshold_steps 99 \\
    --connectivity 26
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from util.config import get_data_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii", ".npz", ".npy")

# NPZ key candidates in priority order
DEFAULT_NPZ_KEYS: Tuple[str, ...] = (
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
# Ground truth pattern
GT_PATTERN_DEFAULT = "*gt_norm*"

# Default subject-ID regex used by this codebase (matches RESP followed by digits).
SUBJECT_ID_REGEX_DEFAULT = r"(RESP\d+)"

# Derive path defaults from config.json when available; fall back to None so
# argparse will require them explicitly when no config is present.
_data_root = get_data_root()
K_FOLD_SPLITS_PATH: Optional[str] = (
    str(_data_root / "preprocessing" / "k_fold_splits.json") if _data_root else None
)
GT_DIR_DEFAULT: Optional[str] = (
    str(_data_root / "preprocessing" / "mri") if _data_root else None
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VolumeMeta:
    """Metadata extracted alongside a loaded volume array."""

    shape: Tuple[int, ...]
    affine: Optional[np.ndarray]
    voxel_spacing: Tuple[float, ...]  # mm per voxel per axis
    voxel_volume_mm3: float


@dataclass
class VoxelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


@dataclass
class ComponentInfo:
    """Properties of one 3-D connected component."""

    component_id: int
    voxel_count: int
    volume_ml: float
    center_of_mass: Tuple[float, float, float]
    # Half-open bounding box in voxel coordinates:
    # (ax0_min, ax0_max, ax1_min, ax1_max, ax2_min, ax2_max)
    bbox: Tuple[int, int, int, int, int, int]


@dataclass
class SubjectRecord:
    """Holds loaded, pre-processed data for one subject."""

    subject_id: str
    pred_path: str
    fold_id: Optional[str]
    split_role: Optional[str]
    gt_path: Optional[str]
    is_control: bool
    pred_soft: np.ndarray  # float32, shape (D,H,W), values in [0,1]
    gt_bin: np.ndarray     # bool, same shape
    meta: VolumeMeta


@dataclass
class MatchedPrediction:
    """Matched prediction/GT paths with fold and split-role provenance."""

    subject_id: str
    pred_path: Path
    gt_path: Optional[Path]
    fold_id: Optional[str]
    split_role: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Return num/den, or None if denominator is 0 or either argument is None."""
    if den is None or den == 0:
        return None
    if num is None:
        return None
    return num / den


def safe_f1(
    precision: Optional[float], recall: Optional[float]
) -> Optional[float]:
    """Harmonic mean of precision and recall; returns 0.0 when undefined."""
    if precision is None or recall is None:
        return 0.0
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def ensure_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars, non-finite floats, and arrays to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return ensure_json_serializable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def auc_pr(
    points: List[Dict[str, Any]],
    recall_key: str = "recall",
    precision_key: str = "precision",
) -> Optional[float]:
    """Approximate PR-AUC via trapezoidal integration after sorting by recall."""
    valid = [
        (p[recall_key], p[precision_key])
        for p in points
        if p.get(recall_key) is not None and p.get(precision_key) is not None
    ]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda x: x[0])
    recalls = [v[0] for v in valid]
    precisions = [v[1] for v in valid]
    # np.trapz was removed in NumPy 2.0; use np.trapezoid with fallback.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(precisions, recalls))


def _nan_mean(values: List[Optional[float]]) -> Optional[float]:
    """Mean over a list, ignoring None entries."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _ci95_percentile(values: List[Optional[float]]) -> Optional[List[float]]:
    """Return empirical 95% CI [p2.5, p97.5] over non-None values."""
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return [float(lo), float(hi)]


def _extract_fold_id(path: str) -> Optional[str]:
    """Extract fold identifier (e.g., fold_0) from a prediction path."""
    for part in Path(path).parts:
        if re.fullmatch(r"fold_\d+", part, re.IGNORECASE):
            return part.lower()
    return None



def _as_finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _bootstrap_ci(
    values: Sequence[Optional[float]],
    statistic: str = "mean",
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 12345,
) -> Optional[List[float]]:
    """Non-parametric bootstrap CI over subject-level values."""
    valid = np.asarray(
        [float(v) for v in values if _as_finite_float(v) is not None],
        dtype=np.float64,
    )
    if valid.size == 0:
        return None
    if valid.size == 1:
        x = float(valid[0])
        return [x, x]

    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported statistic: {statistic}")

    rng = np.random.default_rng(seed)
    n = int(valid.size)
    boot_stats = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = valid[rng.integers(0, n, size=n)]
        if statistic == "mean":
            boot_stats[i] = float(np.mean(sample))
        else:
            boot_stats[i] = float(np.median(sample))

    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(boot_stats, [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


def _bootstrap_binary_rate(
    values: Sequence[Optional[float]],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 12345,
) -> Optional[List[float]]:
    """Bootstrap CI for a binary rate; values should be 0/1 indicators."""
    return _bootstrap_ci(
        values=values,
        statistic="mean",
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )


def _cluster_bootstrap_subjects(
    rows: Sequence[Dict[str, Any]],
    compute_fn: Callable[[List[Dict[str, Any]]], Optional[float]],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 12345,
) -> Optional[List[float]]:
    """Bootstrap CI by resampling subjects, then recomputing a pooled metric."""
    if len(rows) == 0:
        return None
    if len(rows) == 1:
        single = compute_fn([rows[0]])
        if single is None:
            return None
        x = float(single)
        return [x, x]

    rng = np.random.default_rng(seed)
    n = len(rows)
    stats: List[float] = []
    for _ in range(int(n_boot)):
        sample_idxs = rng.integers(0, n, size=n)
        sampled = [rows[int(i)] for i in sample_idxs]
        v = compute_fn(sampled)
        fv = _as_finite_float(v)
        if fv is not None:
            stats.append(fv)

    if not stats:
        return None
    if len(stats) == 1:
        return [stats[0], stats[0]]

    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(np.asarray(stats, dtype=np.float64), [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# File discovery  (mirrors conventions from plot_mri_module_npz_maps.py)
# ---------------------------------------------------------------------------


def _file_has_supported_ext(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _is_validation_prediction_path(path: Path) -> bool:
    """
    Return True when path clearly points to validation predictions.

    Rules:
    - must contain token 'val' or 'validation'
    - must NOT contain token 'train'
    """
    tokens = {
        tok
        for tok in re.split(r"[^a-z0-9]+", str(path).lower())
        if tok
    }
    has_val = ("val" in tokens) or ("validation" in tokens)
    has_train = "train" in tokens
    return has_val and not has_train


def _find_fold_dirs(pred_dir: Path) -> Dict[str, Path]:
    """Discover nnUNet fold directories named fold_<idx>."""
    fold_dirs: Dict[str, Path] = {}

    if pred_dir.is_dir() and re.fullmatch(r"fold_\d+", pred_dir.name):
        fold_dirs[pred_dir.name] = pred_dir
        return fold_dirs

    try:
        children = sorted(pred_dir.iterdir())
    except OSError:
        return fold_dirs

    for child in children:
        if not child.is_dir():
            continue
        if re.fullmatch(r"fold_\d+", child.name):
            fold_dirs[child.name] = child
    return fold_dirs


def _load_kfold_split_ids(json_path: Path, role: str) -> Dict[str, set]:
    """Load subject IDs per fold for the requested split role."""
    with open(str(json_path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    fold_payload = payload.get("folds", {})
    if not isinstance(fold_payload, dict):
        return {}

    role_key = {
        "val": "val_ids",
        "test": "test_ids",
        "all": None,
    }.get(role)

    out: Dict[str, set] = {}
    for fold_name, fold_data in fold_payload.items():
        if not isinstance(fold_data, dict):
            continue
        if role_key is None:
            merged: set = set()
            for k in ("val_ids", "test_ids", "train_ids"):
                ids = fold_data.get(k, [])
                if isinstance(ids, list):
                    merged |= {str(v).upper() for v in ids}
            out[str(fold_name)] = merged
            continue

        ids = fold_data.get(role_key, [])
        if not isinstance(ids, list):
            continue
        out[str(fold_name)] = {str(v).upper() for v in ids}
    return out

def _index_fold_preds_by_role(
    pred_dir: Path,
    pred_pattern: str,
    subject_regex: Optional[str],
    warnings_list: List[str],
    fold_split_json: Optional[Path],
    fold_role: str,
) -> Dict[str, Tuple[Path, str, str]]:
    """Index fold predictions and optionally filter by split role per fold."""
    fold_dirs = _find_fold_dirs(pred_dir)
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold_<idx> directories found under: {pred_dir}"
        )

    selected_ids_by_fold: Dict[str, set] = {}
    if fold_role in {"val", "test"}:
        splits_path = fold_split_json or (Path(K_FOLD_SPLITS_PATH) if K_FOLD_SPLITS_PATH else None)
        if splits_path is None:
            raise ValueError(
                "No fold split JSON available. Pass --fold_split_json or set data_root in config.json."
            )
        if not splits_path.is_file():
            raise FileNotFoundError(
                f"Fold split JSON not found: {splits_path}"
            )
        selected_ids_by_fold = _load_kfold_split_ids(splits_path, fold_role)
    else:
        warnings_list.append(
            "[pred] fold_role=all selected: no split-ID filtering will be applied. "
            "This is diagnostic only and not appropriate for primary performance reporting."
        )

    pred_map: Dict[str, Tuple[Path, str, str]] = {}
    first_fold_seen: Dict[str, str] = {}

    for fold_name, fold_dir in sorted(fold_dirs.items()):
        fold_pred_map = _index_files_by_id(
            fold_dir,
            pred_pattern,
            subject_regex,
            warnings_list,
            f"pred:{fold_name}",
            require_validation_pred_path=False,
            prefer_soft_predictions=True,
        )

        if fold_role in {"val", "test"}:
            allowed_ids = selected_ids_by_fold.get(fold_name, set())
            if not allowed_ids:
                warnings_list.append(
                    f"[pred:{fold_name}] Missing or empty {fold_role}_ids in split JSON; skipping fold."
                )
                continue
            candidate_ids = sorted(set(fold_pred_map.keys()) & allowed_ids)
        else:
            candidate_ids = sorted(fold_pred_map.keys())

        for sid in candidate_ids:
            path = fold_pred_map[sid]
            if sid in pred_map:
                prev_fold = first_fold_seen.get(sid, "unknown")
                warnings_list.append(
                    f"[pred] Subject '{sid}' appears in multiple selected folds "
                    f"({prev_fold}, {fold_name}); keeping first path {pred_map[sid][0]}, ignoring {path}."
                )
                continue
            pred_map[sid] = (path, fold_name, fold_role)
            first_fold_seen[sid] = fold_name

    return pred_map


def _index_legacy_mode_preds(
    pred_dir: Path,
    pred_pattern: str,
    subject_regex: Optional[str],
    warnings_list: List[str],
    nnunet_preds: bool,
    test_set: bool,
) -> Dict[str, Tuple[Path, str, str]]:
    """Legacy prediction discovery for backward compatibility."""
    pred_map: Dict[str, Tuple[Path, str, str]] = {}

    if test_set or nnunet_preds:
        legacy_role = "test" if test_set else "val"
        warnings_list.append(
            f"[pred] Legacy flag selected ({'--test-set' if test_set else '--nnunet-preds'}); "
            f"routing internally as fold-role '{legacy_role}'."
        )
        if K_FOLD_SPLITS_PATH is None:
            raise ValueError(
                "No fold split JSON available for legacy mode. "
                "Pass --fold_split_json or set data_root in config.json."
            )
        return _index_fold_preds_by_role(
            pred_dir=pred_dir,
            pred_pattern=pred_pattern,
            subject_regex=subject_regex,
            warnings_list=warnings_list,
            fold_split_json=Path(K_FOLD_SPLITS_PATH),
            fold_role=legacy_role,
        )

    flat_pred_map = _index_files_by_id(
        pred_dir,
        pred_pattern,
        subject_regex,
        warnings_list,
        "pred",
        require_validation_pred_path=True,
        prefer_soft_predictions=True,
    )
    for sid, path in flat_pred_map.items():
        pred_map[sid] = (path, _extract_fold_id(str(path)) or "__no_fold__", "val")
    return pred_map


def _stem_without_ext(name: str) -> str:
    """Remove all known extensions from a filename stem."""
    for ext in (".nii.gz", ".npy", ".npz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def extract_subject_id(path: Path, subject_regex: Optional[str]) -> Optional[str]:
    """
    Extract subject ID from a file path.

    1. If ``subject_regex`` is given, use the first capture group (or whole match).
    2. Otherwise try the codebase-standard RESP#### pattern (RESP followed by digits).
    3. Fall back to the filename stem (without extension).
    """
    name = path.name
    stem = _stem_without_ext(name)

    if subject_regex:
        m = re.search(subject_regex, name, re.IGNORECASE)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
        return None

    m = re.search(SUBJECT_ID_REGEX_DEFAULT, name, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return stem if stem else None


def _normalize_subject_token(
    raw: str,
    subject_regex: Optional[str],
) -> Optional[str]:
    """Normalize one subject token from a text list into a subject ID."""
    token = str(raw).strip()
    if not token:
        return None

    # Ignore comment lines in subject list files.
    if token.startswith("#"):
        return None

    if subject_regex:
        m = re.search(subject_regex, token, re.IGNORECASE)
        if m:
            sid = m.group(1) if m.lastindex else m.group(0)
            sid = sid.strip()
            return sid.upper() if sid else None

    m = re.search(SUBJECT_ID_REGEX_DEFAULT, token, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return token.upper()


def load_subjects_list(
    subjects_list_path: Path,
    subject_regex: Optional[str],
) -> List[str]:
    """Load a required subject-ID list from a text file (one subject per line)."""
    if not subjects_list_path.is_file():
        raise FileNotFoundError(f"subjects_list file not found: {subjects_list_path}")

    required_ids: List[str] = []
    seen: set = set()

    with open(str(subjects_list_path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            sid = _normalize_subject_token(line, subject_regex)
            if sid is None:
                continue
            if sid in seen:
                continue
            if sid == "":
                raise ValueError(
                    f"Invalid empty subject ID in {subjects_list_path} at line {line_no}."
                )
            seen.add(sid)
            required_ids.append(sid)

    if not required_ids:
        raise ValueError(
            f"subjects_list file {subjects_list_path} did not contain any valid subject IDs."
        )

    return required_ids


def _prediction_path_priority(path: Path) -> int:
    """Higher values are preferred when multiple prediction files map to one subject."""
    name = path.name.lower()
    if name.endswith(".npz"):
        return 3
    if name.endswith(".npy"):
        return 2
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return 1
    return 0


def _index_files_by_id(
    directory: Path,
    pattern: str,
    subject_regex: Optional[str],
    warnings_list: List[str],
    label: str,
    require_validation_pred_path: bool = False,
    prefer_soft_predictions: bool = False,
    require_path_token: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Recursively glob ``pattern`` under ``directory`` and map subject_id -> path.
    Only files with supported volume extensions are kept.
    First match per subject ID is used; duplicates are warned. When
    ``prefer_soft_predictions`` is True, .npz/.npy predictions are preferred over
    NIfTI files for the same subject.
    """
    glob_pattern = str(directory / "**" / pattern)
    matches = sorted(glob(glob_pattern, recursive=True))

    mapping: Dict[str, Path] = {}
    for m in matches:
        p = Path(m)
        if not p.is_file():
            continue
        if not _file_has_supported_ext(p.name):
            continue
        if require_validation_pred_path and not _is_validation_prediction_path(p):
            continue
        if require_path_token is not None:
            path_tokens = {
                tok for tok in re.split(r"[^a-z0-9]+", str(p).lower()) if tok
            }
            if require_path_token.lower() not in path_tokens:
                continue
        sid = extract_subject_id(p, subject_regex)
        if sid is None:
            warnings_list.append(
                f"[{label}] Cannot extract subject ID from: {p}; skipping file."
            )
            continue
        if sid in mapping:
            current = mapping[sid]
            if prefer_soft_predictions and (
                _prediction_path_priority(p) > _prediction_path_priority(current)
            ):
                warnings_list.append(
                    f"[{label}] Duplicate subject ID '{sid}': "
                    f"replacing {current} with preferred prediction file {p}."
                )
                mapping[sid] = p
            else:
                warnings_list.append(
                    f"[{label}] Duplicate subject ID '{sid}': "
                    f"keeping {current}, ignoring {p}."
                )
        else:
            mapping[sid] = p
    return mapping


def discover_subject_files(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str,
    gt_pattern: str,
    subject_regex: Optional[str],
    allow_missing_gt_as_control: bool,
    warnings_list: List[str],
    fold_split_json: Optional[Path],
    fold_role: str,
    nnunet_preds: bool = False,
    test_set: bool = False,
    required_subject_ids: Optional[List[str]] = None,
) -> Tuple[List[MatchedPrediction], List[Dict[str, str]]]:
    """
    Build matched list of (subject_id, pred_path, gt_path).

    - pred without GT: skip unless ``allow_missing_gt_as_control`` is True (gt_path=None).
    - GT without pred: skip with warning.

    Primary mode uses fold-based split-role filtering:
    - fold_role=test: keep test_ids per fold
    - fold_role=val : keep val_ids per fold
    - fold_role=all : do not filter by split IDs

    Legacy flags --nnunet-preds and --test-set are internally mapped to
    fold_role=val and fold_role=test respectively.

    Returns (matched_list, skipped_list), where matched_list carries fold_id and split_role.
    """
    fold_dirs = _find_fold_dirs(pred_dir)
    if fold_dirs:
        pred_map = _index_fold_preds_by_role(
            pred_dir=pred_dir,
            pred_pattern=pred_pattern,
            subject_regex=subject_regex,
            warnings_list=warnings_list,
            fold_split_json=fold_split_json,
            fold_role=fold_role,
        )
    else:
        pred_map = _index_legacy_mode_preds(
            pred_dir=pred_dir,
            pred_pattern=pred_pattern,
            subject_regex=subject_regex,
            warnings_list=warnings_list,
            nnunet_preds=nnunet_preds,
            test_set=test_set,
        )
    gt_map = _index_files_by_id(
        gt_dir, gt_pattern, subject_regex, warnings_list, "gt"
    )
    print(f"Discovered {len(pred_map)} prediction files and {len(gt_map)} GT files.")
    matched: List[MatchedPrediction] = []
    skipped: List[Dict[str, str]] = []

    if required_subject_ids is not None:
        missing_reasons: List[str] = []
        for sid in required_subject_ids:
            pred_info = pred_map.get(sid)
            gt_path = gt_map.get(sid)

            if pred_info is None:
                missing_reasons.append(f"{sid}: prediction missing")
                continue

            if gt_path is None and not allow_missing_gt_as_control:
                missing_reasons.append(
                    f"{sid}: GT missing (set --allow_missing_gt_as_control to allow)"
                )
                continue

            pred_path, fold_id, pred_role = pred_info
            matched.append(
                MatchedPrediction(
                    subject_id=sid,
                    pred_path=pred_path,
                    gt_path=gt_path,
                    fold_id=fold_id,
                    split_role=pred_role,
                )
            )

        if missing_reasons:
            missing_msg = "\n  - ".join(missing_reasons)
            raise ValueError(
                "Required subjects from --subjects_list are missing and evaluation "
                f"must fail:\n  - {missing_msg}"
            )

        # Keep standard skipped reporting for GT-only records among requested IDs.
        requested = set(required_subject_ids)
        for sid in sorted(requested):
            if sid in gt_map and sid not in pred_map:
                skipped.append({
                    "subject_id": sid,
                    "reason": "GT exists but prediction missing; skipping.",
                })

        return matched, skipped

    for sid in sorted(set(pred_map) | set(gt_map)):
        pred_info = pred_map.get(sid)
        gt_path = gt_map.get(sid)

        if pred_info is None:
            skipped.append({
                "subject_id": sid,
                "reason": "GT exists but prediction missing; skipping.",
            })
            continue

        pred_path, fold_id, pred_role = pred_info

        if gt_path is None:
            if allow_missing_gt_as_control:
                matched.append(
                    MatchedPrediction(
                        subject_id=sid,
                        pred_path=pred_path,
                        gt_path=None,
                        fold_id=fold_id,
                        split_role=pred_role,
                    )
                )
            else:
                skipped.append({
                    "subject_id": sid,
                    "reason": (
                        "Prediction exists but GT missing. "
                        "Use --allow_missing_gt_as_control to treat as control."
                    ),
                })
            continue

        matched.append(
            MatchedPrediction(
                subject_id=sid,
                pred_path=pred_path,
                gt_path=gt_path,
                fold_id=fold_id,
                split_role=pred_role,
            )
        )

    return matched, skipped


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------


def _pick_3d_map(arr: np.ndarray) -> np.ndarray:
    """
    Select a 3-D spatial map from an N-D array using channel conventions.
    Mirrors _pick_3d_map from plot_mri_module_maps.py.
    """
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # Channel-first [C, D, H, W] with small C
        if (
            arr.shape[0] <= 4
            and arr.shape[1] > 4
            and arr.shape[2] > 4
            and arr.shape[3] > 4
        ):
            return arr[0] if arr.shape[0] == 1 else arr[1]
        # Channel-last [D, H, W, C] with small C
        if (
            arr.shape[-1] <= 4
            and arr.shape[0] > 4
            and arr.shape[1] > 4
            and arr.shape[2] > 4
        ):
            return arr[..., 0] if arr.shape[-1] == 1 else arr[..., 1]
        # Fall back: reduce smallest axis
        axis = int(np.argmin(arr.shape))
        return arr.take(indices=min(1, arr.shape[axis] - 1), axis=axis)
    if arr.ndim == 5:
        # Common model output: [B, C, D, H, W]
        return _pick_3d_map(arr[0])
    raise ValueError(f"Unsupported array shape for 3-D map extraction: {arr.shape}")


def load_volume(
    path: Path, prob_key: Optional[str] = None
) -> Tuple[np.ndarray, VolumeMeta]:
    """
    Load a volume file (.nii/.nii.gz/.npy/.npz) and return (data_float32, meta).

    For .npy/.npz files without spatial metadata, voxel_volume_mm3 is set to 1.0.
    A runtime warning is emitted for .npy files. For .npz files, warning emission
    is deferred to higher-level logic because NPZ predictions can be aligned to a
    paired NIfTI and inherit real spacing metadata.
    """
    name = path.name.lower()

    # ----- NIfTI -----
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        import nibabel as nib  # type: ignore

        img = nib.load(str(path))
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        affine = np.asarray(img.affine, dtype=np.float64)
        zooms = img.header.get_zooms()
        spacing = tuple(float(z) for z in zooms[:3])
        vol = spacing[0] * spacing[1] * spacing[2] if len(spacing) >= 3 else 1.0
        meta = VolumeMeta(
            shape=data.shape,
            affine=affine,
            voxel_spacing=spacing,
            voxel_volume_mm3=vol,
        )
        return data, meta

    # ----- NumPy -----
    if name.endswith(".npy"):
        data = np.load(str(path)).astype(np.float32)
        warnings.warn(
            f"Loaded .npy file {path}: no voxel spacing available; "
            "assuming 1 mm^3 voxels.",
            stacklevel=2,
        )
        meta = VolumeMeta(
            shape=data.shape,
            affine=None,
            voxel_spacing=(1.0, 1.0, 1.0),
            voxel_volume_mm3=1.0,
        )
        return data, meta

    # ----- NPZ -----
    if name.endswith(".npz"):
        keys_to_try: List[str] = []
        if prob_key:
            keys_to_try.append(prob_key)
        keys_to_try.extend(DEFAULT_NPZ_KEYS)

        with np.load(str(path), allow_pickle=True) as npz:
            available = list(npz.keys())
            if not available:
                raise ValueError(f"No arrays in NPZ: {path}")

            seen: set = set()
            ordered = [
                k for k in keys_to_try + available if not (k in seen or seen.add(k))
            ]

            arr = None
            for k in ordered:
                if k not in npz:
                    continue
                candidate = np.asarray(npz[k])
                if candidate.ndim >= 3:
                    arr = candidate
                    break

        if arr is None:
            raise ValueError(
                f"No suitable 3-D+ array in NPZ {path}. Keys: {available}"
            )

        data = _pick_3d_map(arr).astype(np.float32)
        meta = VolumeMeta(
            shape=data.shape,
            affine=None,
            voxel_spacing=(1.0, 1.0, 1.0),
            voxel_volume_mm3=1.0,
        )
        return data, meta

    raise ValueError(
        f"Unsupported file format: {path}. Supported: {SUPPORTED_EXTENSIONS}"
    )


def _to_probability_like(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary prediction-like arrays onto [0, 1] for orientation scoring."""
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
        out = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
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
    """Match NPZ axis order/flips to a paired reference NIfTI map."""
    src = np.asarray(map3d, dtype=np.float32)
    ref = np.asarray(ref3d, dtype=np.float32)

    best_map = src
    best_desc = "identity"
    best_score = (
        _score_map_similarity(src, ref)
        if src.shape == ref.shape
        else float("-inf")
    )

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


def _find_paired_reference_nifti(
    pred_path: Path,
    subject_id: str,
    pred_root: Path,
) -> Optional[Path]:
    """Find the best paired NIfTI prediction to use as NPZ orientation reference."""
    search_root = pred_root
    for parent in pred_path.parents:
        if re.fullmatch(r"fold_\d+", parent.name):
            search_root = parent
            break
        if parent == pred_root:
            break

    candidates: List[Path] = []
    for match in sorted(glob(str(search_root / "**" / "*.nii*"), recursive=True)):
        path = Path(match)
        if not path.is_file():
            continue
        if extract_subject_id(path, None) != subject_id:
            continue
        candidates.append(path)

    if not candidates:
        return None

    pred_stem = _stem_without_ext(pred_path.name).lower()

    def _priority(path: Path) -> Tuple[int, int, str]:
        stem = _stem_without_ext(path.name).lower()
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", path.name.lower()) if tok}
        same_stem_penalty = 0 if stem == pred_stem else 1
        semantic_penalty = 0
        if "gt" in tokens or "mask" in tokens:
            semantic_penalty += 4
        if "t1" in tokens or "t1w" in tokens:
            semantic_penalty += 2
        return (same_stem_penalty + semantic_penalty, len(path.parts), str(path).lower())

    candidates.sort(key=_priority)
    return candidates[0]


def maybe_resample_to_gt(
    pred: np.ndarray,
    pred_meta: VolumeMeta,
    gt_shape: Tuple[int, ...],
    gt_meta: VolumeMeta,
    resample: bool,
    warnings_list: List[str],
    subject_id: str,
) -> Tuple[np.ndarray, VolumeMeta]:
    """
    If pred shape does not match GT shape, either raise (resample=False) or
    resample prediction to the GT grid using linear interpolation (resample=True).
    """
    if pred.shape == gt_shape:
        return pred, pred_meta

    msg = (
        f"[{subject_id}] Shape mismatch: pred={pred.shape}, gt={gt_shape}."
    )
    if not resample:
        raise ValueError(
            msg + " Use --resample-to-gt to enable resampling."
        )

    warnings_list.append(
        msg
        + " Resampling prediction to GT grid (LINEAR interpolation). "
        "Verify results carefully."
    )
    warnings.warn(msg + " Resampling to GT grid.", stacklevel=2)

    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except ImportError:
        raise ImportError("nibabel is required for resampling (--resample-to-gt).")

    if pred_meta.affine is None or gt_meta.affine is None:
        raise ValueError(
            f"[{subject_id}] Cannot resample: one or both volumes lack affine "
            "information (npy/npz files have no spatial metadata)."
        )

    pred_nii = nib.Nifti1Image(pred.astype(np.float32), pred_meta.affine)
    # Build a target image with GT affine and shape
    gt_dummy = nib.Nifti1Image(np.zeros(gt_shape, dtype=np.float32), gt_meta.affine)
    resampled = resample_from_to(pred_nii, gt_dummy, order=1)
    pred_out = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)

    new_meta = VolumeMeta(
        shape=pred_out.shape,
        affine=gt_meta.affine,
        voxel_spacing=gt_meta.voxel_spacing,
        voxel_volume_mm3=gt_meta.voxel_volume_mm3,
    )
    return pred_out, new_meta


# ---------------------------------------------------------------------------
# Prediction preprocessing
# ---------------------------------------------------------------------------


def preprocess_prediction(
    pred: np.ndarray,
    subject_id: str,
    warnings_list: List[str],
) -> np.ndarray:
    """
    Convert to float32, replace NaN/Inf with 0, clip to [0, 1].
    Warns loudly if values are grossly outside [0, 1].
    """
    pred = pred.astype(np.float32)

    n_nan = int(np.isnan(pred).sum())
    n_inf = int(np.isinf(pred).sum())
    if n_nan > 0 or n_inf > 0:
        warnings_list.append(
            f"[{subject_id}] Prediction has {n_nan} NaN and {n_inf} Inf values; "
            "replacing with 0."
        )
        pred = np.where(np.isfinite(pred), pred, 0.0)

    vmin, vmax = float(pred.min()), float(pred.max())
    if vmin < -0.01 or vmax > 1.01:
        warnings_list.append(
            f"[{subject_id}] Prediction values grossly outside [0,1]: "
            f"min={vmin:.4f}, max={vmax:.4f}. Clipping."
        )
    pred = np.clip(pred, 0.0, 1.0)
    return pred


def binarize_prediction(pred_soft: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize soft prediction at the given threshold."""
    return pred_soft >= threshold


# ---------------------------------------------------------------------------
# Voxel metrics
# ---------------------------------------------------------------------------


def compute_voxel_counts(
    pred_bin: np.ndarray, gt_bin: np.ndarray
) -> VoxelCounts:
    pred_bool = pred_bin.astype(bool)
    gt_bool = gt_bin.astype(bool)
    return VoxelCounts(
        tp=int(np.sum(pred_bool & gt_bool)),
        fp=int(np.sum(pred_bool & ~gt_bool)),
        fn=int(np.sum(~pred_bool & gt_bool)),
        tn=int(np.sum(~pred_bool & ~gt_bool)),
    )


def compute_voxel_metrics(
    counts: VoxelCounts, voxel_volume_mm3: float
) -> Dict[str, Any]:
    """
    Compute voxel-level metrics from TP/FP/FN/TN counts.

    voxel_dice is None when both masks are empty (both pred and GT are all zeros),
    because the metric is undefined. The caller should handle this case based on
    is_control.
    """
    tp, fp, fn, tn = counts.tp, counts.fp, counts.fn, counts.tn
    pred_volume_voxels = tp + fp
    gt_volume_voxels = tp + fn

    dice_denom = 2 * tp + fp + fn
    voxel_precision = safe_div(tp, tp + fp)
    voxel_recall = safe_div(tp, tp + fn)
    voxel_dice: Optional[float] = safe_div(2 * tp, dice_denom)
    if voxel_precision is None or voxel_recall is None:
        voxel_dice = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "voxel_dice": voxel_dice,
        "voxel_precision": voxel_precision,
        "voxel_recall": voxel_recall,
        "voxel_sensitivity": voxel_recall,
        "voxel_specificity": safe_div(tn, tn + fp),
        "pred_volume_voxels": pred_volume_voxels,
        "gt_volume_voxels": gt_volume_voxels,
        "pred_volume_ml": pred_volume_voxels * voxel_volume_mm3 / 1000.0,
        "gt_volume_ml": gt_volume_voxels * voxel_volume_mm3 / 1000.0,
    }


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def get_connectivity_structure(connectivity: int) -> np.ndarray:
    """
    Return 3-D scipy structuring element for the requested connectivity.
      6  -> face-connected only
      18 -> faces + edges
      26 -> faces + edges + corners (full 3x3x3 neighbourhood)
    """
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError(
        f"Unsupported connectivity: {connectivity}. Choose 6, 18, or 26."
    )


def bbox_from_mask_indices(indices: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Compute half-open bounding box from an (N, 3) array of voxel indices.
    Returns (ax0_min, ax0_max, ax1_min, ax1_max, ax2_min, ax2_max).
    """
    ax0_min = int(indices[:, 0].min())
    ax0_max = int(indices[:, 0].max()) + 1
    ax1_min = int(indices[:, 1].min())
    ax1_max = int(indices[:, 1].max()) + 1
    ax2_min = int(indices[:, 2].min())
    ax2_max = int(indices[:, 2].max()) + 1
    return (ax0_min, ax0_max, ax1_min, ax1_max, ax2_min, ax2_max)


def bbox_dice(
    box_a: Tuple[int, int, int, int, int, int],
    box_b: Tuple[int, int, int, int, int, int],
) -> float:
    """
    Bounding-box Dice Similarity Coefficient.

    Boxes are half-open: (ax0_min, ax0_max, ax1_min, ax1_max, ax2_min, ax2_max).
    DSC = 2 * |intersection| / (|A| + |B|).

    Note: this is a coarse spatial overlap metric operating on axis-aligned bounding
    boxes.  It differs from voxel Dice, which measures exact voxel-wise overlap.
    """
    inter0 = max(0, min(box_a[1], box_b[1]) - max(box_a[0], box_b[0]))
    inter1 = max(0, min(box_a[3], box_b[3]) - max(box_a[2], box_b[2]))
    inter2 = max(0, min(box_a[5], box_b[5]) - max(box_a[4], box_b[4]))
    inter_vol = inter0 * inter1 * inter2

    vol_a = (
        max(0, box_a[1] - box_a[0])
        * max(0, box_a[3] - box_a[2])
        * max(0, box_a[5] - box_a[4])
    )
    vol_b = (
        max(0, box_b[1] - box_b[0])
        * max(0, box_b[3] - box_b[2])
        * max(0, box_b[5] - box_b[4])
    )

    denom = vol_a + vol_b
    if denom == 0:
        return 0.0
    return float(2 * inter_vol / denom)


def point_in_mask(
    point: Tuple[float, float, float], mask: np.ndarray
) -> bool:
    """
    Check whether a (float) point lies inside a binary mask.
    Rounds to nearest integer voxel index.
    """
    idx = tuple(int(round(c)) for c in point)
    for dim, i in enumerate(idx):
        if i < 0 or i >= mask.shape[dim]:
            return False
    return bool(mask[idx])


def extract_components(
    mask: np.ndarray,
    voxel_volume_mm3: float,
    connectivity: int,
    min_cluster_size: int = 0,
) -> Tuple[np.ndarray, List[ComponentInfo]]:
    """
    Label connected components in a binary mask and return per-component properties.

    Components with voxel_count < min_cluster_size are zeroed out of the label image
    and excluded from the returned list.

    Returns (label_img, components).
    """
    struct = get_connectivity_structure(connectivity)
    label_img, n_labels = ndimage.label(mask.astype(bool), structure=struct)
    components: List[ComponentInfo] = []

    for comp_id in range(1, n_labels + 1):
        comp_mask = label_img == comp_id
        indices = np.argwhere(comp_mask)
        voxel_count = int(indices.shape[0])

        if voxel_count < min_cluster_size:
            label_img[comp_mask] = 0
            continue

        com = ndimage.center_of_mass(comp_mask)
        bbox = bbox_from_mask_indices(indices)
        volume_ml = voxel_count * voxel_volume_mm3 / 1000.0

        components.append(
            ComponentInfo(
                component_id=comp_id,
                voxel_count=voxel_count,
                volume_ml=volume_ml,
                center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
                bbox=bbox,
            )
        )

    return label_img, components


# ---------------------------------------------------------------------------
# Cluster matching
# ---------------------------------------------------------------------------


def match_pred_clusters_to_gt(
    pred_components: List[ComponentInfo],
    gt_components: List[ComponentInfo],
    gt_label_img: np.ndarray,
    gt_bin: np.ndarray,
    box_dsc_threshold: float,
) -> List[Dict[str, Any]]:
    """
    For each predicted cluster determine whether it is a TP or FP by two criteria:

    Detection criterion (boxDSC):
      The bounding box of the predicted cluster overlaps with the bounding box of
      any GT component with boxDSC > box_dsc_threshold.

    Pinpointing criterion (COM inside GT):
      The centre-of-mass of the predicted cluster (rounded to nearest voxel) lies
      inside the GT binary mask (equivalently, inside any GT component).

    Multiple predicted clusters can match the same GT component; this is intentional
    because per the paper, every qualifying predicted cluster counts as a TP cluster
    for precision, while for sensitivity we count distinct GT components reached.
    """
    results: List[Dict[str, Any]] = []

    for pred_comp in pred_components:
        best_box_dsc = 0.0
        best_gt_id_detection: Optional[int] = None
        best_gt_id_pinpoint: Optional[int] = None
        pinpoint_in_gt = False

        com = pred_comp.center_of_mass

        for gt_comp in gt_components:
            dsc = bbox_dice(pred_comp.bbox, gt_comp.bbox)
            if dsc > best_box_dsc:
                best_box_dsc = dsc
            # Detection: take GT component with highest boxDSC above threshold
            if dsc > box_dsc_threshold:
                if best_gt_id_detection is None or dsc > bbox_dice(
                    pred_comp.bbox,
                    gt_components[
                        next(
                            i
                            for i, g in enumerate(gt_components)
                            if g.component_id == best_gt_id_detection
                        )
                    ].bbox,
                ):
                    best_gt_id_detection = gt_comp.component_id

            # Pinpointing: COM inside this GT component?
            gt_comp_mask = gt_label_img == gt_comp.component_id
            if point_in_mask(com, gt_comp_mask):
                best_gt_id_pinpoint = gt_comp.component_id

        # Also verify against the full GT binary mask for pinpointing
        pinpoint_in_gt = point_in_mask(com, gt_bin)

        is_tp_detection = best_gt_id_detection is not None
        # A cluster pinpoints if its COM is inside any GT component OR the GT mask
        is_tp_pinpoint = best_gt_id_pinpoint is not None or pinpoint_in_gt

        results.append({
            "pred_cluster_id": pred_comp.component_id,
            "pred_voxel_count": pred_comp.voxel_count,
            "pred_volume_ml": pred_comp.volume_ml,
            "pred_center_of_mass": list(pred_comp.center_of_mass),
            "pred_bbox": list(pred_comp.bbox),
            "best_box_dsc": best_box_dsc,
            "matched_gt_component_id_by_detection": best_gt_id_detection,
            "matched_gt_component_id_by_pinpointing": best_gt_id_pinpoint,
            "pinpoint_in_gt_mask": pinpoint_in_gt,
            "is_tp_detection_cluster": is_tp_detection,
            "is_tp_pinpoint_cluster": is_tp_pinpoint,
            "is_fp_detection_cluster": not is_tp_detection,
            "is_fp_pinpoint_cluster": not is_tp_pinpoint,
        })

    return results


# ---------------------------------------------------------------------------
# Subject-level evaluation at a single threshold
# ---------------------------------------------------------------------------


def evaluate_subject_at_threshold(
    record: SubjectRecord,
    threshold: float,
    connectivity: int,
    min_cluster_size: int,
    box_dsc_threshold: float,
) -> Dict[str, Any]:
    """
    Evaluate one subject at a given binarization threshold.

    Returns a dict containing voxel metrics and cluster-level metrics.

    Key design notes
    ----------------
    - Voxel Dice measures precise voxel-overlap; useful for localisation quality.
    - Cluster boxDSC is a coarser detection criterion: it rewards finding the right
      spatial region even when voxel-level segmentation is imperfect.
    - Subject detection rate (aggregated across subjects) is the primary clinical
      metric: did the model flag the correct brain region in this patient?
    - GT components are never filtered by min_cluster_size (only predicted clusters are).
    """
    pred_bin = binarize_prediction(record.pred_soft, threshold)
    gt_bin = record.gt_bin
    vox_vol = record.meta.voxel_volume_mm3

    counts = compute_voxel_counts(pred_bin, gt_bin)
    vox_metrics = compute_voxel_metrics(counts, vox_vol)

    pred_empty = (counts.tp + counts.fp) == 0
    gt_empty = (counts.tp + counts.fn) == 0

    # Extract predicted clusters (filtered by min_cluster_size)
    _pred_label_img, pred_components = extract_components(
        pred_bin.astype(bool), vox_vol, connectivity, min_cluster_size
    )
    # Extract GT components (no size filter)
    gt_label_img, gt_components = extract_components(
        gt_bin.astype(bool), vox_vol, connectivity, 0
    )

    # Match predicted clusters to GT
    cluster_matches = match_pred_clusters_to_gt(
        pred_components, gt_components, gt_label_img, gt_bin, box_dsc_threshold
    )

    n_pred_clusters = len(pred_components)
    n_gt_components = len(gt_components)

    # --- Detection-based cluster metrics ---
    n_tp_det = sum(1 for c in cluster_matches if c["is_tp_detection_cluster"])
    n_fp_det = n_pred_clusters - n_tp_det

    # Unique GT components detected by at least one predicted cluster
    detected_gt_ids = {
        c["matched_gt_component_id_by_detection"]
        for c in cluster_matches
        if c["matched_gt_component_id_by_detection"] is not None
    }
    n_detected_gt = len(detected_gt_ids)

    clust_prec_det = safe_div(n_tp_det, n_pred_clusters)
    clust_sens_det = safe_div(n_detected_gt, n_gt_components)
    clust_f1_det = safe_f1(clust_prec_det, clust_sens_det)
    subject_detected = n_detected_gt > 0

    # --- Pinpointing-based cluster metrics ---
    n_tp_pin = sum(1 for c in cluster_matches if c["is_tp_pinpoint_cluster"])
    n_fp_pin = n_pred_clusters - n_tp_pin

    pinpointed_gt_ids = {
        c["matched_gt_component_id_by_pinpointing"]
        for c in cluster_matches
        if c["matched_gt_component_id_by_pinpointing"] is not None
    }
    n_pinpointed_gt = len(pinpointed_gt_ids)
    # Fallback: if whole-GT-mask pinpointing succeeded but no labeled component matched
    if n_pinpointed_gt == 0 and n_gt_components > 0:
        if any(c["pinpoint_in_gt_mask"] for c in cluster_matches):
            n_pinpointed_gt = 1

    clust_prec_pin = safe_div(n_tp_pin, n_pred_clusters)
    clust_sens_pin = safe_div(n_pinpointed_gt, n_gt_components)
    clust_f1_pin = safe_f1(clust_prec_pin, clust_sens_pin)
    subject_pinpointed = n_pinpointed_gt > 0

    # Serialisable GT component list
    gt_comp_list = [
        {
            "component_id": c.component_id,
            "voxel_count": c.voxel_count,
            "volume_ml": c.volume_ml,
            "center_of_mass": list(c.center_of_mass),
            "bbox": list(c.bbox),
        }
        for c in gt_components
    ]

    return {
        "threshold": threshold,
        "is_control": record.is_control,
        "pred_empty": pred_empty,
        "gt_empty": gt_empty,
        "voxel": vox_metrics,
        "cluster": {
            "n_pred_clusters": n_pred_clusters,
            "n_gt_components": n_gt_components,
            "detection": {
                "n_tp_pred_clusters": n_tp_det,
                "n_fp_pred_clusters": n_fp_det,
                "n_detected_gt_components": n_detected_gt,
                "cluster_precision": clust_prec_det,
                "cluster_sensitivity": clust_sens_det,
                "cluster_f1": clust_f1_det,
                "subject_detected": subject_detected,
            },
            "pinpointing": {
                "n_tp_pred_clusters": n_tp_pin,
                "n_fp_pred_clusters": n_fp_pin,
                "n_pinpointed_gt_components": n_pinpointed_gt,
                "cluster_precision": clust_prec_pin,
                "cluster_sensitivity": clust_sens_pin,
                "cluster_f1": clust_f1_pin,
                "subject_pinpointed": subject_pinpointed,
            },
            # Verbose lists (separated into "clusters" in final JSON output)
            "predicted_clusters": cluster_matches,
            "gt_components": gt_comp_list,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation of fixed-threshold metrics
# ---------------------------------------------------------------------------


def _stat_payload(
    values: Sequence[Optional[float]],
    summary_stat: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    unit: str = "subject",
) -> Dict[str, Any]:
    valid = [_as_finite_float(v) for v in values]
    valid = [v for v in valid if v is not None]
    n = len(valid)

    if valid is not values:
        for i, v in enumerate(values):
            if v is not None and not np.isfinite(v):
                warnings.warn(
                    f"Non-finite value at index {i}: {v}. Ignoring in aggregation."
                )

    if summary_stat == "both":
        mean_est = float(np.mean(valid)) if valid else None
        med_est = float(np.median(valid)) if valid else None
        return {
            "estimate": {
                "mean": mean_est,
                "median": med_est,
            },
            "ci": {
                "mean": _bootstrap_ci(valid, "mean", n_bootstrap, ci_level, seed),
                "median": _bootstrap_ci(valid, "median", n_bootstrap, ci_level, seed + 1),
            },
            "n": n,
            "statistic": "both",
            "unit": unit,
        }

    stat_name = summary_stat
    est = None
    if valid:
        est = float(np.mean(valid)) if stat_name == "mean" else float(np.median(valid))
    return {
        "estimate": est,
        "ci": _bootstrap_ci(valid, stat_name, n_bootstrap, ci_level, seed),
        "n": n,
        "statistic": stat_name,
        "unit": unit,
    }


def aggregate_fixed_metrics(
    per_subject_fixed: List[Dict[str, Any]],
    subject_records: List[SubjectRecord],
    n_bootstrap: int,
    bootstrap_seed: int,
    ci_level: float,
    summary_stat: str,
) -> Dict[str, Any]:
    """
    Aggregate fixed-threshold metrics using subjects as the inferential unit.

    Confidence intervals are non-parametric bootstrap CIs over subjects.
    Folds are used only for identifying held-out predictions.
    """
    if len(per_subject_fixed) != len(subject_records):
        raise ValueError("per_subject_fixed and subject_records must have same length")

    entries: List[Tuple[SubjectRecord, Dict[str, Any]]] = list(zip(subject_records, per_subject_fixed))
    cases = [(r, m) for r, m in entries if not m["is_control"]]
    controls = [(r, m) for r, m in entries if m["is_control"]]
    case_non_empty = [(r, m) for r, m in cases if not m.get("gt_empty", False)]

    def _value_or_zero(v: Any) -> float:
        fv = _as_finite_float(v)
        return 0.0 if fv is None else float(fv)

    def _rows_for_boot(rows: List[Tuple[SubjectRecord, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [{"record": r, "metrics": m} for r, m in rows]

    all_rows = _rows_for_boot(entries)

    # Voxel macro over all subjects.
    voxel_macro_all_subjects = {
        "voxel_dice": _stat_payload(
            [_value_or_zero(m["voxel"].get("voxel_dice")) for _, m in entries],
            summary_stat,
            n_bootstrap,
            ci_level,
            bootstrap_seed + 10,
        ),
        "voxel_precision": _stat_payload(
            [_value_or_zero(m["voxel"].get("voxel_precision")) for _, m in entries],
            summary_stat,
            n_bootstrap,
            ci_level,
            bootstrap_seed + 11,
        ),
        "voxel_recall": _stat_payload(
            [_value_or_zero(m["voxel"].get("voxel_recall")) for _, m in entries],
            summary_stat,
            n_bootstrap,
            ci_level,
            bootstrap_seed + 12,
        ),
    }

    # Voxel micro pooled counts, CI via subject bootstrap (not voxel bootstrap).
    all_tp = int(sum(m["voxel"]["tp"] for _, m in entries))
    all_fp = int(sum(m["voxel"]["fp"] for _, m in entries))
    all_fn = int(sum(m["voxel"]["fn"] for _, m in entries))
    all_tn = int(sum(m["voxel"]["tn"] for _, m in entries))

    def _voxel_micro_metric(sampled_rows: List[Dict[str, Any]], metric: str) -> Optional[float]:
        tp = int(sum(row["metrics"]["voxel"]["tp"] for row in sampled_rows))
        fp = int(sum(row["metrics"]["voxel"]["fp"] for row in sampled_rows))
        fn = int(sum(row["metrics"]["voxel"]["fn"] for row in sampled_rows))
        tn = int(sum(row["metrics"]["voxel"]["tn"] for row in sampled_rows))
        if metric == "dice":
            return safe_div(2 * tp, 2 * tp + fp + fn)
        if metric == "precision":
            return safe_div(tp, tp + fp)
        if metric == "recall":
            return safe_div(tp, tp + fn)
        if metric == "specificity":
            return safe_div(tn, tn + fp)
        raise ValueError(metric)

    voxel_micro_all_subjects = {
        "counts": {"tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn},
        "dice": {
            "estimate": safe_div(2 * all_tp, 2 * all_tp + all_fp + all_fn),
            "voxel_micro_cluster_bootstrap_ci": _cluster_bootstrap_subjects(
                all_rows,
                lambda rows: _voxel_micro_metric(rows, "dice"),
                n_bootstrap,
                ci_level,
                bootstrap_seed + 30,
            ),
            "n": len(entries),
            "unit": "subject",
            "statistic": "pooled_micro",
        },
        "precision": {
            "estimate": safe_div(all_tp, all_tp + all_fp),
            "voxel_micro_cluster_bootstrap_ci": _cluster_bootstrap_subjects(
                all_rows,
                lambda rows: _voxel_micro_metric(rows, "precision"),
                n_bootstrap,
                ci_level,
                bootstrap_seed + 31,
            ),
            "n": len(entries),
            "unit": "subject",
            "statistic": "pooled_micro",
        },
        "recall": {
            "estimate": safe_div(all_tp, all_tp + all_fn),
            "voxel_micro_cluster_bootstrap_ci": _cluster_bootstrap_subjects(
                all_rows,
                lambda rows: _voxel_micro_metric(rows, "recall"),
                n_bootstrap,
                ci_level,
                bootstrap_seed + 32,
            ),
            "n": len(entries),
            "unit": "subject",
            "statistic": "pooled_micro",
        },
        "specificity": {
            "estimate": safe_div(all_tn, all_tn + all_fp),
            "voxel_micro_cluster_bootstrap_ci": _cluster_bootstrap_subjects(
                all_rows,
                lambda rows: _voxel_micro_metric(rows, "specificity"),
                n_bootstrap,
                ci_level,
                bootstrap_seed + 33,
            ),
            "n": len(entries),
            "unit": "subject",
            "statistic": "pooled_micro",
        },
    }

    def _cluster_micro(sampled_rows: List[Dict[str, Any]], mode: str, metric: str) -> Optional[float]:
        tp = int(sum(row["metrics"]["cluster"][mode]["n_tp_pred_clusters"] for row in sampled_rows))
        n_pred = int(sum(row["metrics"]["cluster"]["n_pred_clusters"] for row in sampled_rows))
        if mode == "detection":
            det_key = "n_detected_gt_components"
        else:
            det_key = "n_pinpointed_gt_components"
        n_detected = int(sum(row["metrics"]["cluster"][mode][det_key] for row in sampled_rows))
        n_gt = int(sum(row["metrics"]["cluster"]["n_gt_components"] for row in sampled_rows))
        precision = safe_div(tp, n_pred)
        recall = safe_div(n_detected, n_gt)
        if metric == "precision":
            return precision
        if metric == "recall":
            return recall
        if metric == "f1":
            return safe_f1(precision, recall)
        raise ValueError(metric)

    def _cluster_section(mode: str, seed_offset: int) -> Dict[str, Any]:
        if mode == "detection":
            rec_key = "subject_detected"
            det_key = "n_detected_gt_components"
        else:
            rec_key = "subject_pinpointed"
            det_key = "n_pinpointed_gt_components"

        # Use all subjects and treat undefined per-subject precision/recall as 0.
        subject_precision_values = [
            _value_or_zero(m["cluster"][mode].get("cluster_precision")) for _, m in entries
        ]
        subject_recall_values = [
            _value_or_zero(m["cluster"][mode].get("cluster_sensitivity")) for _, m in entries
        ]

        def _bootstrap_paired_macro_f1_ci(
            p_vals: List[float],
            r_vals: List[float],
            stat_name: str,
            n_boot: int,
            ci: float,
            seed: int,
        ) -> Optional[List[float]]:
            p_arr = np.asarray(p_vals, dtype=float)
            r_arr = np.asarray(r_vals, dtype=float)
            valid = np.isfinite(p_arr) & np.isfinite(r_arr)
            p_arr = p_arr[valid]
            r_arr = r_arr[valid]
            if p_arr.size == 0:
                return None
            if p_arr.size == 1:
                p_stat = float(np.mean(p_arr)) if stat_name == "mean" else float(np.median(p_arr))
                r_stat = float(np.mean(r_arr)) if stat_name == "mean" else float(np.median(r_arr))
                x = safe_f1(p_stat, r_stat)
                return None if x is None else [float(x), float(x)]
            rng = np.random.default_rng(seed)
            n = p_arr.size
            boot_vals = np.empty(int(n_boot), dtype=float)
            for i in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                ps = p_arr[idx]
                rs = r_arr[idx]
                p_stat = float(np.mean(ps)) if stat_name == "mean" else float(np.median(ps))
                r_stat = float(np.mean(rs)) if stat_name == "mean" else float(np.median(rs))
                f1 = safe_f1(p_stat, r_stat)
                boot_vals[i] = np.nan if f1 is None else float(f1)
            boot_vals = boot_vals[np.isfinite(boot_vals)]
            if boot_vals.size == 0:
                return None
            alpha = (1.0 - float(ci)) / 2.0
            lo, hi = np.quantile(boot_vals, [alpha, 1.0 - alpha])
            return [float(lo), float(hi)]

        def _macro_f1_payload_from_pr(
            p_vals: List[float],
            r_vals: List[float],
            summary_stat_name: str,
            n_boot: int,
            ci: float,
            seed: int,
            unit: str = "subject",
        ) -> Dict[str, Any]:
            n = len(p_vals)
            if summary_stat_name == "both":
                p_mean = float(np.mean(p_vals)) if n > 0 else None
                r_mean = float(np.mean(r_vals)) if n > 0 else None
                p_median = float(np.median(p_vals)) if n > 0 else None
                r_median = float(np.median(r_vals)) if n > 0 else None
                return {
                    "estimate": {
                        "mean": safe_f1(p_mean, r_mean),
                        "median": safe_f1(p_median, r_median),
                    },
                    "ci": {
                        "mean": _bootstrap_paired_macro_f1_ci(
                            p_vals, r_vals, "mean", n_boot, ci, seed
                        ),
                        "median": _bootstrap_paired_macro_f1_ci(
                            p_vals, r_vals, "median", n_boot, ci, seed + 1
                        ),
                    },
                    "n": n,
                    "statistic": "both",
                    "unit": unit,
                }

            stat_name = summary_stat_name
            if n == 0:
                est = None
            else:
                p_stat = float(np.mean(p_vals)) if stat_name == "mean" else float(np.median(p_vals))
                r_stat = float(np.mean(r_vals)) if stat_name == "mean" else float(np.median(r_vals))
                est = safe_f1(p_stat, r_stat)
            return {
                "estimate": est,
                "ci": _bootstrap_paired_macro_f1_ci(p_vals, r_vals, stat_name, n_boot, ci, seed),
                "n": n,
                "statistic": stat_name,
                "unit": unit,
            }

        tp_total = int(sum(m["cluster"][mode]["n_tp_pred_clusters"] for _, m in entries))
        fp_total = int(sum(m["cluster"][mode]["n_fp_pred_clusters"] for _, m in entries))
        n_pred_total = int(sum(m["cluster"]["n_pred_clusters"] for _, m in entries))
        n_detected_total = int(sum(m["cluster"][mode][det_key] for _, m in entries))
        n_gt_total = int(sum(m["cluster"]["n_gt_components"] for _, m in entries))

        pooled_precision = safe_div(tp_total, n_pred_total)
        pooled_recall = safe_div(n_detected_total, n_gt_total)
        pooled_f1 = safe_f1(pooled_precision, pooled_recall)

        return {
            "subject_macro": {
                "precision": _stat_payload(
                    subject_precision_values,
                    summary_stat,
                    n_bootstrap,
                    ci_level,
                    bootstrap_seed + seed_offset,
                ),
                "recall": _stat_payload(
                    subject_recall_values,
                    summary_stat,
                    n_bootstrap,
                    ci_level,
                    bootstrap_seed + seed_offset + 1,
                ),
                "f1": _macro_f1_payload_from_pr(
                    subject_precision_values,
                    subject_recall_values,
                    summary_stat,
                    n_bootstrap,
                    ci_level,
                    bootstrap_seed + seed_offset + 2,
                ),
                "cohort": {
                    "n_subjects": len(entries),
                },
            },
            "micro_cluster_bootstrap": {
                "counts": {
                    "tp_pred_clusters": tp_total,
                    "fp_pred_clusters": fp_total,
                    "n_pred_clusters": n_pred_total,
                    "detected_gt_components": n_detected_total,
                    "n_gt_components": n_gt_total,
                },
                "precision": {
                    "estimate": pooled_precision,
                    "ci": _cluster_bootstrap_subjects(
                        all_rows,
                        lambda rows: _cluster_micro(rows, mode, "precision"),
                        n_bootstrap,
                        ci_level,
                        bootstrap_seed + seed_offset + 10,
                    ),
                    "n": len(entries),
                    "statistic": "pooled_micro",
                    "unit": "subject",
                },
                "recall": {
                    "estimate": pooled_recall,
                    "ci": _cluster_bootstrap_subjects(
                        all_rows,
                        lambda rows: _cluster_micro(rows, mode, "recall"),
                        n_bootstrap,
                        ci_level,
                        bootstrap_seed + seed_offset + 11,
                    ),
                    "n": len(entries),
                    "statistic": "pooled_micro",
                    "unit": "subject",
                },
                "f1": {
                    "estimate": pooled_f1,
                    "ci": _cluster_bootstrap_subjects(
                        all_rows,
                        lambda rows: _cluster_micro(rows, mode, "f1"),
                        n_bootstrap,
                        ci_level,
                        bootstrap_seed + seed_offset + 12,
                    ),
                    "n": len(entries),
                    "statistic": "pooled_micro",
                    "unit": "subject",
                },
            },
            "subject_binary": {
                rec_key: {
                    "estimate": safe_div(
                        sum(1 for _, m in entries if bool(m["cluster"][mode].get(rec_key, False))),
                        len(entries),
                    ),
                    "ci": _bootstrap_binary_rate(
                        [1.0 if bool(m["cluster"][mode].get(rec_key, False)) else 0.0 for _, m in entries],
                        n_boot=n_bootstrap,
                        ci=ci_level,
                        seed=bootstrap_seed + seed_offset + 13,
                    ),
                    "n": len(entries),
                    "statistic": "mean",
                    "unit": "subject",
                }
            },
        }

        return out

    cluster_detection = _cluster_section("detection", 100)
    cluster_pinpointing = _cluster_section("pinpointing", 130)

    all_detect = [1.0 if bool(m["cluster"]["detection"].get("subject_detected", False)) else 0.0 for _, m in entries]
    all_pin = [1.0 if bool(m["cluster"]["pinpointing"].get("subject_pinpointed", False)) else 0.0 for _, m in entries]
    fp_cluster_counts = [float(m["cluster"]["detection"].get("n_fp_pred_clusters", 0.0)) for _, m in entries]
    ctrl_fp_subject = [1.0 if float(m["cluster"].get("n_pred_clusters", 0)) > 0 else 0.0 for _, m in controls]

    fp_rate_est = safe_div(sum(ctrl_fp_subject), len(ctrl_fp_subject)) if ctrl_fp_subject else None
    fp_rate_ci = _bootstrap_binary_rate(ctrl_fp_subject, n_bootstrap, ci_level, bootstrap_seed + 200)
    spec_est = None if fp_rate_est is None else (1.0 - fp_rate_est)
    spec_ci = None
    if fp_rate_ci is not None:
        spec_ci = [float(1.0 - fp_rate_ci[1]), float(1.0 - fp_rate_ci[0])]

    subject_rates = {
        "detection_rate_all_subjects": {
            "estimate": safe_div(sum(all_detect), len(all_detect)) if all_detect else None,
            "ci": _bootstrap_binary_rate(all_detect, n_bootstrap, ci_level, bootstrap_seed + 201),
            "n": len(all_detect),
            "statistic": "mean",
            "unit": "subject",
        },
        "pinpointing_rate_all_subjects": {
            "estimate": safe_div(sum(all_pin), len(all_pin)) if all_pin else None,
            "ci": _bootstrap_binary_rate(all_pin, n_bootstrap, ci_level, bootstrap_seed + 202),
            "n": len(all_pin),
            "statistic": "mean",
            "unit": "subject",
        },
        "mean_fp_clusters_per_subject": {
            "estimate": float(np.mean(fp_cluster_counts)) if fp_cluster_counts else None,
            "ci": _bootstrap_ci(fp_cluster_counts, "mean", n_bootstrap, ci_level, bootstrap_seed + 203),
            "n": len(fp_cluster_counts),
            "statistic": "mean",
            "unit": "subject",
        },
        "false_positive_subject_rate_controls": {
            "estimate": fp_rate_est,
            "ci": fp_rate_ci,
            "n": len(ctrl_fp_subject),
            "statistic": "mean",
            "unit": "subject",
        },
        "specificity_controls": {
            "estimate": spec_est,
            "ci": spec_ci,
            "n": len(ctrl_fp_subject),
            "statistic": "mean",
            "unit": "subject",
        },
        # Backward-compatible aliases
        "detection_rate_cases": {
            "estimate": safe_div(sum(all_detect), len(all_detect)) if all_detect else None,
            "ci": _bootstrap_binary_rate(all_detect, n_bootstrap, ci_level, bootstrap_seed + 201),
            "n": len(all_detect),
            "statistic": "mean",
            "unit": "subject",
        },
        "pinpointing_rate_cases": {
            "estimate": safe_div(sum(all_pin), len(all_pin)) if all_pin else None,
            "ci": _bootstrap_binary_rate(all_pin, n_bootstrap, ci_level, bootstrap_seed + 202),
            "n": len(all_pin),
            "statistic": "mean",
            "unit": "subject",
        },
    }

    # Compatibility view for downstream code expecting old keys.
    legacy_subject_level = {
        "cases": {
            "n": len(cases),
            "detection_rate": subject_rates["detection_rate_cases"]["estimate"],
            "pinpointing_rate": subject_rates["pinpointing_rate_cases"]["estimate"],
            "mean_pred_clusters_per_case": _nan_mean([float(m["cluster"].get("n_pred_clusters", 0)) for _, m in cases]),
            "median_pred_clusters_per_case": float(np.median([float(m["cluster"].get("n_pred_clusters", 0)) for _, m in cases])) if cases else None,
            "mean_fp_clusters_per_case": _nan_mean([float(m["cluster"]["detection"].get("n_fp_pred_clusters", 0)) for _, m in cases]),
            "median_fp_clusters_per_case": float(np.median([float(m["cluster"]["detection"].get("n_fp_pred_clusters", 0)) for _, m in cases])) if cases else None,
            "percent_empty_prediction_cases": safe_div(sum(1 for _, m in cases if m.get("pred_empty", False)) * 100.0, len(cases)),
        },
        "controls": {
            "n": len(controls),
            "specificity": subject_rates["specificity_controls"]["estimate"],
            "false_positive_subject_rate": subject_rates["false_positive_subject_rate_controls"]["estimate"],
            "mean_pred_clusters_per_control": _nan_mean([float(m["cluster"].get("n_pred_clusters", 0)) for _, m in controls]),
            "median_pred_clusters_per_control": float(np.median([float(m["cluster"].get("n_pred_clusters", 0)) for _, m in controls])) if controls else None,
            "percent_empty_prediction_controls": safe_div(sum(1 for _, m in controls if m.get("pred_empty", False)) * 100.0, len(controls)),
        },
    }

    return {
        "voxel_macro_all_subjects": voxel_macro_all_subjects,
        "voxel_macro_cases": voxel_macro_all_subjects,
        "voxel_micro_all_subjects": voxel_micro_all_subjects,
        "cluster_detection": cluster_detection,
        "cluster_pinpointing": cluster_pinpointing,
        "subject_rates": subject_rates,
        "legacy_subject_level": legacy_subject_level,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def evaluate_threshold_sweep(
    subject_records: List[SubjectRecord],
    thresholds: List[float],
    connectivity: int,
    min_cluster_size: int,
    box_dsc_threshold: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    For each threshold, re-evaluate all subjects and accumulate pooled metrics.

    Voxel PR counts are pooled over case subjects only (PR does not use TN and we
    want to avoid the huge TN count from controls dominating).
    FP-cluster burden is computed across ALL subjects.

    Returns a dict with four curve types:
    - voxel_pr
    - cluster_detection_pr
    - cluster_pinpoint_pr
    - detection_vs_fp_cluster_burden
    """
    voxel_pr_points: List[Dict[str, Any]] = []
    cluster_det_pr_points: List[Dict[str, Any]] = []
    cluster_pin_pr_points: List[Dict[str, Any]] = []
    det_fp_burden_points: List[Dict[str, Any]] = []

    for thr in tqdm(
        thresholds,
        desc="Threshold sweep",
        unit="thr",
    ):

        # Pooled voxel PR counts (cases only)
        v_tp = v_fp = v_fn = 0

        # Cluster counts pooled over all subjects.
        n_tp_det = n_fp_det = 0
        n_tp_pin = n_fp_pin = 0
        n_pred_all_clusters = 0
        n_detected_gt = 0
        n_gt_all = 0
        n_pinpointed_gt = 0

        # Subject rates (cases)
        n_cases_detected = 0
        n_cases_pinpointed = 0
        n_cases = 0

        # Controls
        n_controls = 0
        n_controls_empty = 0
        ctrl_pred_total = 0

        # All-subject FP burden
        fp_all = 0
        fp_cases = 0
        n_subjects = 0

        for rec in subject_records:
            m = evaluate_subject_at_threshold(
                rec, thr, connectivity, min_cluster_size, box_dsc_threshold
            )
            n_subjects += 1
            clust = m["cluster"]
            fp_all += clust["detection"]["n_fp_pred_clusters"]
            n_pred_all_clusters += clust["n_pred_clusters"]
            n_tp_det += clust["detection"]["n_tp_pred_clusters"]
            n_fp_det += clust["detection"]["n_fp_pred_clusters"]
            n_tp_pin += clust["pinpointing"]["n_tp_pred_clusters"]
            n_fp_pin += clust["pinpointing"]["n_fp_pred_clusters"]

            if not rec.is_control:
                v_tp += m["voxel"]["tp"]
                v_fp += m["voxel"]["fp"]
                v_fn += m["voxel"]["fn"]

                n_detected_gt += clust["detection"]["n_detected_gt_components"]
                n_gt_all += clust["n_gt_components"]

                n_pinpointed_gt += clust["pinpointing"]["n_pinpointed_gt_components"]

                if clust["detection"]["subject_detected"]:
                    n_cases_detected += 1
                if clust["pinpointing"]["subject_pinpointed"]:
                    n_cases_pinpointed += 1

                n_cases += 1
                fp_cases += clust["detection"]["n_fp_pred_clusters"]
            else:
                n_controls += 1
                if m["pred_empty"]:
                    n_controls_empty += 1
                ctrl_pred_total += clust["n_pred_clusters"]

        # Voxel PR
        vox_prec = safe_div(v_tp, v_tp + v_fp)
        vox_rec = safe_div(v_tp, v_tp + v_fn)

        # Cluster PR (detection)
        cdet_prec = safe_div(n_tp_det, n_pred_all_clusters)
        cdet_sens = safe_div(n_detected_gt, n_gt_all)

        # Cluster PR (pinpointing)
        cpin_prec = safe_div(n_tp_pin, n_pred_all_clusters)
        cpin_sens = safe_div(n_pinpointed_gt, n_gt_all)

        # Subject rates
        det_rate = safe_div(n_cases_detected, n_cases)
        pin_rate = safe_div(n_cases_pinpointed, n_cases)
        specificity = safe_div(n_controls_empty, n_controls)
        fp_subj_rate = safe_div(n_controls - n_controls_empty, n_controls)

        # FP burden
        mean_fp_per_subj = safe_div(fp_all, n_subjects)
        mean_fp_per_case = safe_div(fp_cases, n_cases)
        mean_pred_per_ctrl = safe_div(ctrl_pred_total, n_controls)

        voxel_pr_points.append({
            "threshold": thr,
            "recall": vox_rec,
            "precision": vox_prec,
            "dice": safe_div(2 * v_tp, 2 * v_tp + v_fp + v_fn),
            "tp": v_tp,
            "fp": v_fp,
            "fn": v_fn,
        })

        cluster_det_pr_points.append({
            "threshold": thr,
            "recall": cdet_sens,
            "precision": cdet_prec,
            "f1": safe_f1(cdet_prec, cdet_sens),
            "n_tp": n_tp_det,
            "n_fp": n_fp_det,
            "n_detected_gt": n_detected_gt,
            "n_gt": n_gt_all,
            "n_pred": n_pred_all_clusters,
        })

        cluster_pin_pr_points.append({
            "threshold": thr,
            "recall": cpin_sens,
            "precision": cpin_prec,
            "f1": safe_f1(cpin_prec, cpin_sens),
            "n_tp": n_tp_pin,
            "n_fp": n_fp_pin,
            "n_pinpointed_gt": n_pinpointed_gt,
            "n_gt": n_gt_all,
            "n_pred": n_pred_all_clusters,
        })

        det_fp_burden_points.append({
            "threshold": thr,
            "subject_detection_rate": det_rate,
            "subject_pinpointing_rate": pin_rate,
            "mean_fp_clusters_per_subject": mean_fp_per_subj,
            "mean_fp_clusters_per_case": mean_fp_per_case,
            "mean_pred_clusters_per_control": mean_pred_per_ctrl,
            "specificity": specificity,
            "false_positive_subject_rate": fp_subj_rate,
            "n_pred_clusters_total": n_pred_all_clusters,
            "n_fp_clusters_total": fp_all,
            "n_tp_clusters_total": n_tp_det,
        })

    return {
        "voxel_pr": {
            "auc": auc_pr(voxel_pr_points, "recall", "precision"),
            "points": voxel_pr_points,
        },
        "cluster_detection_pr": {
            "auc": auc_pr(cluster_det_pr_points, "recall", "precision"),
            "points": cluster_det_pr_points,
        },
        "cluster_pinpoint_pr": {
            "auc": auc_pr(cluster_pin_pr_points, "recall", "precision"),
            "points": cluster_pin_pr_points,
        },
        "detection_vs_fp_cluster_burden": {
            "points": det_fp_burden_points,
        },
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise, readable evaluation summary to the terminal."""
    dataset = results["dataset"]
    cfg = results["config"]
    agg = results["aggregate_fixed_threshold"]
    curves = results["curves"]

    def _fmt(v: Any, decimals: int = 3) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.{decimals}f}"

    def _fmt_ci(ci: Any, decimals: int = 3) -> str:
        if ci is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return "N/A"
        return f"[{_fmt(ci[0], decimals)}, {_fmt(ci[1], decimals)}]"

    print("\n=== Evaluation summary ===")
    print(f"Subjects evaluated: {dataset['n_subjects_evaluated']}")
    print(f"Cases:              {dataset['n_cases']}")
    print(f"Controls:           {dataset['n_controls']}")
    print(f"Skipped:            {dataset['n_skipped']}")

    vm = agg.get("voxel_macro_all_subjects", agg["voxel_macro_cases"])
    vmi = agg["voxel_micro_all_subjects"]
    cd = agg["cluster_detection"]
    cp = agg["cluster_pinpointing"]
    sr = agg["subject_rates"]

    thr = cfg["threshold"]
    print("\nPrimary estimates use one held-out prediction per subject.")
    print("Confidence intervals are non-parametric patient-level bootstrap CIs.")
    print("Folds are used only to identify held-out predictions.")
    print(f"\nFixed threshold: {thr:.3f}")

    def _metric_est_ci(block: Dict[str, Any]) -> Tuple[Any, Any]:
        return block.get("estimate"), block.get("ci")

    dice_est, dice_ci = _metric_est_ci(vm.get("voxel_dice", {}))
    prec_est, prec_ci = _metric_est_ci(vm.get("voxel_precision", {}))
    rec_est, rec_ci = _metric_est_ci(vm.get("voxel_recall", {}))

    print(f"Voxel Dice, macro all subjects:       {_fmt(dice_est)}")
    print(
        f"  Bootstrap CI:                        {_fmt_ci(dice_ci)}"
    )
    print(f"Voxel precision / recall, macro all subjects: {_fmt(prec_est)} / {_fmt(rec_est)}")
    print(
        f"  Bootstrap CI precision / recall:     "
        f"{_fmt_ci(prec_ci)} / {_fmt_ci(rec_ci)}"
    )
    print(
        f"Voxel Dice, micro pooled:             {_fmt(vmi.get('dice', {}).get('estimate'))}"
    )

    print("\nCluster detection criterion (boxDSC):")
    cd_macro = cd.get("subject_macro", {})
    print(f"  Precision:              {_fmt(cd_macro.get('precision', {}).get('estimate'))}")
    print(f"  Sensitivity:            {_fmt(cd_macro.get('recall', {}).get('estimate'))}")
    print(f"  F1:                     {_fmt(cd_macro.get('f1', {}).get('estimate'))}")
    print(
        f"  Precision bootstrap CI: {_fmt_ci(cd_macro.get('precision', {}).get('ci'))}"
    )
    print(
        f"  Recall bootstrap CI:    {_fmt_ci(cd_macro.get('recall', {}).get('ci'))}"
    )
    print(
        f"  F1 bootstrap CI:        {_fmt_ci(cd_macro.get('f1', {}).get('ci'))}"
    )
    print(f"  Macro cohort n (subjects): {cd_macro.get('cohort', {}).get('n_subjects', 'N/A')}")

    print("\nCluster pinpointing criterion (COM in GT):")
    cp_macro = cp.get("subject_macro", {})
    print(f"  Precision:              {_fmt(cp_macro.get('precision', {}).get('estimate'))}")
    print(f"  Sensitivity:            {_fmt(cp_macro.get('recall', {}).get('estimate'))}")
    print(f"  F1:                     {_fmt(cp_macro.get('f1', {}).get('estimate'))}")
    print(
        f"  Precision bootstrap CI: {_fmt_ci(cp_macro.get('precision', {}).get('ci'))}"
    )
    print(
        f"  Recall bootstrap CI:    {_fmt_ci(cp_macro.get('recall', {}).get('ci'))}"
    )
    print(
        f"  F1 bootstrap CI:        {_fmt_ci(cp_macro.get('f1', {}).get('ci'))}"
    )
    print(f"  Macro cohort n (subjects): {cp_macro.get('cohort', {}).get('n_subjects', 'N/A')}")

    print("\nSubject level:")
    print(f"  Detection rate:         {_fmt(sr.get('detection_rate_all_subjects', {}).get('estimate'))}")
    print(f"  Pinpointing rate:       {_fmt(sr.get('pinpointing_rate_all_subjects', {}).get('estimate'))}")
    print(f"  Mean FP clusters/patient: {_fmt(sr.get('mean_fp_clusters_per_subject', {}).get('estimate'))}")
    print(f"  Specificity (controls): {_fmt(sr.get('specificity_controls', {}).get('estimate'))}")

    vpr = curves.get("voxel_pr", {})
    cdr = curves.get("cluster_detection_pr", {})
    cpr = curves.get("cluster_pinpoint_pr", {})
    dfp = curves.get("detection_vs_fp_cluster_burden", {})

    print("\nCurves:")
    print(f"  Voxel PR-AUC:              {_fmt(vpr.get('auc'))}")
    print(f"  Cluster detection PR-AUC:  {_fmt(cdr.get('auc'))}")
    print(f"  Cluster pinpoint PR-AUC:   {_fmt(cpr.get('auc'))}")
    print(
        f"  Detection-vs-FP-burden points saved: "
        f"{len(dfp.get('points', []))}"
    )
    print()


def save_json(results: Dict[str, Any], out_json: Path) -> None:
    """Serialise results to a JSON file, converting all numpy types."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    safe = ensure_json_serializable(results)
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    print(f"Results saved to: {out_json}")


def _resolve_aggregate_nifti_reference(
    subject_records: Sequence[SubjectRecord],
    warnings_list: List[str],
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Resolve a shared grid and affine for aggregate count-map export."""
    if not subject_records:
        raise ValueError("No subject records available for aggregate NIfTI export.")

    ref_shape = tuple(int(v) for v in subject_records[0].meta.shape)
    ref_affine: Optional[np.ndarray] = None

    for record in subject_records:
        if tuple(int(v) for v in record.meta.shape) != ref_shape:
            raise ValueError(
                "Aggregate NIfTI export requires all evaluated subjects to share "
                f"one grid. Found shapes {ref_shape} and {record.meta.shape}."
            )
        if record.meta.affine is not None and ref_affine is None:
            ref_affine = np.asarray(record.meta.affine, dtype=np.float32)

    if ref_affine is None:
        warnings_list.append(
            "[aggregate-nifti] No affine metadata available across evaluated subjects; "
            "writing aggregate maps with identity affine."
        )
        ref_affine = np.eye(4, dtype=np.float32)
    else:
        for record in subject_records:
            if record.meta.affine is None:
                continue
            if not np.allclose(record.meta.affine, ref_affine, atol=1e-4):
                raise ValueError(
                    "Aggregate NIfTI export requires all evaluated subjects to share "
                    "one affine. Found mismatched affines across subjects."
                )

    return ref_shape, ref_affine


def export_fixed_threshold_aggregate_niftis(
    subject_records: Sequence[SubjectRecord],
    per_subject_fixed: Sequence[Dict[str, Any]],
    threshold: float,
    connectivity: int,
    min_cluster_size: int,
    out_dir: Path,
    warnings_list: List[str],
) -> Dict[str, str]:
    """
    Export aggregate count maps for GT, thresholded predictions, and detection FPs.

    The false-positive map uses the cluster-level detection criterion at the fixed
    threshold: only voxels belonging to predicted clusters classified as detection
    false positives are accumulated.
    """
    if len(subject_records) != len(per_subject_fixed):
        raise ValueError("subject_records and per_subject_fixed must have same length")

    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to export aggregate NIfTI maps."
        ) from exc

    ref_shape, ref_affine = _resolve_aggregate_nifti_reference(
        subject_records, warnings_list
    )
    gt_counts = np.zeros(ref_shape, dtype=np.int32)
    pred_counts = np.zeros(ref_shape, dtype=np.int32)
    fp_counts = np.zeros(ref_shape, dtype=np.int32)

    for record, fixed_metrics in zip(subject_records, per_subject_fixed):
        pred_bin = binarize_prediction(record.pred_soft, threshold)
        pred_counts += pred_bin.astype(np.int32)
        gt_counts += record.gt_bin.astype(np.int32)

        pred_label_img, _ = extract_components(
            pred_bin.astype(bool),
            record.meta.voxel_volume_mm3,
            connectivity,
            min_cluster_size,
        )
        fp_cluster_ids = [
            int(cluster["pred_cluster_id"])
            for cluster in fixed_metrics["cluster"].get("predicted_clusters", [])
            if bool(cluster.get("is_fp_detection_cluster", False))
        ]
        if fp_cluster_ids:
            fp_counts += np.isin(pred_label_img, fp_cluster_ids).astype(np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "aggregate_gt_counts.nii.gz"
    pred_path = out_dir / "aggregate_prediction_counts.nii.gz"
    fp_path = out_dir / "aggregate_false_positive_counts.nii.gz"

    nib.save(nib.Nifti1Image(gt_counts.astype(np.int32), ref_affine), str(gt_path))
    nib.save(nib.Nifti1Image(pred_counts.astype(np.int32), ref_affine), str(pred_path))
    nib.save(nib.Nifti1Image(fp_counts.astype(np.int32), ref_affine), str(fp_path))

    print(f"Aggregate GT count map saved to: {gt_path}")
    print(f"Aggregate prediction count map saved to: {pred_path}")
    print(f"Aggregate false-positive count map saved to: {fp_path}")

    return {
        "gt_counts": str(gt_path),
        "prediction_counts": str(pred_path),
        "false_positive_counts": str(fp_path),
    }


def save_subject_table(per_subject_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write one-row-per-subject flat table for downstream stats/plotting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject_id",
        "fold_id",
        "split_role",
        "prediction_path",
        "gt_path",
        "is_control",
        "gt_empty",
        "voxel_dice",
        "voxel_precision",
        "voxel_recall",
        "cluster_det_precision",
        "cluster_det_recall",
        "cluster_det_f1",
        "cluster_pin_precision",
        "cluster_pin_recall",
        "cluster_pin_f1",
        "subject_detected",
        "subject_pinpointed",
        "n_pred_clusters",
        "n_fp_det_clusters",
        "n_fp_pin_clusters",
        "voxel_tp",
        "voxel_fp",
        "voxel_fn",
        "voxel_tn",
    ]

    with open(str(out_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_subject_rows:
            fm = row.get("fixed_threshold_metrics", {})
            vox = fm.get("voxel", {})
            cl = fm.get("cluster", {})
            det = cl.get("detection", {})
            pin = cl.get("pinpointing", {})
            writer.writerow(
                {
                    "subject_id": row.get("subject_id"),
                    "fold_id": row.get("fold_id"),
                    "split_role": row.get("split_role"),
                    "prediction_path": row.get("prediction_path"),
                    "gt_path": row.get("gt_path"),
                    "is_control": row.get("is_control"),
                    "gt_empty": fm.get("gt_empty"),
                    "voxel_dice": vox.get("voxel_dice"),
                    "voxel_precision": vox.get("voxel_precision"),
                    "voxel_recall": vox.get("voxel_recall"),
                    "cluster_det_precision": det.get("cluster_precision"),
                    "cluster_det_recall": det.get("cluster_sensitivity"),
                    "cluster_det_f1": det.get("cluster_f1"),
                    "cluster_pin_precision": pin.get("cluster_precision"),
                    "cluster_pin_recall": pin.get("cluster_sensitivity"),
                    "cluster_pin_f1": pin.get("cluster_f1"),
                    "subject_detected": det.get("subject_detected"),
                    "subject_pinpointed": pin.get("subject_pinpointed"),
                    "n_pred_clusters": cl.get("n_pred_clusters"),
                    "n_fp_det_clusters": det.get("n_fp_pred_clusters"),
                    "n_fp_pin_clusters": pin.get("n_fp_pred_clusters"),
                    "voxel_tp": vox.get("tp"),
                    "voxel_fp": vox.get("fp"),
                    "voxel_fn": vox.get("fn"),
                    "voxel_tn": vox.get("tn"),
                }
            )

    print(f"Subject table saved to: {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 3D soft FCD prediction maps against binary ground-truth masks. "
            "Computes voxel-level, cluster-level, and subject-level metrics."
        )
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        type=Path,
        help="Directory containing prediction maps.",
    )
    parser.add_argument(
        "--out_json",
        required=True,
        type=Path,
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--gt_dir",
        default=GT_DIR_DEFAULT,
        type=Path,
        help=(
            "Directory containing ground-truth masks. "
            "Defaults to <data_root>/preprocessing/mri from config.json when omitted."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed binarization threshold (default: 0.5).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Explicit list of thresholds for curves (overrides --threshold_min/max/steps).",
    )
    parser.add_argument(
        "--threshold_min",
        type=float,
        default=0.01,
        help="Lower bound for threshold sweep linspace (default: 0.01).",
    )
    parser.add_argument(
        "--threshold_max",
        type=float,
        default=0.99,
        help="Upper bound for threshold sweep linspace (default: 0.99).",
    )
    parser.add_argument(
        "--threshold_steps",
        type=int,
        default=10,
        help="Number of thresholds in linspace sweep (default: 10).",
    )
    parser.add_argument(
        "--box_dsc_threshold",
        type=float,
        default=0.22,
        help=(
            "Minimum bounding-box Dice for a predicted cluster to count as a "
            "detection match to a GT component (default: 0.22)."
        ),
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=0,
        help=(
            "Remove predicted clusters with fewer voxels than this before "
            "cluster-level evaluation (default: 0, i.e. keep all clusters)."
        ),
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=26,
        choices=[6, 18, 26],
        help="3-D connected-component connectivity: 6, 18, or 26 (default: 26).",
    )
    parser.add_argument(
        "--allow_missing_gt_as_control",
        action="store_true",
        help=(
            "If set, subjects without a GT mask file are treated as "
            "healthy controls (empty GT)."
        ),
    )
    parser.add_argument(
        "--treat_empty_gt_as_control",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help=(
            "If GT mask exists but contains no positive voxels, treat the "
            "subject as a control (default: True)."
        ),
    )
    parser.add_argument(
        "--save_per_subject",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help="Include per-subject details in the JSON output (default: True).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    parser.add_argument(
        "--pred_pattern",
        type=str,
        default="*",
        help=(
            "Glob pattern for prediction files under --pred-dir "
            "(e.g. '*.nii.gz', '*.npz'). Default: '*' (any supported format). "
            "Only prediction paths containing 'val' or 'validation' and not "
            "'train' are evaluated."
        ),
    )
    parser.add_argument(
        "--nnunet_preds",
        action="store_true",
        help=(
            "Enable nnUNet fold-mode prediction discovery: find fold_<idx> "
            "directories and keep only val_ids per fold from k_fold_splits.json. "
            "Deprecated legacy alias for --fold-role val."
        ),
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help=(
            "Enable test-set fold discovery: --pred_dir must contain fold_<idx> "
            "directories directly, and prediction files are searched recursively "
            "under each fold directory. Deprecated legacy alias for --fold-role test."
        ),
    )
    parser.add_argument(
        "--fold_split_json",
        type=Path,
        default=None,
        help=(
            "Path to fold split JSON containing per-fold val_ids/test_ids. "
            "Defaults to the repository k_fold_splits.json path when omitted."
        ),
    )
    parser.add_argument(
        "--fold_role",
        type=str,
        choices=["test", "val", "all"],
        default="test",
        help=(
            "Split role to select per fold. Primary mode is test: one held-out "
            "test prediction per subject."
        ),
    )
    parser.add_argument(
        "--gt_pattern",
        type=str,
        default=GT_PATTERN_DEFAULT,
        help=(
            "Glob pattern for GT mask files under --gt-dir "
            "(e.g. '*_gt_norm.nii.gz'). Default: '*_gt_norm*'."
        ),
    )
    parser.add_argument(
        "--subject_regex",
        type=str,
        default=None,
        help=(
            "Optional regex with a capture group for extracting subject ID "
            "from filenames. Default: uses RESP#### codebase convention."
        ),
    )
    parser.add_argument(
        "--subjects_list",
        type=Path,
        default=None,
        help=(
            "Optional TXT file with one subject ID per line. When provided, only "
            "these subjects are evaluated and the script fails if any listed "
            "subject is missing prediction/GT according to current settings."
        ),
    )
    parser.add_argument(
        "--prob_key",
        type=str,
        default=None,
        help=(
            "Key to load from .npz prediction files (auto-detected when omitted). "
            "Common choices: prob, probs, prediction, pred, arr_0."
        ),
    )
    parser.add_argument(
        "--resample_to_gt",
        action="store_true",
        help=(
            "If prediction and GT shapes differ, resample prediction to the GT grid "
            "using linear interpolation.  Disabled by default; enable with care."
        ),
    )
    parser.add_argument(
        "--skip_threshold_sweep",
        action="store_true",
        help="Skip threshold sweep and compute only fixed-threshold metrics.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap replicates for patient-level CIs (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap resampling (default: 12345).",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CIs (default: 0.95).",
    )
    parser.add_argument(
        "--summary_stat",
        type=str,
        choices=["mean", "median", "both"],
        default="mean",
        help="Summary statistic for macro subject metrics (default: mean).",
    )
    parser.add_argument(
        "--out_subject_csv",
        type=Path,
        default=None,
        help="Optional path to export one-row-per-subject flat metrics CSV.",
    )
    parser.add_argument(
        "--save_aggregate_niftis",
        action="store_true",
        help=(
            "If set, export aggregate NIfTI count maps for GT, thresholded "
            "predictions, and detection-level false-positive clusters."
        ),
    )
    parser.add_argument(
        "--aggregate_nifti_dir",
        type=Path,
        default=None,
        help="Output directory for aggregate NIfTI exports.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    warnings_list: List[str] = []
    skipped: List[Dict[str, str]] = []
    required_subject_ids: Optional[List[str]] = None
    aggregate_nifti_paths: Dict[str, str] = {}

    if args.save_aggregate_niftis and args.aggregate_nifti_dir is None:
        raise SystemExit(
            "ERROR: --aggregate_nifti_dir must be provided when "
            "--save_aggregate_niftis is set."
        )

    if args.subjects_list is not None:
        try:
            required_subject_ids = load_subjects_list(
                subjects_list_path=args.subjects_list,
                subject_regex=args.subject_regex,
            )
        except Exception as exc:
            raise SystemExit(f"ERROR: {exc}")
        print(
            f"Loaded {len(required_subject_ids)} required subject IDs from "
            f"{args.subjects_list}."
        )

    # ------------------------------------------------------------------ #
    # 1. Discover prediction and GT files
    # ------------------------------------------------------------------ #
    effective_fold_role = args.fold_role
    if args.nnunet_preds and args.test_set:
        warnings_list.append(
            "Both --nnunet-preds and --test-set were set. --test-set takes precedence."
        )
    if args.nnunet_preds and args.fold_role == "test":
        effective_fold_role = "val"
        warnings_list.append(
            "Legacy --nnunet-preds detected; using fold-role 'val'."
        )
    if args.test_set:
        effective_fold_role = "test"

    if effective_fold_role == "all":
        warnings_list.append(
            "fold-role=all selected; this is not appropriate for primary performance reporting."
        )

    if args.verbose:
        print(
            f"Discovering files...\n"
            f"  pred_dir : {args.pred_dir}\n"
            f"  gt_dir   : {args.gt_dir}\n"
            f"  fold_role: {effective_fold_role}"
        )

    try:
        matched, file_skipped = discover_subject_files(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            pred_pattern=args.pred_pattern,
            gt_pattern=args.gt_pattern,
            subject_regex=args.subject_regex,
            allow_missing_gt_as_control=args.allow_missing_gt_as_control,
            warnings_list=warnings_list,
            fold_split_json=args.fold_split_json,
            fold_role=effective_fold_role,
            nnunet_preds=args.nnunet_preds,
            test_set=args.test_set,
            required_subject_ids=required_subject_ids,
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")
    skipped.extend(file_skipped)

    if not matched:
        if args.test_set:
            hint = (
                "Check that --pred-dir contains fold_<idx> directories directly and "
                "that prediction files exist recursively inside each fold directory."
            )
        elif args.nnunet_preds:
            hint = "Check fold_<idx> directories exist and val_ids are populated in k_fold_splits.json."
        else:
            hint = (
                "Prediction paths must include 'val' or 'validation' and must not include 'train'. "
                "Use --nnunet-preds for fold-based val discovery or --test-set for test-set folds."
            )
        print(
            "ERROR: No matched subject files found. "
            f"Check --pred_dir, --gt_dir, and --pred-pattern / --gt-pattern. {hint}"
        )
        return

    if args.verbose:
        print(f"Found {len(matched)} subjects after file matching.")

    # ------------------------------------------------------------------ #
    # 2. Load and preprocess each subject
    # ------------------------------------------------------------------ #
    subject_records: List[SubjectRecord] = []

    for match in tqdm(
        matched, desc="Loading subjects", unit="subject"
    ):
        sid = match.subject_id
        pred_path = match.pred_path
        gt_path = match.gt_path
        if args.verbose:
            tqdm.write(
                f"  Loading {sid}: pred={pred_path.name}, "
                f"gt={gt_path.name if gt_path else 'None (control)'}"
            )

        # Load prediction
        try:
            pred_raw, pred_meta = load_volume(pred_path, prob_key=args.prob_key)
        except Exception as exc:
            reason = f"Failed to load prediction from {pred_path}: {exc}"
            skipped.append({"subject_id": sid, "reason": reason})
            warnings_list.append(f"[{sid}] {reason}")
            continue

        if pred_path.name.lower().endswith(".npz"):
            ref_pred_path = _find_paired_reference_nifti(
                pred_path=pred_path,
                subject_id=sid,
                pred_root=args.pred_dir,
            )
            if ref_pred_path is not None:
                try:
                    ref_pred_raw, ref_pred_meta = load_volume(ref_pred_path)
                    aligned_pred, transform_desc, transform_score = _reorient_npz_to_reference(
                        pred_raw,
                        ref_pred_raw,
                    )
                    if aligned_pred.shape == ref_pred_raw.shape:
                        pred_raw = aligned_pred
                        pred_meta = ref_pred_meta
                        if transform_desc != "identity" or args.verbose:
                            warnings_list.append(
                                f"[{sid}] Matched NPZ prediction orientation to paired NIfTI "
                                f"{ref_pred_path.name} using {transform_desc} "
                                f"(similarity={transform_score:.6f})."
                            )
                    else:
                        warnings_list.append(
                            f"[{sid}] Found paired NIfTI {ref_pred_path.name} for NPZ prediction, "
                            "but no compatible axis permutation/flip produced matching shape."
                        )
                except Exception as exc:
                    warnings_list.append(
                        f"[{sid}] Failed to align NPZ prediction to paired NIfTI "
                        f"{ref_pred_path}: {exc}"
                    )
            elif args.verbose:
                warnings_list.append(
                    f"[{sid}] NPZ prediction selected but no paired NIfTI reference was found "
                    f"under {args.pred_dir}."
                )

        # Load GT (or create empty GT for controls with missing GT file)
        if gt_path is not None:
            try:
                gt_raw, gt_meta = load_volume(gt_path)
                gt_bin = gt_raw > 0
            except Exception as exc:
                reason = f"Failed to load GT from {gt_path}: {exc}"
                skipped.append({"subject_id": sid, "reason": reason})
                warnings_list.append(f"[{sid}] {reason}")
                continue
        else:
            gt_bin = np.zeros(pred_raw.shape, dtype=bool)
            gt_meta = VolumeMeta(
                shape=pred_raw.shape,
                affine=pred_meta.affine,
                voxel_spacing=pred_meta.voxel_spacing,
                voxel_volume_mm3=pred_meta.voxel_volume_mm3,
            )

        # Resample or check shape mismatch
        try:
            pred_raw, pred_meta = maybe_resample_to_gt(
                pred=pred_raw,
                pred_meta=pred_meta,
                gt_shape=gt_bin.shape,
                gt_meta=gt_meta,
                resample=args.resample_to_gt,
                warnings_list=warnings_list,
                subject_id=sid,
            )
        except ValueError as exc:
            skipped.append({"subject_id": sid, "reason": str(exc)})
            continue

        # Preprocess prediction
        pred_soft = preprocess_prediction(pred_raw, sid, warnings_list)

        # Determine control status
        gt_empty = not np.any(gt_bin)
        is_control = (gt_path is None) or (
            gt_empty and args.treat_empty_gt_as_control
        )

        subject_records.append(
            SubjectRecord(
                subject_id=sid,
                pred_path=str(pred_path),
                fold_id=match.fold_id or _extract_fold_id(str(pred_path)),
                split_role=match.split_role,
                gt_path=str(gt_path) if gt_path else None,
                is_control=is_control,
                pred_soft=pred_soft,
                gt_bin=gt_bin.astype(bool),
                meta=pred_meta,
            )
        )

    if not subject_records:
        print(
            "ERROR: No subjects could be loaded successfully. "
            "Check file formats and subject ID extraction."
        )
        return

    n_cases_loaded = sum(1 for r in subject_records if not r.is_control)
    n_controls_loaded = sum(1 for r in subject_records if r.is_control)
    print(
        f"Loaded {len(subject_records)} subjects "
        f"({n_cases_loaded} cases, {n_controls_loaded} controls)."
    )

    # ------------------------------------------------------------------ #
    # 3. Evaluate at fixed threshold
    # ------------------------------------------------------------------ #
    print(f"Evaluating at fixed threshold {args.threshold:.3f} ...")

    per_subject_fixed: List[Dict[str, Any]] = []
    for rec in tqdm(
        subject_records,
        desc=f"Fixed-threshold eval (thr={args.threshold:.3f})",
        unit="subject",
    ):
        m = evaluate_subject_at_threshold(
            rec,
            args.threshold,
            args.connectivity,
            args.min_cluster_size,
            args.box_dsc_threshold,
        )
        per_subject_fixed.append(m)

    agg = aggregate_fixed_metrics(
        per_subject_fixed=per_subject_fixed,
        subject_records=subject_records,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
        summary_stat=args.summary_stat,
    )

    if args.save_aggregate_niftis:
        aggregate_nifti_paths = export_fixed_threshold_aggregate_niftis(
            subject_records=subject_records,
            per_subject_fixed=per_subject_fixed,
            threshold=args.threshold,
            connectivity=args.connectivity,
            min_cluster_size=args.min_cluster_size,
            out_dir=args.aggregate_nifti_dir,
            warnings_list=warnings_list,
        )

    # ------------------------------------------------------------------ #
    # 4. Build per-subject JSON entries
    # ------------------------------------------------------------------ #
    per_subject_out: List[Dict[str, Any]] = []
    if args.save_per_subject:
        for rec, fixed_m in tqdm(
            zip(subject_records, per_subject_fixed),
            desc="Building per-subject JSON",
            unit="subject",
            total=len(subject_records),
        ):
            # Separate verbose cluster lists from compact metrics
            cluster_section = dict(fixed_m["cluster"])
            predicted_clusters = cluster_section.pop("predicted_clusters", [])
            gt_comp_details = cluster_section.pop("gt_components", [])

            fixed_metrics = {
                k: v for k, v in fixed_m.items() if k != "cluster"
            }
            fixed_metrics["cluster"] = cluster_section

            per_subject_out.append({
                "subject_id": rec.subject_id,
                "fold_id": rec.fold_id,
                "split_role": rec.split_role,
                "prediction_path": rec.pred_path,
                "gt_path": rec.gt_path,
                "is_control": rec.is_control,
                "shape": list(rec.meta.shape),
                "voxel_volume_mm3": rec.meta.voxel_volume_mm3,
                "fixed_threshold_metrics": fixed_metrics,
                "clusters": {
                    "predicted": predicted_clusters,
                    "gt_components": gt_comp_details,
                },
            })

    # ------------------------------------------------------------------ #
    # 5. Threshold sweep
    # ------------------------------------------------------------------ #
    if args.thresholds is not None:
        sweep_thresholds = sorted(set(args.thresholds))
    else:
        sweep_thresholds = list(
            np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
        )

    if args.skip_threshold_sweep:
        if args.verbose:
            print("Skipping threshold sweep (--skip-threshold-sweep).")
        curves = {
            "voxel_pr": {"auc": None, "points": []},
            "cluster_detection_pr": {"auc": None, "points": []},
            "cluster_pinpoint_pr": {"auc": None, "points": []},
            "detection_vs_fp_cluster_burden": {"points": []},
        }
    else:
        print(
            f"Running threshold sweep over {len(sweep_thresholds)} thresholds "
            f"({sweep_thresholds[0]:.3f} .. {sweep_thresholds[-1]:.3f}) ..."
        )
        curves = evaluate_threshold_sweep(
            subject_records=subject_records,
            thresholds=sweep_thresholds,
            connectivity=args.connectivity,
            min_cluster_size=args.min_cluster_size,
            box_dsc_threshold=args.box_dsc_threshold,
            verbose=args.verbose,
        )

    # ------------------------------------------------------------------ #
    # 6. Assemble and save results
    # ------------------------------------------------------------------ #
    results: Dict[str, Any] = {
        "config": {
            "pred_dir": str(args.pred_dir),
            "gt_dir": str(args.gt_dir),
            "threshold": args.threshold,
            "box_dsc_threshold": args.box_dsc_threshold,
            "min_cluster_size": args.min_cluster_size,
            "connectivity": args.connectivity,
            "allow_missing_gt_as_control": args.allow_missing_gt_as_control,
            "treat_empty_gt_as_control": args.treat_empty_gt_as_control,
            "resample_to_gt": args.resample_to_gt,
            "pred_pattern": args.pred_pattern,
            "nnunet_preds": args.nnunet_preds,
            "test_set": args.test_set,
            "fold_split_json": str(args.fold_split_json) if args.fold_split_json else None,
            "fold_role": effective_fold_role,
            "gt_pattern": args.gt_pattern,
            "subject_regex": args.subject_regex,
            "subjects_list": str(args.subjects_list) if args.subjects_list else None,
            "n_subjects_requested": len(required_subject_ids) if required_subject_ids is not None else None,
            "thresholds": sweep_thresholds,
            "skip_threshold_sweep": args.skip_threshold_sweep,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "ci_level": args.ci_level,
            "summary_stat": args.summary_stat,
            "save_aggregate_niftis": args.save_aggregate_niftis,
            "aggregate_nifti_dir": str(args.aggregate_nifti_dir) if args.aggregate_nifti_dir else None,
        },
        "inference_unit": "subject",
        "ci_method": "patient_bootstrap",
        "bootstrap": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.bootstrap_seed,
            "ci_level": args.ci_level,
        },
        "dataset": {
            "n_subjects_evaluated": len(subject_records),
            "n_cases": n_cases_loaded,
            "n_controls": n_controls_loaded,
            "n_skipped": len(skipped),
        },
        "aggregate_fixed_threshold": agg,
        "aggregate_niftis": aggregate_nifti_paths,
        "curves": curves,
        "per_subject": per_subject_out if args.save_per_subject else [],
        "skipped": skipped,
        "warnings": warnings_list,
    }

    print_summary(results)
    save_json(results, args.out_json)
    if args.out_subject_csv is not None:
        save_subject_table(per_subject_out, args.out_subject_csv)


if __name__ == "__main__":
    main()
