"""
evaluate/evaluate_eeg_maps.py

Aggregate per-subject metrics from fold validation.json / predictions.json files
across all cross-validation folds and report mean + 95% CI.

Optionally exports aggregate NIfTI maps (GT voxel counts and summed prediction maps).

Usage:
    python -m evaluate.evaluate_eeg_maps <runs_root> [--test_set] [--subjects_list FILE]
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import nibabel as nib
import numpy as np

from util.config import get_data_root

FOLD_PATTERN = re.compile(r"_fold\d+_")
_data_root = get_data_root()
PRED_DIR_DEFAULT: Optional[Path] = (
    _data_root / "results" / "runs" / "eeg_new" if _data_root else None
)
GT_DIR_DEFAULT: Optional[Path] = (
    _data_root / "preprocessing" / "mri" if _data_root else None
)
SUBJECT_ID_REGEX_DEFAULT = re.compile(r"(RESP\d+)", re.IGNORECASE)
GT_SUFFIX = "_gt_norm.nii.gz"
PRIOR_SUFFIXES: Tuple[str, ...] = ("_deconv_prior.nii.gz", "_prior.nii.gz")

def mean_ci95(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance) / math.sqrt(n)
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def fold_mean_ci95(per_fold_values: Dict[str, List[float]]) -> Tuple[float, float, float]:
    fold_means = [sum(values) / len(values) for values in per_fold_values.values() if values]
    return mean_ci95(fold_means)


def is_numeric(value):
    # bool is a subclass of int; skip it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def collect_numeric_metrics(metrics, prefix=""):
    collected = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key
        if is_numeric(value):
            collected[metric_name] = float(value)
        elif isinstance(value, dict):
            collected.update(collect_numeric_metrics(value, prefix=f"{metric_name}."))
    return collected


def find_fold_dirs(root_dir):
    return sorted(
        p
        for p in root_dir.iterdir()
        if p.is_dir() and FOLD_PATTERN.search(p.name)
    )


def normalize_subject_id(token: str) -> Optional[str]:
    token = str(token).strip()
    if not token or token.startswith("#"):
        return None

    match = SUBJECT_ID_REGEX_DEFAULT.search(token)
    if match:
        return match.group(1).upper()
    return token.upper()


def load_subjects_list(subjects_list_path: Path) -> List[str]:
    if not subjects_list_path.exists() or not subjects_list_path.is_file():
        raise FileNotFoundError(f"subjects_list file not found: {subjects_list_path}")

    subject_ids: List[str] = []
    seen: Set[str] = set()

    with subjects_list_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            sid = normalize_subject_id(line)
            if sid is None:
                continue
            if sid == "":
                raise ValueError(
                    f"Invalid empty subject ID in {subjects_list_path} at line {line_no}."
                )
            if sid in seen:
                continue
            seen.add(sid)
            subject_ids.append(sid)

    if not subject_ids:
        raise ValueError(
            f"subjects_list file {subjects_list_path} did not contain any valid subject IDs."
        )

    return subject_ids


def load_case_metrics(fold_dir, use_test_set):
    if use_test_set:
        predictions_path = fold_dir / "predictions.json"
        if not predictions_path.exists():
            return None

        with predictions_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("test", {})
        if not isinstance(cases, dict):
            return None
        return cases

    validation_path = fold_dir / "validation.json"
    if not validation_path.exists():
        return None

    with validation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", {})
    if not isinstance(cases, dict):
        return None
    return cases


def _extract_subject_id_from_suffix(path: Path, suffix: str) -> Optional[str]:
    name = path.name
    if not name.endswith(suffix):
        return None
    return normalize_subject_id(name[: -len(suffix)])


def index_gt_files(gt_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(gt_dir.rglob(f"*{GT_SUFFIX}")):
        sid = _extract_subject_id_from_suffix(path, GT_SUFFIX)
        if sid is not None and sid not in mapping:
            mapping[sid] = path
    return mapping


def index_prior_files(fold_dir: Path, split_dir_token: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    prior_root = fold_dir / "prior_niftis"
    if not prior_root.is_dir():
        return mapping

    split_dir_token = split_dir_token.lower()
    for suffix in PRIOR_SUFFIXES:
        for path in sorted(prior_root.rglob(f"*{suffix}")):
            path_parts = [part.lower() for part in path.parts]
            if split_dir_token not in path_parts:
                continue
            sid = _extract_subject_id_from_suffix(path, suffix)
            if sid is not None and sid not in mapping:
                mapping[sid] = path
    return mapping


def load_nifti_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    return data, affine


def _validate_shared_grid(
    reference_shape: Optional[Tuple[int, ...]],
    reference_affine: Optional[np.ndarray],
    current_shape: Tuple[int, ...],
    current_affine: np.ndarray,
    current_path: Path,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    if reference_shape is None or reference_affine is None:
        return current_shape, current_affine
    if current_shape != reference_shape:
        raise ValueError(
            "Aggregate EEG NIfTI export requires all maps to share one grid. "
            f"Found shapes {reference_shape} and {current_shape} ({current_path})."
        )
    if not np.allclose(current_affine, reference_affine, atol=1e-4):
        raise ValueError(
            "Aggregate EEG NIfTI export requires all maps to share one affine. "
            f"Found mismatch at {current_path}."
        )
    return reference_shape, reference_affine


def filter_cases_by_subjects(
    cases: Dict[str, dict],
    required_subject_ids: Optional[List[str]],
) -> Tuple[Dict[str, dict], Set[str]]:
    if required_subject_ids is None:
        normalized_keys = {
            sid for sid in (normalize_subject_id(k) for k in cases.keys()) if sid is not None
        }
        return cases, normalized_keys

    case_by_subject: Dict[str, dict] = {}
    for key, value in cases.items():
        sid = normalize_subject_id(key)
        if sid is None:
            continue
        if sid not in case_by_subject and isinstance(value, dict):
            case_by_subject[sid] = value

    filtered: Dict[str, dict] = {}
    for sid in required_subject_ids:
        if sid in case_by_subject:
            filtered[sid] = case_by_subject[sid]

    return filtered, set(case_by_subject.keys())


def aggregate_metrics(
    runs_root,
    use_test_set=False,
    required_subject_ids: Optional[List[str]] = None,
):
    metric_values = defaultdict(list)
    metric_values_by_fold = defaultdict(lambda: defaultdict(list))
    total_case_rows = 0
    used_metric_files = 0
    found_required_subject_ids: Set[str] = set()
    evaluated_subject_rows: List[Tuple[str, Path]] = []

    for fold_dir in find_fold_dirs(runs_root):
        cases = load_case_metrics(fold_dir, use_test_set=use_test_set)
        if not cases:
            continue

        cases, fold_subject_ids = filter_cases_by_subjects(
            cases,
            required_subject_ids=required_subject_ids,
        )
        if required_subject_ids is not None:
            found_required_subject_ids.update(
                sid for sid in required_subject_ids if sid in fold_subject_ids
            )
            if not cases:
                continue

        used_metric_files += 1

        for case_key, case_metrics in cases.items():
            if not isinstance(case_metrics, dict):
                continue
            sid = normalize_subject_id(case_key)
            if sid is None:
                continue
            total_case_rows += 1
            evaluated_subject_rows.append((sid, fold_dir))

            for metric_name, value in collect_numeric_metrics(case_metrics).items():
                metric_values[metric_name].append(value)
                metric_values_by_fold[metric_name][fold_dir.name].append(value)

    return (
        metric_values,
        metric_values_by_fold,
        used_metric_files,
        total_case_rows,
        found_required_subject_ids,
        evaluated_subject_rows,
    )


def export_aggregate_niftis(
    evaluated_subject_rows: Sequence[Tuple[str, Path]],
    gt_dir: Path,
    use_test_set: bool,
    out_dir: Path,
) -> Dict[str, str]:
    if not evaluated_subject_rows:
        raise ValueError("No evaluated EEG subjects available for aggregate NIfTI export.")

    split_dir_token = "test" if use_test_set else "val"
    gt_paths = index_gt_files(gt_dir)
    prior_paths_by_fold: Dict[Path, Dict[str, Path]] = {}

    gt_counts: Optional[np.ndarray] = None
    pred_sum: Optional[np.ndarray] = None
    ref_shape: Optional[Tuple[int, ...]] = None
    ref_affine: Optional[np.ndarray] = None

    for subject_id, fold_dir in evaluated_subject_rows:
        if fold_dir not in prior_paths_by_fold:
            prior_paths_by_fold[fold_dir] = index_prior_files(fold_dir, split_dir_token)

        pred_path = prior_paths_by_fold[fold_dir].get(subject_id)
        if pred_path is None:
            raise ValueError(
                "Aggregate EEG NIfTI export requires per-subject prior maps under "
                f"{fold_dir / 'prior_niftis' / split_dir_token}. Missing map for {subject_id}."
            )

        gt_path = gt_paths.get(subject_id)
        if gt_path is None:
            raise ValueError(
                f"Aggregate EEG NIfTI export requires a GT mask for {subject_id} under {gt_dir}."
            )

        pred_vol, pred_affine = load_nifti_volume(pred_path)
        gt_vol, gt_affine = load_nifti_volume(gt_path)

        ref_shape, ref_affine = _validate_shared_grid(
            ref_shape,
            ref_affine,
            tuple(int(v) for v in pred_vol.shape),
            pred_affine,
            pred_path,
        )
        ref_shape, ref_affine = _validate_shared_grid(
            ref_shape,
            ref_affine,
            tuple(int(v) for v in gt_vol.shape),
            gt_affine,
            gt_path,
        )

        if not np.isfinite(pred_vol).all():
            pred_vol = np.where(np.isfinite(pred_vol), pred_vol, 0.0)

        if gt_counts is None or pred_sum is None:
            gt_counts = np.zeros(ref_shape, dtype=np.int32)
            pred_sum = np.zeros(ref_shape, dtype=np.float32)

        gt_counts += (gt_vol > 0).astype(np.int32)
        pred_sum += pred_vol.astype(np.float32)

    if gt_counts is None or pred_sum is None or ref_affine is None:
        raise ValueError("Failed to accumulate EEG aggregate NIfTI maps.")

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "aggregate_gt_counts.nii.gz"
    pred_path = out_dir / "aggregate_prediction_sum.nii.gz"
    nib.save(nib.Nifti1Image(gt_counts.astype(np.int32), ref_affine), str(gt_path))
    nib.save(nib.Nifti1Image(pred_sum.astype(np.float32), ref_affine), str(pred_path))

    print(f"Aggregate GT count map saved to: {gt_path}")
    print(f"Aggregate prediction sum map saved to: {pred_path}")
    return {
        "gt_counts": str(gt_path),
        "prediction_sum": str(pred_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all numeric per-subject metrics from fold validation.json files, or from predictions.json['test'] when requested, and print mean + 95% CI."
    )
    parser.add_argument(
        "runs_root",
        type=Path,
        default=PRED_DIR_DEFAULT,
        nargs="?",
        help="Directory containing fold run folders (e.g. *_fold0_*).",
    )
    parser.add_argument(
        "--test_set",
        action="store_true",
        help="If set, aggregate metrics from predictions.json['test'] instead of validation.json.",
    )
    parser.add_argument(
        "--subjects_list",
        type=Path,
        default=None,
        help=(
            "Optional TXT file with one subject ID per line. When provided, only these "
            "subjects are aggregated and the script fails if any listed subject is missing."
        ),
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=GT_DIR_DEFAULT,
        help="Directory containing GT mask NIfTIs for aggregate export.",
    )
    parser.add_argument(
        "--save_aggregate_niftis",
        action="store_true",
        help=(
            "If set, export aggregate EEG NIfTIs: GT voxel counts and the summed "
            "continuous prediction maps over evaluated subjects."
        ),
    )
    parser.add_argument(
        "--aggregate_nifti_dir",
        type=Path,
        default=None,
        help="Output directory for aggregate EEG NIfTI exports.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root
    if not runs_root.exists() or not runs_root.is_dir():
        raise SystemExit(f"Invalid runs_root directory: {runs_root}")
    if args.save_aggregate_niftis and args.aggregate_nifti_dir is None:
        raise SystemExit(
            "ERROR: --aggregate_nifti_dir must be provided when --save_aggregate_niftis is set."
        )

    required_subject_ids: Optional[List[str]] = None
    if args.subjects_list is not None:
        try:
            required_subject_ids = load_subjects_list(args.subjects_list)
        except Exception as exc:
            raise SystemExit(f"ERROR: {exc}")

        print(
            f"Loaded {len(required_subject_ids)} required subject IDs from {args.subjects_list}."
        )

    (
        metric_values,
        metric_values_by_fold,
        used_validation_files,
        total_case_rows,
        found_required_subject_ids,
        evaluated_subject_rows,
    ) = aggregate_metrics(
        runs_root,
        use_test_set=args.test_set,
        required_subject_ids=required_subject_ids,
    )

    if required_subject_ids is not None:
        missing = [sid for sid in required_subject_ids if sid not in found_required_subject_ids]
        if missing:
            joined = "\n  - ".join(missing)
            raise SystemExit(
                "ERROR: Required subjects from --subjects_list are missing and evaluation "
                f"must fail:\n  - {joined}"
            )

    if used_validation_files == 0:
        if args.test_set:
            print("No fold predictions.json files with a test block found.")
        else:
            print("No fold validation.json files found.")
        return

    if total_case_rows == 0:
        if args.test_set:
            print("No case-level metrics found in predictions.json['test'] blocks.")
        else:
            print("No case-level metrics found in validation.json files.")
        return

    print(f"Runs root: {runs_root}")
    print(f"Validation files used: {used_validation_files}")
    print(f"Total subject rows: {total_case_rows}")
    print(f"Metrics: {list(metric_values.keys())}")

    if args.save_aggregate_niftis:
        export_aggregate_niftis(
            evaluated_subject_rows=evaluated_subject_rows,
            gt_dir=args.gt_dir,
            use_test_set=args.test_set,
            out_dir=args.aggregate_nifti_dir,
        )

    for metric_name in sorted(metric_values.keys()):
        values = metric_values[metric_name]
        mean, ci_lo, ci_hi = fold_mean_ci95(metric_values_by_fold[metric_name])
        print(f"{metric_name}:")
        print(f"  n = {len(values)}")
        print(f"  folds = {len(metric_values_by_fold[metric_name])}")
        print(f"  mean = {mean:.6f}")
        print(f"  95% CI (fold means) = [{ci_lo:.6f}, {ci_hi:.6f}]")
        print()


if __name__ == "__main__":
    main()
