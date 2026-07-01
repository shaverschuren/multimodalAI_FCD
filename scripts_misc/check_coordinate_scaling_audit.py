"""
Audit coordinate scaling consistency across EEG GT targets, MRI affines, and validation metrics.

This script checks:
1) GT JSON round-trip consistency between normalized and mm coordinates.
2) Whether mm coordinates map consistently through each preprocessed MRI affine.
3) (Optional) Whether validation.json reported mm distances match normalized distances
   scaled by the configured MNI extent.

Usage examples:

  python check_coordinate_scaling_audit.py \
      --gt_json L:/.../gt_coords.json \
      --mri_npy_dir L:/.../preprocessing/mri

  python check_coordinate_scaling_audit.py \
      --gt_json L:/.../gt_coords.json \
      --mri_npy_dir L:/.../preprocessing/mri \
      --validation_json L:/.../runs/.../validation.json \
      --output_json L:/.../scaling_audit_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_MNI_EXTENT_MM = np.array([90.0, 126.0, 72.0], dtype=np.float64)


def _as_vec3(d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])], dtype=np.float64)


def load_gt_json(path: str) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    with open(path, "r") as f:
        raw = json.load(f)

    out: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for pid, entry in raw.items():
        mu_mm = _as_vec3(entry["mni_mu_mm"]) if "mni_mu_mm" in entry else None
        mu_norm = _as_vec3(entry["normalized_mu"]) if "normalized_mu" in entry else None
        sigma_mm = _as_vec3(entry["mni_sigma_mm"]) if "mni_sigma_mm" in entry else None
        sigma_norm = _as_vec3(entry["normalized_sigma"]) if "normalized_sigma" in entry else None
        out[pid] = {
            "mu_mm": mu_mm,
            "mu_norm": mu_norm,
            "sigma_mm": sigma_mm,
            "sigma_norm": sigma_norm,
        }
    return out


def audit_gt_roundtrip(
    gt_data: Dict[str, Dict[str, Optional[np.ndarray]]], extent_mm: np.ndarray
) -> Dict[str, Dict[str, float]]:
    per_case: Dict[str, Dict[str, float]] = {}

    for pid, entry in gt_data.items():
        mu_mm = entry["mu_mm"]
        mu_norm = entry["mu_norm"]
        sigma_mm = entry["sigma_mm"]
        sigma_norm = entry["sigma_norm"]

        metrics: Dict[str, float] = {}

        if mu_mm is not None and mu_norm is not None:
            mm_from_norm = mu_norm * extent_mm
            norm_from_mm = mu_mm / extent_mm
            metrics["mu_mm_from_norm_max_abs_err"] = float(np.max(np.abs(mm_from_norm - mu_mm)))
            metrics["mu_norm_from_mm_max_abs_err"] = float(np.max(np.abs(norm_from_mm - mu_norm)))

        if sigma_mm is not None and sigma_norm is not None:
            mm_from_norm = sigma_norm * extent_mm
            norm_from_mm = sigma_mm / extent_mm
            metrics["sigma_mm_from_norm_max_abs_err"] = float(np.max(np.abs(mm_from_norm - sigma_mm)))
            metrics["sigma_norm_from_mm_max_abs_err"] = float(np.max(np.abs(norm_from_mm - sigma_norm)))

        if metrics:
            per_case[pid] = metrics

    return per_case


def _load_npz_affine(npz_path: str) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    npz = np.load(npz_path, allow_pickle=True)
    affine = np.asarray(npz["affine"], dtype=np.float64)
    shape = None
    if "image" in npz:
        img = npz["image"]
        if img.ndim == 4:
            shape = (int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
    npz.close()
    return affine, shape


def _mm_to_vox(mm_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv_aff = np.linalg.inv(affine)
    mm_h = np.array([mm_xyz[0], mm_xyz[1], mm_xyz[2], 1.0], dtype=np.float64)
    vox_h = inv_aff @ mm_h
    return vox_h[:3]


def _vox_to_mm(vox_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    vox_h = np.array([vox_xyz[0], vox_xyz[1], vox_xyz[2], 1.0], dtype=np.float64)
    mm_h = affine @ vox_h
    return mm_h[:3]


def audit_affine_mapping(
    gt_data: Dict[str, Dict[str, Optional[np.ndarray]]],
    mri_npy_dir: str,
) -> Dict[str, Dict[str, float]]:
    per_case: Dict[str, Dict[str, float]] = {}

    for pid, entry in gt_data.items():
        mu_mm = entry["mu_mm"]
        if mu_mm is None:
            continue

        npz_path = os.path.join(mri_npy_dir, f"{pid}_preproc.npz")
        if not os.path.exists(npz_path):
            continue

        try:
            affine, shape = _load_npz_affine(npz_path)
        except Exception:
            continue

        vox = _mm_to_vox(mu_mm, affine)
        mm_back = _vox_to_mm(vox, affine)
        roundtrip_err = float(np.max(np.abs(mm_back - mu_mm)))

        R = affine[:3, :3]
        voxel_sizes = np.sqrt((R ** 2).sum(axis=0))
        det_R = float(np.linalg.det(R))

        metrics: Dict[str, float] = {
            "affine_mm_roundtrip_max_abs_err": roundtrip_err,
            "vox_size_x_mm": float(voxel_sizes[0]),
            "vox_size_y_mm": float(voxel_sizes[1]),
            "vox_size_z_mm": float(voxel_sizes[2]),
            "det_affine_3x3": det_R,
        }

        if shape is not None:
            in_bounds = (
                (0.0 <= vox[0] < shape[0])
                and (0.0 <= vox[1] < shape[1])
                and (0.0 <= vox[2] < shape[2])
            )
            metrics["mu_vox_in_bounds"] = 1.0 if in_bounds else 0.0

        per_case[pid] = metrics

    return per_case


def load_validation_cases(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r") as f:
        data = json.load(f)

    cases = data.get("cases", data)
    if not isinstance(cases, dict):
        raise ValueError("validation.json has unsupported format.")
    return cases


def audit_validation_scaling(
    validation_cases: Dict[str, Dict[str, object]], extent_mm: np.ndarray
) -> Dict[str, Dict[str, float]]:
    per_case: Dict[str, Dict[str, float]] = {}

    for pid, entry in validation_cases.items():
        pred_mu = entry.get("pred_mu")
        gt_mu = entry.get("gt_mu")
        if pred_mu is None or gt_mu is None:
            continue

        pred = np.asarray(pred_mu, dtype=np.float64)
        gt = np.asarray(gt_mu, dtype=np.float64)
        diff_norm = pred - gt

        calc_norm = float(np.linalg.norm(diff_norm))
        calc_mm = float(np.linalg.norm(diff_norm * extent_mm))

        stored_norm = entry.get("coord_euclidean_norm", entry.get("euclidean_norm"))
        stored_mm = entry.get("coord_euclidean_mm", entry.get("euclidean_mm"))

        metrics: Dict[str, float] = {
            "calc_euclidean_norm": calc_norm,
            "calc_euclidean_mm": calc_mm,
        }

        if stored_norm is not None:
            metrics["stored_euclidean_norm"] = float(stored_norm)
            metrics["norm_abs_err"] = abs(calc_norm - float(stored_norm))

        if stored_mm is not None:
            metrics["stored_euclidean_mm"] = float(stored_mm)
            metrics["mm_abs_err"] = abs(calc_mm - float(stored_mm))

        per_case[pid] = metrics

    return per_case


def summarize_max_errors(report: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    maxima: Dict[str, float] = {}

    for section in report.values():
        for metrics in section.values():
            for k, v in metrics.items():
                if not ("err" in k):
                    continue
                maxima[k] = max(maxima.get(k, 0.0), float(v))

    return maxima


def select_patients(all_ids: List[str], keep_ids: Optional[List[str]]) -> List[str]:
    if not keep_ids:
        return all_ids
    keep = set(keep_ids)
    return [pid for pid in all_ids if pid in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mm/normalized coordinate scaling consistency.")
    parser.add_argument("--gt_json", required=True, help="Path to gt_coords-like JSON.")
    parser.add_argument(
        "--mri_npy_dir",
        required=True,
        help="Directory containing {patient_id}_preproc.npz files.",
    )
    parser.add_argument(
        "--validation_json",
        default=None,
        help="Optional validation.json (new or legacy format) for metric consistency checks.",
    )
    parser.add_argument(
        "--patient_ids",
        nargs="*",
        default=None,
        help="Optional list of patient IDs to restrict the audit.",
    )
    parser.add_argument(
        "--extent_mm",
        nargs=3,
        type=float,
        default=list(DEFAULT_MNI_EXTENT_MM),
        metavar=("EX", "EY", "EZ"),
        help="Normalization extent in mm (default: 90 126 72).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output path for full audit report JSON.",
    )
    args = parser.parse_args()

    extent_mm = np.asarray(args.extent_mm, dtype=np.float64)
    gt_all = load_gt_json(args.gt_json)

    selected_ids = select_patients(sorted(gt_all.keys()), args.patient_ids)
    gt_data = {pid: gt_all[pid] for pid in selected_ids}

    report: Dict[str, Dict[str, Dict[str, float]]] = {
        "gt_roundtrip": audit_gt_roundtrip(gt_data, extent_mm),
        "affine_mapping": audit_affine_mapping(gt_data, args.mri_npy_dir),
    }

    if args.validation_json is not None:
        val_cases = load_validation_cases(args.validation_json)
        if args.patient_ids:
            val_cases = {k: v for k, v in val_cases.items() if k in set(args.patient_ids)}
        report["validation_scaling"] = audit_validation_scaling(val_cases, extent_mm)

    maxima = summarize_max_errors(report)

    print("=" * 80)
    print("Coordinate Scaling Audit")
    print("=" * 80)
    print(f"Patients requested: {len(selected_ids)}")
    print(f"GT roundtrip cases: {len(report['gt_roundtrip'])}")
    print(f"Affine mapping cases: {len(report['affine_mapping'])}")
    if "validation_scaling" in report:
        print(f"Validation scaling cases: {len(report['validation_scaling'])}")

    if maxima:
        print("\nMax absolute errors:")
        for k in sorted(maxima.keys()):
            print(f"  {k}: {maxima[k]:.6e}")
    else:
        print("\nNo error metrics available to summarize.")

    if args.output_json is not None:
        serializable = {
            "extent_mm": extent_mm.tolist(),
            "max_errors": maxima,
            "report": report,
        }
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved full report to: {args.output_json}")


if __name__ == "__main__":
    main()
