"""
inference/eeg_spike_mil_mh_inference.py

Standalone inference script for SpikeMILModel (multi-head, deconv-head variant).

Runs inference on an explicit test patient set without re-training. Mirrors the
deconv inference pipeline used in training/eeg_spike_mil_mh_training.py.

Usage:
    python -m inference.eeg_spike_mil_mh_inference \\
        --checkpoint <path> \\
        --data_dir <path> \\
        --patient_ids RESP0001 RESP0002 ...
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from datasets.eeg import (
    HEMI_LABEL_TO_INT,
    LOBE_CLASSES,
    MultiHeadTargetDataset,
    PatientMILSpikeDataset,
    load_multitask_targets,
    mil_multitask_collate,
)
from models.eeg import SpikeMILModel
from training.eeg_spike_mil_mh_training import infer_predictions, generate_deconv_niftis


def _load_patient_ids(patient_ids: Optional[List[str]], patient_ids_path: Optional[str]) -> List[str]:
    """Load test patient IDs from CLI list and/or a file, preserving order."""
    out: List[str] = []

    if patient_ids:
        for raw in patient_ids:
            for token in str(raw).split(","):
                pid = token.strip()
                if pid:
                    out.append(pid)

    if patient_ids_path is not None:
        if not os.path.exists(patient_ids_path):
            raise FileNotFoundError(f"patient_ids_path not found: {patient_ids_path}")

        suffix = os.path.splitext(patient_ids_path)[1].lower()
        if suffix == ".json":
            with open(patient_ids_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                out.extend(str(x).strip() for x in data if str(x).strip())
            elif isinstance(data, dict):
                for key in ("test_ids", "patient_ids", "ids"):
                    if key in data and isinstance(data[key], list):
                        out.extend(str(x).strip() for x in data[key] if str(x).strip())
                        break
                else:
                    raise ValueError(
                        "JSON patient-id file must be a list or contain one of keys "
                        "['test_ids', 'patient_ids', 'ids']."
                    )
            else:
                raise ValueError("Unsupported JSON patient-id format.")
        else:
            with open(patient_ids_path, "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        out.append(pid)

    # Deduplicate while preserving first occurrence.
    dedup: List[str] = []
    seen = set()
    for pid in out:
        if pid not in seen:
            seen.add(pid)
            dedup.append(pid)

    if not dedup:
        raise ValueError("No patient IDs provided. Use --patient_ids and/or --patient_ids_path.")

    return dedup


def _find_patient_files_only(
    data_dir: str,
    patient_ids: List[str],
    test_mode: bool = False,
    file_suffix: str = "_spikes_1-70Hz.npy",
):
    """Resolve patient spike files without requiring label/target JSON."""
    valid_ids: List[str] = []
    files: List[str] = []
    skipped: List[str] = []

    for pid in patient_ids:
        path = os.path.join(data_dir, f"{pid}{file_suffix}")
        if os.path.exists(path):
            valid_ids.append(pid)
            files.append(path)
        else:
            skipped.append(pid)

    if skipped:
        print(
            f"Warning: Skipped {len(skipped)} patients (missing file): {sorted(skipped)}"
        )

    if test_mode:
        valid_ids = valid_ids[:4]
        files = files[:4]
        print("Warning: test_mode enabled - limited to first 4 valid patients.")

    print(f"Found {len(valid_ids)} valid patients out of {len(patient_ids)} requested.")
    return valid_ids, files


def _dummy_target_entry() -> Dict[str, torch.Tensor]:
    """Build a target entry compatible with MultiHeadTargetDataset without labels."""
    return {
        "mu": torch.zeros(3, dtype=torch.float32),
        "hemi_target": torch.tensor(0, dtype=torch.long),
        "lobe_target": torch.tensor(0, dtype=torch.long),
        "hemi_mask": torch.tensor(0.0, dtype=torch.float32),
        "lobe_mask": torch.tensor(0.0, dtype=torch.float32),
    }


def _build_test_summary(test_preds: Dict[str, Dict]) -> Dict:
    """Aggregate test-set deconv metrics across predictions."""
    metric_keys = [
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

    values: Dict[str, List[float]] = {k: [] for k in metric_keys}
    num_with_stats = 0
    num_with_gt_metrics = 0

    for pred in test_preds.values():
        stats = pred.get("deconv_spatial_stats")
        if not stats:
            continue
        num_with_stats += 1
        if stats.get("dice") is not None:
            num_with_gt_metrics += 1

        for key in metric_keys:
            val = stats.get(key)
            if val is not None:
                values[key].append(float(val))

    means = {
        key: (sum(vals) / len(vals) if vals else None)
        for key, vals in values.items()
    }

    return {
        "num_cases": len(test_preds),
        "num_cases_with_deconv_stats": num_with_stats,
        "num_cases_with_gt_deconv_metrics": num_with_gt_metrics,
        "means": means,
    }


def run_test_inference(
    checkpoint_path: str,
    data_dir: str,
    targets_json: Optional[str],
    patient_ids: List[str],
    output_dir: Optional[str] = None,
    lobe_json: Optional[str] = None,
    hemi_json: Optional[str] = None,
    mri_npy_dir: Optional[str] = None,
    in_channels: int = 21,
    max_spikes: int = 128,
    min_spikes_per_patient: int = 64,
    num_workers: int = 0,
    test_mode: bool = False,
    generate_niftis: bool = False,
    deconv_save_raw_prior_nifti: bool = False,
):
    """Load one trained fold checkpoint and run inference on explicit test IDs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args: Dict = checkpoint.get("args", {})

    use_coord_head = bool(saved_args.get("use_coord_head", True))
    use_hemi_head = bool(saved_args.get("use_hemi_head", True))
    use_lobe_head = bool(saved_args.get("use_lobe_head", True))

    model_kwargs = {
        "emb_dim": saved_args.get("emb_dim"),
        "hidden": saved_args.get("hidden"),
        "dropout": saved_args.get("dropout"),
        "encoder_type": saved_args.get("encoder_type"),
        "pooling": saved_args.get("pooling"),
        "spatial_head": saved_args.get("spatial_head", "none"),
        "num_gaussians": saved_args.get("num_gaussians", 3),
        "gaussian_coord_dim": saved_args.get("gaussian_coord_dim", 3),
        "gaussian_sigma_min": saved_args.get("gaussian_sigma_min"),
        "gaussian_sigma_max": saved_args.get("gaussian_sigma_max"),
        "gaussian_isotropic": saved_args.get("gaussian_isotropic", True),
        "gaussian_output_space": saved_args.get("gaussian_output_space", "normalized"),
        "gaussian_make_heatmap": saved_args.get("gaussian_make_heatmap", False),
        "gaussian_heatmap_shape": saved_args.get("gaussian_heatmap_shape"),
        "deconv_output_shape": saved_args.get("deconv_output_shape"),
        "deconv_latent_shape": saved_args.get("deconv_latent_shape"),
        "deconv_base_channels": saved_args.get("deconv_base_channels"),
        "deconv_dropout": saved_args.get("deconv_dropout"),
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    resolved_spatial_head = model_kwargs.get("spatial_head", "none")
    if resolved_spatial_head != "deconv":
        raise ValueError(
            f"Checkpoint spatial_head is {resolved_spatial_head!r}. "
            "This script is intended for deconv-head inference."
        )

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

    found_ids, patient_files = _find_patient_files_only(
        data_dir,
        patient_ids,
        test_mode=test_mode,
    )
    if not found_ids:
        raise ValueError("No valid test patients found after file/target checks.")

    if targets_json is not None:
        target_dict = load_multitask_targets(
            targets_json,
            lobe_json_path=lobe_json,
            hemi_json_path=hemi_json,
        )
    else:
        target_dict = {}

    # Keep only IDs present in both requested files and target map when targets are provided.
    if targets_json is not None:
        missing_targets = [pid for pid in found_ids if pid not in target_dict]
        if missing_targets:
            print(
                "Warning: Skipping patients missing targets in targets_json: "
                f"{sorted(missing_targets)}"
            )
            keep = [pid for pid in found_ids if pid in target_dict]
            keep_set = set(keep)
            patient_files = [f for pid, f in zip(found_ids, patient_files) if pid in keep_set]
            found_ids = keep

    if not found_ids:
        raise ValueError("No valid test patients left after target filtering.")

    if targets_json is None:
        for pid in found_ids:
            target_dict[pid] = _dummy_target_entry()

    patient_targets = [target_dict[pid] for pid in found_ids]

    test_base = PatientMILSpikeDataset(
        found_ids,
        patient_files,
        patient_targets,
        max_spikes_per_bag=max_spikes,
        training=False,
        min_spikes_per_patient=min_spikes_per_patient,
    )

    test_dataset = MultiHeadTargetDataset(
        test_base,
        target_by_pid=target_dict,
        deconv_enabled=bool(targets_json is not None or mri_npy_dir is not None),
        deconv_output_shape=model_kwargs.get("deconv_output_shape"),
        deconv_target_blur_sigma=float(saved_args.get("deconv_target_blur_sigma", 0.0)),
        deconv_require_mni_alignment=bool(targets_json is not None or mri_npy_dir is not None),
        deconv_mask_npz_dir=(mri_npy_dir if mri_npy_dir is not None else None),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_multitask_collate,
        pin_memory=True,
    )

    print("Running inference on test set...")
    test_preds = infer_predictions(model, test_loader, test_dataset.patient_ids, device)
    print(f"Got predictions for {len(test_preds)} test cases")

    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "test_predictions.json")
    with open(output_path, "w") as f:
        json.dump({"test": test_preds}, f, indent=2)
    print(f"Saved test predictions to: {output_path}")

    deconv_metrics_csv_path = os.path.join(output_dir, "test_deconv_inference_metrics.csv")
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
        for case_id, pred in test_preds.items():
            stats = pred.get("deconv_spatial_stats") or {}
            row = {"split": "test", "case_id": case_id}
            for key in csv_fields[2:]:
                row[key] = stats.get(key)
            writer_csv.writerow(row)
    print(f"Saved test deconv metrics CSV to: {deconv_metrics_csv_path}")

    summary = _build_test_summary(test_preds)
    summary_path = os.path.join(output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved test summary to: {summary_path}")

    inference_settings = {
        "checkpoint_path": checkpoint_path,
        "data_dir": data_dir,
        "targets_json": targets_json,
        "lobe_json": lobe_json,
        "hemi_json": hemi_json,
        "patient_ids": patient_ids,
        "resolved_patient_ids": test_dataset.patient_ids,
        "in_channels": in_channels,
        "max_spikes": max_spikes,
        "min_spikes_per_patient": min_spikes_per_patient,
        "num_workers": num_workers,
        "test_mode": test_mode,
        "mri_npy_dir": mri_npy_dir,
        "generate_niftis": generate_niftis,
        "deconv_save_raw_prior_nifti": deconv_save_raw_prior_nifti,
        "use_coord_head": use_coord_head,
        "use_hemi_head": use_hemi_head,
        "use_lobe_head": use_lobe_head,
        "resolved_model_kwargs": model_kwargs,
        "saved_train_args": saved_args,
    }
    settings_path = os.path.join(output_dir, "test_inference_settings.json")
    with open(settings_path, "w") as f:
        json.dump(inference_settings, f, indent=2)
    print(f"Saved test inference settings to: {settings_path}")

    if generate_niftis:
        if mri_npy_dir is None:
            print(
                "Warning: generate_niftis=True but mri_npy_dir is None. "
                "Skipping NIfTI generation."
            )
        else:
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

    return test_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-fold deconv-focused EEG MIL inference on test patients."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained checkpoint (.pt).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing <pid>_spikes_1-70Hz.npy files.")
    parser.add_argument("--targets_json", type=str, default=None,
                        help="Optional JSON with normalized_mu targets and mask metadata. "
                             "Not required for prediction-only inference.")
    parser.add_argument("--patient_ids", type=str, nargs="*", default=None,
                        help="Test patient IDs (space- or comma-separated).")
    parser.add_argument("--patient_ids_path", type=str, default=None,
                        help="Path to .txt/.json file with test patient IDs.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to checkpoint directory).")
    parser.add_argument("--lobe_json", type=str, default=None,
                        help="Optional lobe label JSON.")
    parser.add_argument("--hemi_json", type=str, default=None,
                        help="Optional hemisphere label JSON.")
    parser.add_argument("--mri_npy_dir", type=str, default=None,
                        help="Directory with <pid>_preproc.npz files for deconv masks/NIfTI export.")
    parser.add_argument("--in_channels", type=int, default=21,
                        help="Number of EEG channels in input.")
    parser.add_argument("--max_spikes", type=int, default=128,
                        help="Max spikes sampled per patient at inference.")
    parser.add_argument("--min_spikes_per_patient", type=int, default=64,
                        help="Minimum spikes required per patient.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker count.")
    parser.add_argument("--test_mode", action="store_true",
                        help="If set, limit to first 4 valid patients.")
    parser.add_argument("--generate_niftis", action="store_true",
                        help="If set, export deconv prior NIfTIs for test patients.")
    parser.add_argument("--deconv_save_raw_prior_nifti", action=argparse.BooleanOptionalAction, default=False,
                        help="When exporting, also save unmasked raw deconv prior volumes.")

    args = parser.parse_args()
    test_ids = _load_patient_ids(args.patient_ids, args.patient_ids_path)

    run_test_inference(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        targets_json=args.targets_json,
        patient_ids=test_ids,
        output_dir=args.output_dir,
        lobe_json=args.lobe_json,
        hemi_json=args.hemi_json,
        mri_npy_dir=args.mri_npy_dir,
        in_channels=args.in_channels,
        max_spikes=args.max_spikes,
        min_spikes_per_patient=args.min_spikes_per_patient,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
        generate_niftis=args.generate_niftis,
        deconv_save_raw_prior_nifti=args.deconv_save_raw_prior_nifti,
    )
