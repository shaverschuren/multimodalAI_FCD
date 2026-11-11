"""Reconstruct oracle MNI probability maps from channel-by-depth GT targets.

Targets use raw sensitivity overlap; reconstruction uses spatially normalized
channel-depth bases, so this is not an exact inverse of target generation.
The mask-distance bins measure distance to the binary brain-mask boundary,
including sulcal and ventricular boundaries, not specifically pial depth.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from datasets.eeg import (
    CHANNEL_LABEL_TO_INT,
    DEPTH_BIN_NAMES,
    assert_same_grid,
    compute_depth_map_and_bins,
    load_brain_template,
    print_depth_bin_qc,
    resolve_sensitivity_maps,
    resample_mask_to_reference,
    save_depth_qc,
)

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
DATA_DIR = PROJECT_DIR.parent / "data"


def load_targets(path, bin_names=DEPTH_BIN_NAMES):
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    targets = {}
    channel_names = {name.casefold(): name for name in CHANNELS}
    depth_names = {name.casefold(): name for name in bin_names}
    for patient, patient_data in raw.items():
        if not patient_data:
            continue
        target = np.zeros((len(CHANNELS), len(bin_names)), dtype=np.float32)
        for raw_channel, depth_weights in patient_data.items():
            channel = channel_names.get(raw_channel.casefold())
            if channel is None:
                raise ValueError(f"Unknown channel {raw_channel!r} for {patient}.")
            if not isinstance(depth_weights, dict):
                raise ValueError(f"Expected depth-bin object for {patient}/{raw_channel}.")
            for raw_depth, value in depth_weights.items():
                depth = depth_names.get(raw_depth.casefold())
                if depth is None:
                    raise ValueError(f"Unknown depth bin {raw_depth!r} for {patient}.")
                target[CHANNEL_LABEL_TO_INT[channel], bin_names.index(depth)] += float(value)
        if not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError(f"Invalid joint target for {patient}.")
        if not np.isclose(target.sum(), 1.0, atol=1e-5):
            raise ValueError(f"Joint target for {patient} sums to {target.sum():.6g}, not one.")
        targets[patient] = target
    if not targets:
        raise ValueError(f"No non-empty channel-depth targets in {path}.")
    return targets


def load_channel_targets(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    targets = {}
    names = {name.casefold(): name for name in CHANNELS}
    for patient, entry in raw.items():
        vector = np.zeros(len(CHANNELS), dtype=np.float32)
        for raw_channel, value in entry.items():
            channel = names.get(raw_channel.casefold())
            if channel is not None:
                vector[CHANNEL_LABEL_TO_INT[channel]] = float(value)
        if vector.sum() > 0:
            vector /= vector.sum()
            targets[patient] = vector
    return targets


def bins_for_channel(bins, channel):
    return bins(channel) if callable(bins) else bins


def component_normalizers(map_paths, reference, bins):
    normalizers = None
    for channel_index, channel in enumerate(CHANNELS):
        channel_bins = bins_for_channel(bins, channel)
        if normalizers is None:
            normalizers = np.zeros(
                (len(CHANNELS), channel_bins.shape[0]), dtype=np.float64
            )
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        if not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError(f"Invalid sensitivity values for {channel}.")
        for bin_index in range(channel_bins.shape[0]):
            normalizers[channel_index, bin_index] = np.sum(
                sensitivity * channel_bins[bin_index], dtype=np.float64
            )
    return normalizers


def reconstruct_oracle(target, map_paths, reference, bins, normalizers,
                       brain_mask):
    volume = np.zeros(reference.shape[:3], dtype=np.float32)
    for channel_index, channel in enumerate(CHANNELS):
        if not target[channel_index].any():
            continue
        channel_bins = bins_for_channel(bins, channel)
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        for depth_index, weight in enumerate(target[channel_index]):
            if weight <= 0:
                continue
            normalizer = normalizers[channel_index, depth_index]
            if normalizer <= 0:
                raise ValueError(f"Nonzero target has empty basis: {channel}/bin-{depth_index}.")
            volume += (weight / normalizer) * sensitivity * channel_bins[depth_index]
    volume[~brain_mask] = 0.0
    if not np.isfinite(volume).all() or volume.sum() <= 0:
        raise ValueError("Oracle reconstruction is empty or non-finite.")
    return volume / volume.sum(dtype=np.float64)


def reconstruct_channel_only(target, map_paths, reference, brain_mask):
    volume = np.zeros(reference.shape[:3], dtype=np.float32)
    for channel_index, channel in enumerate(CHANNELS):
        if target[channel_index] <= 0:
            continue
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        total = sensitivity.sum(dtype=np.float64)
        if total <= 0:
            raise ValueError(f"Sensitivity map for {channel} is empty.")
        volume += target[channel_index] * sensitivity / total
    volume[~brain_mask] = 0.0
    total = volume.sum(dtype=np.float64)
    return volume / total if total > 0 else volume


def depth_distribution(volume, depth_bins):
    distribution = np.sum(volume[None] * depth_bins, axis=(1, 2, 3), dtype=np.float64)
    total = distribution.sum()
    return distribution / total if total > 0 else distribution


def equal_volume_dice(volume, mask, brain_mask):
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    brain_values = volume[brain_mask]
    selected = np.argpartition(brain_values, -count)[-count:]
    predicted_mask = np.zeros(brain_mask.sum(), dtype=bool)
    predicted_mask[selected] = True
    intersection = np.count_nonzero(predicted_mask & mask[brain_mask])
    return float(2 * intersection / (count + count))


def save_patient_qc(patient, mask, original, oracle, reference, output_dir):
    center = np.argwhere(mask).mean(axis=0)
    z_index = int(np.clip(round(center[2]), 0, mask.shape[2] - 1))
    mask_slice = mask[:, :, z_index].T
    import matplotlib.pyplot as plt

    figure, figure_axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, data, title in zip(
        figure_axes,
        (mask.astype(np.float32), original, oracle),
        ("Resection mask", "Channel-only", "Depth-aware oracle"),
    ):
        image = axis.imshow(data[:, :, z_index].T, origin="lower", cmap="magma")
        axis.contour(mask_slice.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.78)
    world_z = reference.affine[2, 2] * z_index + reference.affine[2, 3]
    figure.suptitle(f"{patient} | axial z={world_z:.1f} mm | cyan: GT boundary")
    path = Path(output_dir) / f"{patient}_depth_oracle_qc.png"
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def reconstruct(targets_path, sensitivity_dir, output_dir):
    targets_path = Path(targets_path)
    output_dir = Path(output_dir)
    targets = load_targets(targets_path)
    channel_targets = load_channel_targets(targets_path.with_name("gt_channels.json"))
    map_paths = resolve_sensitivity_maps(sensitivity_dir, CHANNELS)
    reference = nib.load(str(map_paths[CHANNELS[0]]))
    template_path = PROJECT_DIR / "preprocessing" / "eeg" / "MNI152_T1_1mm_brain.nii.gz"
    template, brain_mask = load_brain_template(template_path)
    assert_same_grid(reference, template, "Sensitivity-map reference")
    depth_mm, depth_bins = compute_depth_map_and_bins(brain_mask, template.affine)
    print_depth_bin_qc(depth_mm, depth_bins, brain_mask)
    save_depth_qc(depth_mm, depth_bins, brain_mask, template.affine,
                  output_dir / "depth_map_qc.png")
    normalizers = component_normalizers(map_paths, reference, depth_bins)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = DATA_DIR / "preprocessing" / "mri"
    mean_mask_depth = {}
    for patient in targets:
        mask_path = mask_dir / patient / f"{patient}_gt_norm.nii.gz"
        if mask_path.exists():
            mask = (resample_mask_to_reference(mask_path, template) > 0.5) & brain_mask
            if mask.any():
                mean_mask_depth[patient] = float(depth_mm[mask].mean())
                if patient not in channel_targets:
                    mean_mask_depth.pop(patient)

    qc_path = output_dir / "depth_oracle_qc.csv"
    fields = ["patient_id", "target_superficial", "target_intermediate", "target_deep",
              "mask_superficial", "mask_intermediate", "mask_deep",
              "channel_superficial", "channel_intermediate", "channel_deep",
              "oracle_superficial", "oracle_intermediate", "oracle_deep",
              "channel_mass_in_mask", "oracle_mass_in_mask",
              "channel_equal_volume_dice", "oracle_equal_volume_dice"]
    with qc_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for patient, target in targets.items():
            oracle = reconstruct_oracle(target, map_paths, reference, depth_bins,
                                        normalizers, brain_mask)
            nii = nib.Nifti1Image(oracle.astype(np.float32), reference.affine,
                                  template.header.copy())
            nii.set_data_dtype(np.float32)
            nib.save(nii, output_dir / f"{patient}_eeg_v2_depth_oracle.nii.gz")

            row = {"patient_id": patient}
            for name, value in zip(DEPTH_BIN_NAMES, target.sum(axis=0)):
                row[f"target_{name}"] = float(value)
            mask_path = mask_dir / patient / f"{patient}_gt_norm.nii.gz"
            mask = None
            if mask_path.exists():
                mask = (resample_mask_to_reference(mask_path, template) > 0.5) & brain_mask
            channel = channel_targets.get(patient)
            channel_volume = None
            if channel is not None:
                channel_volume = reconstruct_channel_only(channel, map_paths,
                                                          reference, brain_mask)
                for name, value in zip(DEPTH_BIN_NAMES, depth_distribution(channel_volume, depth_bins)):
                    row[f"channel_{name}"] = float(value)
            if mask is not None and mask.any():
                mask_distribution = np.mean(depth_bins[:, mask], axis=1)
                oracle_distribution = depth_distribution(oracle, depth_bins)
                for name, value in zip(DEPTH_BIN_NAMES, mask_distribution):
                    row[f"mask_{name}"] = float(value)
                for name, value in zip(DEPTH_BIN_NAMES, oracle_distribution):
                    row[f"oracle_{name}"] = float(value)
                row["oracle_mass_in_mask"] = float(oracle[mask].sum())
                row["oracle_equal_volume_dice"] = equal_volume_dice(oracle, mask, brain_mask)
                if channel_volume is not None:
                    row["channel_mass_in_mask"] = float(channel_volume[mask].sum())
                    row["channel_equal_volume_dice"] = equal_volume_dice(channel_volume, mask, brain_mask)
                if channel_volume is not None:
                    save_patient_qc(patient, mask, channel_volume, oracle, template, output_dir)
            writer.writerow(row)
            print(f"{patient} target depth: " + ", ".join(
                f"{name}={value:.3f}" for name, value in zip(DEPTH_BIN_NAMES, target.sum(axis=0))
            ))
    print(f"Saved oracle volumes and QC report to {output_dir}")
    print(f"QC table: {qc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets_json", type=Path, help="gt_channel_depth.json")
    parser.add_argument("sensitivity_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    reconstruct(args.targets_json, args.sensitivity_dir, args.output_dir)