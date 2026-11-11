"""Build and compare channel-only, depth and electrode-distance oracle maps.

This is a geometric oracle, not a biophysical EEG forward model. The targets
use sensitivity-weighted overlaps, while reconstruction uses spatially
normalized basis functions, so the results are not exact inverse solutions.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.eeg import (
    CHANNEL_LABEL_TO_INT,
    DEPTH_BIN_NAMES,
    DISTANCE_BIN_NAMES,
    assert_same_grid,
    compute_depth_map_and_bins,
    compute_electrode_distance_bins,
    load_brain_template,
    resolve_sensitivity_maps,
    resample_mask_to_reference,
    validate_electrode_coordinates,
)
from reconstruct_eeg_v2_depth_oracle_nii import (
    component_normalizers,
    depth_distribution,
    equal_volume_dice,
    load_channel_targets,
    load_targets,
    reconstruct_channel_only,
    reconstruct_oracle,
)

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
DATA_DIR = PROJECT_DIR.parent / "data"
COORDINATE_FILE = PROJECT_DIR / "preprocessing" / "eeg" / "standard_1005_mni_coordinates.json"


def load_scalp_coordinates(template, brain_mask):
    with COORDINATE_FILE.open(encoding="utf-8") as handle:
        document = json.load(handle)
    source = document.get("source", {})
    if source.get("unit") != "mm":
        raise ValueError(f"Expected scalp coordinates in millimetres, got {source.get('unit')!r}.")
    coordinate_system = source.get("coordinate_system", "unspecified")
    if "mni" not in coordinate_system.casefold():
        warnings.warn(
            f"Coordinate metadata declares {coordinate_system!r}, not MNI; "
            "confirm template alignment before interpreting distance targets.",
            UserWarning,
            stacklevel=2,
        )
    coords = {channel: document["data"][channel]["mni"] for channel in CHANNELS}
    validate_electrode_coordinates(coords, brain_mask, template.affine, template.header)
    return coords


def center_world(volume, affine):
    center = center_of_mass(volume)
    return nib.affines.apply_affine(affine, center)


def save_comparison_figure(patient, mask, maps, reference, output_dir):
    import matplotlib.pyplot as plt

    center = np.argwhere(mask).mean(axis=0)
    z_index = int(np.clip(round(center[2]), 0, mask.shape[2] - 1))
    brain_slice = mask[:, :, z_index].T
    views = [volume[:, :, z_index].T for volume in maps]
    positive = [view[view > 0] for view in views if np.any(view > 0)]
    vmax = max((float(np.percentile(values, 99.5)) for values in positive), default=1.0)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    titles = ("Resection mask", "Channel-only", "Anatomical depth", "Electrode distance")
    axes[0].imshow(brain_slice, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(titles[0])
    axes[0].set_axis_off()
    for axis, view, title in zip(axes[1:], views, titles[1:]):
        image = axis.imshow(view, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.contour(brain_slice.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, label="Probability mass per voxel", shrink=0.78)
    world_z = reference.affine[2, 2] * z_index + reference.affine[2, 3]
    figure.suptitle(f"{patient} | shared axial slice z={world_z:.1f} mm | common map scale")
    output_path = Path(output_dir) / f"{patient}_oracle_comparison.png"
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


def reconstruct(distance_targets_path, sensitivity_dir, output_dir):
    distance_targets_path = Path(distance_targets_path)
    output_dir = Path(output_dir)
    channel_path = distance_targets_path.with_name("gt_channels.json")
    depth_path = distance_targets_path.with_name("gt_channel_depth.json")
    distance_targets = load_targets(distance_targets_path, DISTANCE_BIN_NAMES)
    depth_targets = load_targets(depth_path, DEPTH_BIN_NAMES)
    channel_targets = load_channel_targets(channel_path)
    patients = sorted(distance_targets)
    comparable_patients = set(patients) & set(depth_targets) & set(channel_targets)

    map_paths = resolve_sensitivity_maps(sensitivity_dir, CHANNELS)
    reference = nib.load(str(map_paths[CHANNELS[0]]))
    template_path = PROJECT_DIR / "preprocessing" / "eeg" / "MNI152_T1_1mm_brain.nii.gz"
    template, brain_mask = load_brain_template(template_path)
    assert_same_grid(reference, template, "Sensitivity-map reference")
    depth_mm, depth_bins = compute_depth_map_and_bins(brain_mask, template.affine)
    coords = load_scalp_coordinates(template, brain_mask)
    output_dir.mkdir(parents=True, exist_ok=True)

    def distance_bins_for(channel):
        return compute_electrode_distance_bins(
            coords[channel], brain_mask, template.affine
        )[1]

    distance_normalizers = component_normalizers(map_paths, reference, distance_bins_for)
    depth_normalizers = component_normalizers(map_paths, reference, depth_bins)
    attenuation_rows = []
    for channel_index, channel in enumerate(CHANNELS):
        shares = distance_normalizers[channel_index]
        shares = shares / shares.sum()
        attenuation_rows.append([channel, *shares.tolist()])
    total_attenuation = distance_normalizers.sum(axis=0)
    total_attenuation /= total_attenuation.sum()
    print("Gaussian-weighted near/intermediate/far fractions: "
          + ", ".join(f"{value:.3f}" for value in total_attenuation))
    attenuation_path = output_dir / "gaussian_distance_attenuation.csv"
    with attenuation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", *[f"sensitivity_{name}" for name in DISTANCE_BIN_NAMES]])
        writer.writerows(attenuation_rows)

    mask_dir = DATA_DIR / "preprocessing" / "mri"
    mask_mean_depth = {}
    mask_mean_x = {}
    for patient in comparable_patients:
        mask_path = mask_dir / patient / f"{patient}_gt_norm.nii.gz"
        if mask_path.exists():
            mask = (resample_mask_to_reference(mask_path, template) > 0.5) & brain_mask
            if mask.any():
                mask_mean_depth[patient] = float(depth_mm[mask].mean())
                mask_center = nib.affines.apply_affine(
                    template.affine, np.argwhere(mask).mean(axis=0)
                )
                mask_mean_x[patient] = float(mask_center[0])

    metric_fields = ["patient_id", "target_near", "target_intermediate", "target_far"]
    for method in ("channel", "depth", "distance"):
        metric_fields.extend((f"{method}_center_error_mm", f"{method}_mass_in_mask",
                              f"{method}_equal_volume_dice"))
        metric_fields.extend(f"{method}_depth_{name}" for name in DEPTH_BIN_NAMES)
    metrics_path = output_dir / "oracle_comparison_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for patient in patients:
            distance = distance_targets[patient]
            distance_volume = reconstruct_oracle(
                distance, map_paths, reference, distance_bins_for,
                distance_normalizers, brain_mask,
            )
            distance_path = output_dir / f"{patient}_eeg_v2_distance_oracle.nii.gz"
            distance_image = nib.Nifti1Image(
                distance_volume.astype(np.float32), reference.affine,
                template.header.copy(),
            )
            distance_image.set_data_dtype(np.float32)
            nib.save(distance_image, distance_path)
            row = {"patient_id": patient}
            row.update({f"target_{name}": float(value)
                        for name, value in zip(DISTANCE_BIN_NAMES, distance.sum(axis=0))})
            if patient not in comparable_patients:
                print(f"{patient}: saved distance oracle; channel/depth target unavailable for comparison.")
                writer.writerow(row)
                continue
            mask_path = mask_dir / patient / f"{patient}_gt_norm.nii.gz"
            if not mask_path.exists():
                print(f"{patient}: saved distance oracle; no resection mask at {mask_path}")
                writer.writerow(row)
                continue
            mask = (resample_mask_to_reference(mask_path, template) > 0.5) & brain_mask
            if not mask.any():
                print(f"{patient}: saved distance oracle; resection mask is empty on the MNI grid.")
                writer.writerow(row)
                continue

            channel = channel_targets[patient]
            depth = depth_targets[patient]
            channel_volume = reconstruct_channel_only(channel, map_paths, reference, brain_mask)
            depth_volume = reconstruct_oracle(
                depth, map_paths, reference, depth_bins, depth_normalizers, brain_mask
            )
            volumes = (channel_volume, depth_volume, distance_volume)
            output_names = (
                f"{patient}_eeg_v2_channel_oracle.nii.gz",
                f"{patient}_eeg_v2_depth_oracle.nii.gz",
            )
            for volume, name in zip(volumes, output_names):
                image = nib.Nifti1Image(volume.astype(np.float32), reference.affine,
                                         template.header.copy())
                image.set_data_dtype(np.float32)
                nib.save(image, output_dir / name)

            target_center = nib.affines.apply_affine(
                template.affine, center_of_mass(mask.astype(np.float32))
            )
            for method, volume in zip(("channel", "depth", "distance"), volumes):
                row[f"{method}_center_error_mm"] = float(
                    np.linalg.norm(center_world(volume, template.affine) - target_center)
                )
                row[f"{method}_mass_in_mask"] = float(volume[mask].sum())
                row[f"{method}_equal_volume_dice"] = equal_volume_dice(volume, mask, brain_mask)
                row.update({
                    f"{method}_depth_{name}": float(value)
                    for name, value in zip(DEPTH_BIN_NAMES, depth_distribution(volume, depth_bins))
                })
            writer.writerow(row)
            save_comparison_figure(patient, mask, volumes, template, output_dir)
            print(f"{patient}: distance target near/intermediate/far="
                + ", ".join(f"{value:.3f}" for value in distance.sum(axis=0)))
    print(f"Saved three oracle maps, comparison figures and metrics under {output_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Gaussian distance attenuation: {attenuation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distance_targets_json", type=Path, help="gt_channel_distance.json")
    parser.add_argument("sensitivity_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    reconstruct(args.distance_targets_json, args.sensitivity_dir, args.output_dir)
