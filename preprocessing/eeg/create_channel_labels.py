import json
import csv
import shutil
import sys
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from datasets.eeg import (
    CHANNEL_LABEL_TO_INT,
    DISTANCE_BOUNDARIES_MM,
    DISTANCE_BIN_NAMES,
    DEPTH_BIN_NAMES,
    assert_same_grid,
    compute_depth_map_and_bins,
    compute_electrode_distance_bins,
    load_brain_template,
    print_depth_bin_qc,
    resolve_sensitivity_maps,
    resample_mask_to_reference,
    save_depth_qc,
    validate_electrode_coordinates,
)

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
CHANNEL_THRESHOLD = 0.01
DISTANCE_COORDINATE_FILE = Path(__file__).with_name("standard_1005_mni_coordinates.json")

def create_channel_labels(ground_truth_path, sensitivity_map_dir, output_dir, patient_id, available_channels=None):
    """Create and save the original, thresholded channel distribution."""
    channels = CHANNELS
    if available_channels is not None:
        allowed = {channel.casefold() for channel in available_channels}
        channels = [channel for channel in channels if channel.casefold() in allowed]
    map_paths = resolve_sensitivity_maps(sensitivity_map_dir, channels)
    reference = nib.load(str(map_paths[channels[0]]))
    channel_labels, _, _ = compute_patient_labels(
        ground_truth_path, map_paths, channels, reference
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{patient_id}_channel_labels.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(channel_labels, handle, indent=2)
    tqdm.write(f"Channel labels saved to: {output_file}")
    if channel_labels:
        tqdm.write(f"Number of channels with overlap: {len(channel_labels)}")
    else:
        tqdm.write(f"Warning: No channels with overlap for patient {patient_id}.")
    return channel_labels


def compute_patient_labels(ground_truth_path, map_paths, channels, reference,
                           depth_bins=None, threshold=CHANNEL_THRESHOLD):
    mask = resample_mask_to_reference(ground_truth_path, reference)
    channel_raw = np.zeros(len(channels), dtype=np.float64)
    joint_raw = (np.zeros((len(channels), len(DEPTH_BIN_NAMES)), dtype=np.float64)
                 if depth_bins is not None else None)

    for index, channel in enumerate(channels):
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        if not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError(f"Invalid sensitivity values for {channel}.")
        overlap = mask * sensitivity
        channel_raw[index] = overlap.sum(dtype=np.float64)
        if depth_bins is not None:
            joint_raw[index] = [
                np.sum(overlap * depth_bins[depth_index], dtype=np.float64)
                for depth_index in range(len(DEPTH_BIN_NAMES))
            ]

    if joint_raw is not None and not np.allclose(
        joint_raw.sum(axis=1), channel_raw, rtol=1e-5, atol=1e-7
    ):
        raise ValueError("Depth marginals do not reproduce the raw channel overlaps.")
    total = channel_raw.sum()
    if total <= 0:
        return {}, {}, np.zeros(len(DEPTH_BIN_NAMES))

    keep = channel_raw / total >= threshold
    kept_total = channel_raw[keep].sum()
    if kept_total <= 0:
        return {}, {}, np.zeros(len(DEPTH_BIN_NAMES))
    channel_labels = {
        channel: float(channel_raw[index] / kept_total)
        for index, channel in enumerate(channels) if keep[index]
    }
    def normalize_joint(raw, bin_names):
        raw[~keep] = 0.0
        joint = raw / raw.sum()
        marginals = joint.sum(axis=1)
        expected = np.array([channel_labels.get(channel, 0.0) for channel in channels])
        if not np.allclose(marginals, expected, rtol=1e-5, atol=1e-7):
            raise ValueError(f"Thresholded {bin_names[0]} marginals differ from channel labels.")
        labels = {
            channel: {name: float(joint[index, k]) for k, name in enumerate(bin_names)}
            for index, channel in enumerate(channels)
        }
        return labels, joint.sum(axis=0)

    if joint_raw is None:
        return channel_labels, {}, np.zeros(len(DEPTH_BIN_NAMES))
    depth_labels, depth_marginal = normalize_joint(joint_raw, DEPTH_BIN_NAMES)
    return channel_labels, depth_labels, depth_marginal


def compute_distance_targets(patient_masks, channel_labels, map_paths, channels,
                             electrode_coords, reference, brain_mask,
                             threshold=CHANNEL_THRESHOLD):
    patients = list(patient_masks)
    raw = np.zeros((len(patients), len(channels), len(DISTANCE_BIN_NAMES)), dtype=np.float64)
    for channel_index, channel in enumerate(channels):
        _, distance_bins = compute_electrode_distance_bins(
            electrode_coords[channel], brain_mask, reference.affine
        )
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        if not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError(f"Invalid sensitivity values for {channel}.")
        for patient_index, patient in tqdm(enumerate(patients), desc=f"Processing {channel}"):
            mask = resample_mask_to_reference(patient_masks[patient], reference)
            overlap = mask * sensitivity
            raw[patient_index, channel_index] = [
                np.sum(overlap * distance_bins[bin_index], dtype=np.float64)
                for bin_index in range(len(DISTANCE_BIN_NAMES))
            ]
        del distance_bins, sensitivity

    targets = {}
    marginals = {}
    for patient_index, patient in enumerate(patients):
        channel_raw = raw[patient_index].sum(axis=1)
        if channel_raw.sum() <= 0:
            targets[patient] = {}
            marginals[patient] = np.zeros(len(DISTANCE_BIN_NAMES))
            continue
        keep = channel_raw / channel_raw.sum() >= threshold
        raw[patient_index, ~keep] = 0.0
        kept_raw = raw[patient_index].sum(axis=1)
        expected = np.array([channel_labels[patient].get(channel, 0.0) for channel in channels])
        if not np.allclose(kept_raw / kept_raw.sum(), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Distance marginal differs from channel GT for {patient}.")
        joint = raw[patient_index] / raw[patient_index].sum()
        marginal = joint.sum(axis=0)
        targets[patient] = {
            channel: {
                name: float(joint[channel_index, bin_index])
                for bin_index, name in enumerate(DISTANCE_BIN_NAMES)
            }
            for channel_index, channel in enumerate(channels)
        }
        marginals[patient] = marginal
    return targets, marginals


def load_scalp_coordinates(coordinate_path, template, brain_mask):
    with Path(coordinate_path).open(encoding="utf-8") as handle:
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
    coords = {
        channel: document["data"][channel]["mni"]
        for channel in CHANNELS
    }
    validate_electrode_coordinates(coords, brain_mask, template.affine, template.header)
    return coords


def save_electrode_distance_qc(map_paths, electrode_coords, brain_mask,
                               reference, output_dir):
    import matplotlib.pyplot as plt

    corners = np.array([
        [i, j, k] for i in (0, reference.shape[0] - 1)
        for j in (0, reference.shape[1] - 1)
        for k in (0, reference.shape[2] - 1)
    ])
    world_corners = nib.affines.apply_affine(reference.affine, corners)
    coords = np.array([
        [electrode_coords[channel][axis] for axis in "xyz"] for channel in CHANNELS
    ])
    maximum_distance = np.linalg.norm(
        world_corners[:, None, :] - coords[None, :, :], axis=2
    ).max()
    edges = np.arange(0.0, np.ceil(maximum_distance / 5.0) * 5.0 + 5.0, 5.0)
    brain_histograms = []
    sensitivity_histograms = []
    rows = []
    for channel in CHANNELS:
        distances, bins = compute_electrode_distance_bins(
            electrode_coords[channel], brain_mask, reference.affine
        )
        brain_values = distances[brain_mask]
        image = nib.load(str(map_paths[channel]))
        assert_same_grid(image, reference, f"Sensitivity map for {channel}")
        sensitivity = image.get_fdata(dtype=np.float32)
        if not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
            raise ValueError(f"Invalid sensitivity values for {channel}.")
        sensitivity_values = sensitivity[brain_mask]
        brain_hist = np.histogram(brain_values, bins=edges)[0].astype(float)
        weighted_hist = np.histogram(brain_values, bins=edges, weights=sensitivity_values)[0]
        brain_hist /= brain_hist.sum()
        weighted_hist /= max(weighted_hist.sum(), 1e-12)
        brain_histograms.append(brain_hist)
        sensitivity_histograms.append(weighted_hist)
        brain_fraction = bins[:, brain_mask].mean(axis=1)
        sensitivity_fraction = (bins[:, brain_mask] * sensitivity_values[None]).sum(axis=1)
        sensitivity_fraction /= max(sensitivity_values.sum(), 1e-12)
        rows.append((channel, *brain_fraction, *sensitivity_fraction))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "electrode_distance_qc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "brain_near", "brain_intermediate", "brain_far",
                         "sensitivity_near", "sensitivity_intermediate", "sensitivity_far"])
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for axis, histograms, title in zip(
        axes,
        (brain_histograms, sensitivity_histograms),
        ("Brain-voxel distance histograms", "Sensitivity-weighted distance histograms"),
    ):
        image = axis.imshow(np.asarray(histograms), origin="lower", aspect="auto",
                            extent=[edges[0], edges[-1], -0.5, len(CHANNELS) - 0.5],
                            cmap="magma")
        axis.set_yticks(np.arange(len(CHANNELS)), CHANNELS)
        axis.set_xlabel("Electrode-to-voxel distance (mm)")
        axis.set_title(title)
        for boundary in DISTANCE_BOUNDARIES_MM:
            axis.axvline(boundary, color="cyan", linestyle="--", linewidth=0.8)
        figure.colorbar(image, ax=axis, label="Fraction of histogram mass")
    figure.suptitle("Distance-bin QC; right panel shows Gaussian attenuation")
    figure.savefig(output_dir / "electrode_distance_qc.png", dpi=150, facecolor="white")
    plt.close(figure)

    print("Gaussian sensitivity mass in near/intermediate/far bins:")
    for row in rows:
        print(f"  {row[0]}: {row[4]:.3f}, {row[5]:.3f}, {row[6]:.3f}")
    return csv_path


def main(patients_dir, sensitivity_map_dir, output_dir, channel_output_file=None,
         depth_output_file=None, template_path=None, distance_output_file=None):
    patients_dir = Path(patients_dir)
    sensitivity_map_dir = Path(sensitivity_map_dir)
    output_dir = Path(output_dir)
    gt_dir = output_dir.parent.parent / "gt"
    channel_output_file = Path(channel_output_file or gt_dir / "gt_channels.json")
    depth_output_file = Path(depth_output_file or gt_dir / "gt_channel_depth.json")
    template_path = Path(template_path or Path(__file__).with_name("MNI152_T1_1mm_brain.nii.gz"))
    distance_output_file = Path(distance_output_file or gt_dir / "gt_channel_distance.json")

    map_paths = resolve_sensitivity_maps(sensitivity_map_dir, CHANNELS)
    reference = nib.load(str(map_paths[CHANNELS[0]]))
    template, brain_mask = load_brain_template(template_path)
    assert_same_grid(reference, template, "Sensitivity-map reference")
    electrode_coords = load_scalp_coordinates(DISTANCE_COORDINATE_FILE, template, brain_mask)
    depth_mm, depth_bins = compute_depth_map_and_bins(brain_mask, template.affine)
    bin_means = print_depth_bin_qc(depth_mm, depth_bins, brain_mask)
    if not bin_means[0] < bin_means[1] < bin_means[2]:
        raise ValueError(f"Depth bins are not spatially ordered: {bin_means}.")
    save_depth_qc(depth_mm, depth_bins, brain_mask, template.affine,
                  output_dir / "depth_map_qc.png")
    save_electrode_distance_qc(
        map_paths, electrode_coords, brain_mask, reference, output_dir
    )

    previous_channels = {}
    if channel_output_file.exists():
        with channel_output_file.open(encoding="utf-8") as handle:
            previous_channels = json.load(handle)

    labels = {}
    depth_labels = {}
    patient_masks = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for patient_folder in tqdm(list(patients_dir.iterdir()), desc="Processing patients"):
        if not patient_folder.is_dir() or not patient_folder.name.startswith("RESP"):
            continue
        patient_id = patient_folder.name
        ground_truth_path = patient_folder / f"{patient_id}_gt_norm.nii.gz"
        if not ground_truth_path.exists():
            tqdm.write(f"Warning: Ground truth not found: {ground_truth_path}")
            continue
        try:
            channel_target, depth_target, depth_marginal = compute_patient_labels(
                ground_truth_path, map_paths, CHANNELS, reference, depth_bins
            )
            labels[patient_id] = channel_target
            depth_labels[patient_id] = depth_target
            patient_masks[patient_id] = ground_truth_path
            per_patient_path = output_dir / f"{patient_id}_channel_labels.json"
            with per_patient_path.open("w", encoding="utf-8") as handle:
                json.dump(channel_target, handle, indent=2)
            if depth_target:
                tqdm.write(
                    f"{patient_id} depth marginal: " + ", ".join(
                        f"{name}={value:.3f}"
                        for name, value in zip(DEPTH_BIN_NAMES, depth_marginal)
                    )
                )
            else:
                tqdm.write(f"Warning: No sensitivity overlap for {patient_id}.")
        except Exception as error:
            tqdm.write(f"Error processing patient {patient_id}: {error}")

    comparable = set(labels) & set(previous_channels)
    if comparable:
        maximum_difference = 0.0
        for patient_id in comparable:
            old = {name.casefold(): value for name, value in previous_channels[patient_id].items()}
            new = {name.casefold(): value for name, value in labels[patient_id].items()}
            maximum_difference = max(
                maximum_difference,
                max((abs(old.get(channel.casefold(), 0.0) - new.get(channel.casefold(), 0.0))
                     for channel in CHANNELS), default=0.0),
            )
        tqdm.write(
            "Existing gt_channels comparison (only meaningful for identical maps): "
            f"{len(comparable)} shared patients, max marginal difference={maximum_difference:.6g}"
        )

    distance_labels, distance_marginals = compute_distance_targets(
        patient_masks, labels, map_paths, CHANNELS, electrode_coords,
        reference, brain_mask,
    )
    for patient_id, marginal in distance_marginals.items():
        if distance_labels[patient_id]:
            tqdm.write(
                f"{patient_id} distance marginal: " + ", ".join(
                    f"{name}={value:.3f}"
                    for name, value in zip(DISTANCE_BIN_NAMES, marginal)
                )
            )

    channel_output_file.parent.mkdir(parents=True, exist_ok=True)
    with channel_output_file.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
    with depth_output_file.open("w", encoding="utf-8") as handle:
        json.dump(depth_labels, handle, indent=2)
    with distance_output_file.open("w", encoding="utf-8") as handle:
        json.dump(distance_labels, handle, indent=2)
    shutil.copyfile(channel_output_file, output_dir / "gt_channels.json")
    tqdm.write(f"Saved channel labels: {channel_output_file}")
    tqdm.write(f"Saved joint channel-depth labels: {depth_output_file}")
    tqdm.write(f"Saved joint channel-distance labels: {distance_output_file}")
    return labels

if __name__ == "__main__":
    data_dir = PROJECT_DIR.parent / "data"
    main(
        data_dir / "preprocessing" / "mri",
        data_dir / "preprocessing" / "eeg" / "sensitivity_maps",
        data_dir / "preprocessing" / "eeg" / "channel_labels",
        data_dir / "preprocessing" / "gt" / "gt_channels.json",
        data_dir / "preprocessing" / "gt" / "gt_channel_depth.json",
        distance_output_file=data_dir / "preprocessing" / "gt" / "gt_channel_distance.json",
    )