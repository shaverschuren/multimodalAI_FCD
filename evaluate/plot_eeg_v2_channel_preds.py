"""Plot patient-level EEG channel predictions from pred_*/target_* CSV columns.

Example:
    python scripts_misc/plot_channel_validation.py val_predictions.csv --gt-json ../data/preprocessing/gt/gt_channels.json
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BIN_GROUPS = (
    ("near", "intermediate", "far"),
    ("superficial", "intermediate", "deep"),
)


def split_label(name):
    normalized = name.casefold()
    for bins in BIN_GROUPS:
        for bin_name in bins:
            suffix = f"_{bin_name}"
            if normalized.endswith(suffix):
                return name[:-len(suffix)], bin_name
    return name, ""


def validate_distribution(values, name, patients):
    values = values.reshape(len(patients), -1)
    invalid = (~np.isfinite(values).all(axis=1)
               | (values < 0).any(axis=1)
               | ~np.isclose(values.sum(axis=1), 1.0, atol=1e-5))
    if invalid.any():
        invalid_patients = [patient for patient, bad in zip(patients, invalid) if bad]
        raise ValueError(f"{name} must be finite, nonnegative distributions summing to one: {invalid_patients}")


def load_predictions(csv_path, gt_path=None, depth_gt_path=None):
    if csv_path.is_dir():
        csv_paths = sorted(set(csv_path.rglob("test_predictions.csv"))
                           | set(csv_path.rglob("test_prediction.csv")))
    else:
        csv_paths = [csv_path]
    if not csv_paths:
        raise ValueError(f"No test_predictions.csv files found under {csv_path}.")
    rows = []
    fields = None
    for prediction_path in csv_paths:
        with prediction_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            file_fields = reader.fieldnames or []
            if fields is None:
                fields = file_fields
            elif file_fields != fields:
                raise ValueError(f"CSV schema differs in {prediction_path}.")
            for row in reader:
                rows.append((row, prediction_path))
    fields = fields or []
    if "patient_id" not in fields:
        raise ValueError("CSV must contain a patient_id column.")
    if len(fields) != len(set(fields)):
        raise ValueError("CSV contains duplicate column names.")
    prediction_columns = [name for name in fields if name.startswith("pred_")]
    target_columns = [name for name in fields if name.startswith("target_")]
    if not prediction_columns or not target_columns:
        raise ValueError("CSV must contain pred_<channel> and target_<channel> columns.")
    predictions_by_label = {}
    targets_by_label = {}
    depth_predictions_by_bin = {}
    depth_targets_by_bin = {}
    channel_names = {}
    bin_names = []
    for prefix, columns, destination, depth_destination in (
        ("pred_", prediction_columns, predictions_by_label, depth_predictions_by_bin),
        ("target_", target_columns, targets_by_label, depth_targets_by_bin),
    ):
        for column in columns:
            channel, bin_name = split_label(column[len(prefix):])
            if channel.casefold() == "depth":
                if not bin_name or bin_name in depth_destination:
                    raise ValueError(f"Duplicate or invalid global-depth column {column!r}.")
                depth_destination[bin_name] = column
                continue
            label = (channel.casefold(), bin_name)
            if not channel or label in destination:
                raise ValueError(f"Duplicate or invalid output column {column!r}.")
            destination[label] = column
            channel_names.setdefault(channel.casefold(), channel)
            if bin_name and bin_name not in bin_names:
                bin_names.append(bin_name)
    has_depth = bool(depth_predictions_by_bin)
    if (predictions_by_label.keys() != targets_by_label.keys()
            or depth_predictions_by_bin.keys() != depth_targets_by_bin.keys()
            or has_depth != bool(depth_targets_by_bin)):
        raise ValueError("Prediction and target channel/bin columns must match.")
    if has_depth:
        if bin_names:
            raise ValueError("Factorized global-depth columns cannot be mixed with joint channel/bin columns.")
        depth_bins = next((group for group in BIN_GROUPS
                           if set(depth_predictions_by_bin) == set(group)), None)
        if depth_bins is None or len(depth_predictions_by_bin) != 3:
            raise ValueError(f"Expected a recognized three-bin global-depth scheme, got {list(depth_predictions_by_bin)}.")
        bins = [""]
        channels = list(channel_names.values())
        labels = [(channel.casefold(), "") for channel in channels]
        if predictions_by_label.keys() != set(labels):
            raise ValueError("Factorized channel columns must contain one scalar per electrode.")
    elif bin_names:
        expected_bins = next((group for group in BIN_GROUPS if set(bin_names) == set(group)), None)
        if expected_bins is None or len(bin_names) != 3:
            raise ValueError(f"Expected one recognized three-bin scheme, got {bin_names}.")
        bins = list(expected_bins)
        channels = list(channel_names.values())
        missing = [channel for channel in channels for bin_name in bins
                   if (channel.casefold(), bin_name) not in predictions_by_label]
        if missing:
            raise ValueError(f"Each electrode needs all three bins; incomplete labels: {missing}.")
    else:
        bins = [""]
        channels = list(channel_names.values())
    labels = [(channel.casefold(), bin_name) for channel in channels for bin_name in bins]
    valid_rows = []
    seen_patients = set()
    duplicate_count = unlabelled_count = 0
    for row, source_path in rows:
        patient = row.get("patient_id", "").strip()
        if not patient:
            raise ValueError(f"Empty patient_id in {source_path}.")
        prediction_values = [row.get(predictions_by_label[label], "").strip()
                             for label in labels]
        if has_depth:
            prediction_values.extend(row.get(depth_predictions_by_bin[name], "").strip()
                                     for name in depth_bins)
        if any(not value for value in prediction_values):
            raise ValueError(f"Blank prediction value for {patient} in {source_path}.")
        target_values = [row.get(targets_by_label[label], "").strip() for label in labels]
        if has_depth:
            target_values.extend(row.get(depth_targets_by_bin[name], "").strip()
                                 for name in depth_bins)
        if all(not value for value in target_values):
            unlabelled_count += 1
            continue
        if any(not value for value in target_values):
            raise ValueError(f"Partially blank target distribution for {patient} in {source_path}.")
        if patient in seen_patients:
            duplicate_count += 1
            continue
        seen_patients.add(patient)
        valid_rows.append(row)
    if unlabelled_count:
        print(f"Skipped {unlabelled_count} rows with predictions but no target labels.")
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate labeled patients; kept first sorted occurrence.")
    rows = valid_rows
    if not rows:
        raise ValueError("No labeled prediction rows remain after filtering blank test targets.")
    patients = [row["patient_id"] for row in rows]
    if len(set(patients)) != len(patients):
        raise ValueError("Patient IDs must be nonempty and unique.")
    predictions = np.array([[float(row[predictions_by_label[label]]) for label in labels]
                             for row in rows]).reshape(len(rows), len(channels), len(bins))
    targets = np.array([[float(row[targets_by_label[label]]) for label in labels]
                        for row in rows]).reshape(len(rows), len(channels), len(bins))
    validate_distribution(predictions, "Predictions", patients)
    validate_distribution(targets, "Targets", patients)
    depth_predictions = depth_targets = depth_reference = None
    depth_bin_names = None
    if has_depth:
        depth_bin_names = list(depth_bins)
        depth_predictions = np.array([
            [float(row[depth_predictions_by_bin[name]]) for name in depth_bin_names]
            for row in rows
        ])
        depth_targets = np.array([
            [float(row[depth_targets_by_bin[name]]) for name in depth_bin_names]
            for row in rows
        ])
        validate_distribution(depth_predictions, "Global-depth predictions", patients)
        validate_distribution(depth_targets, "Global-depth targets", patients)
        if depth_gt_path is not None:
            with depth_gt_path.open(encoding="utf-8") as handle:
                raw_depth = json.load(handle)
            reference_rows = []
            depth_name_lookup = {name.casefold(): name for name in depth_bin_names}
            for patient in patients:
                if patient not in raw_depth or not isinstance(raw_depth[patient], dict):
                    raise ValueError(f"Global-depth reference GT lacks a distribution for {patient}.")
                entry = {name.casefold(): value for name, value in raw_depth[patient].items()}
                if set(entry) != set(depth_name_lookup):
                    raise ValueError(f"Global-depth reference bins differ for {patient}.")
                reference_rows.append([float(entry[name.casefold()]) for name in depth_bin_names])
            depth_reference = np.asarray(reference_rows, dtype=np.float64)
            validate_distribution(depth_reference, "Global-depth reference GT", patients)
    elif depth_gt_path is not None:
        raise ValueError("--depth-gt-json requires factorized global-depth CSV columns.")
    reference = None
    if gt_path is not None:
        with gt_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        reference_rows = []
        reference_is_joint = None
        for patient in patients:
            if patient not in raw or not isinstance(raw[patient], dict):
                raise ValueError(f"Reference GT lacks a channel distribution for {patient}.")
            entry = raw[patient]
            is_joint = any(isinstance(value, dict) for value in entry.values())
            if any(isinstance(value, dict) != is_joint for value in entry.values()):
                raise ValueError(f"Mixed channel-only and joint labels in reference GT for {patient}.")
            if reference_is_joint is None:
                reference_is_joint = is_joint
            elif reference_is_joint != is_joint:
                raise ValueError("Reference GT mixes channel-only and joint target formats.")
            channel_values = {name.casefold(): value for name, value in entry.items()}
            unknown = set(channel_values) - set(channel_names)
            if unknown:
                raise ValueError(f"Reference channels absent from CSV for {patient}: {sorted(unknown)}")
            if is_joint:
                joint = np.zeros((len(channels), len(bins)), dtype=np.float64)
                for channel_index, channel in enumerate(channels):
                    bin_values = channel_values.get(channel.casefold(), {})
                    for bin_index, bin_name in enumerate(bins):
                        if bin_name and bin_name not in bin_values:
                            raise ValueError(f"Reference GT lacks {channel}/{bin_name} for {patient}.")
                        if bin_name:
                            joint[channel_index, bin_index] = float(bin_values[bin_name])
                        elif bin_values:
                            joint[channel_index, 0] = sum(float(value) for value in bin_values.values())
                reference_rows.append(joint.sum(axis=1) if bins == [""] else joint)
            else:
                reference_rows.append([
                    float(channel_values.get(channel.casefold(), 0.0))
                    for channel in channels
                ])
        reference = np.array(reference_rows)
        validate_distribution(reference, "Reference GT", patients)
    return (patients, channels, bins, predictions, targets, reference,
            depth_bin_names, depth_predictions, depth_targets, depth_reference)


def plot_page(patients, channels, bins, predictions, targets, reference, title, output, dpi,
              depth_bin_names=None, depth_predictions=None, depth_targets=None,
              depth_reference=None):
    count = len(patients)
    positions = np.arange(count)
    channel_predictions = predictions.sum(axis=2)
    channel_targets = targets.sum(axis=2)
    overlap = np.minimum(predictions, targets).sum(axis=(1, 2))
    predicted_top = predictions.reshape(count, -1).argmax(axis=1)
    target_top = targets.reshape(count, -1).argmax(axis=1)
    bin_count = len(bins)
    category_names = [
        f"{channel}-{bin_name[0].upper()}" if bin_name else channel
        for channel in channels for bin_name in bins
    ]

    def label_at(index):
        channel = channels[index // bin_count]
        bin_name = bins[index % bin_count]
        return f"{channel}/{bin_name}" if bin_name else channel

    panels = [(targets.reshape(count, -1), category_names, "CSV target"),
              (predictions.reshape(count, -1), category_names, "Prediction")]
    depth_rows_per_patient = 0
    if depth_predictions is not None:
        depth_values = [depth_targets, depth_predictions]
        depth_row_names = ["target", "prediction"]
        if depth_reference is not None:
            depth_values.append(depth_reference)
            depth_row_names.append("reference")
        depth_rows_per_patient = len(depth_values)
        depth_matrix = np.stack(depth_values, axis=1).reshape(
            count * depth_rows_per_patient, len(depth_bin_names))
        panels.append((depth_matrix, depth_bin_names, "Global depth distributions"))
    if reference is not None:
        if reference.ndim == 3:
            panels.append((reference.reshape(count, -1), category_names, "Reference joint GT"))
        else:
            panels.append((reference, channels, "Reference channel GT"))
    if depth_predictions is not None:
        bin_labels = depth_bin_names
        bin_targets = depth_targets
        bin_predictions = depth_predictions
    elif bin_count > 1:
        bin_labels = bins
        bin_targets = targets.sum(axis=1)
        bin_predictions = predictions.sum(axis=1)
    else:
        bin_labels = bin_targets = bin_predictions = None

    bottom_column = 2
    bin_summary_column = None
    if bin_labels is not None:
        bin_summary_column = bottom_column
        bottom_column += 1
    depth_agreement_column = None
    if depth_predictions is not None:
        depth_agreement_column = bottom_column
        bottom_column += 1
    reference_drift_column = None
    if reference is not None:
        reference_drift_column = bottom_column
        bottom_column += 1
    columns = max(len(panels), bottom_column)
    figure = plt.figure(figsize=(6.3 * columns, max(8, 3.8 + 0.57 * count)), layout="constrained")
    grid = figure.add_gridspec(2, columns, height_ratios=[1.15, 1])
    heatmap_axes = []
    joint_vmax = max(float(targets.max()), float(predictions.max()), 1e-8)
    joint_image = reference_image = depth_image = None
    reference_axis = None
    depth_axis = None
    for column, (values, labels, heading) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        heatmap_axes.append(axis)
        is_reference = heading.startswith("Reference")
        is_depth_panel = heading == "Global depth distributions"
        vmax = max(float(values.max()), 1e-8) if is_reference or is_depth_panel else joint_vmax
        image = axis.imshow(values, vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto")
        if is_reference:
            reference_image = image
            reference_axis = axis
        elif is_depth_panel:
            depth_image = image
            depth_axis = axis
        else:
            joint_image = image
        if is_depth_panel:
            axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
            axis.set_yticks(
                np.arange(count) * depth_rows_per_patient + (depth_rows_per_patient - 1) / 2,
                patients,
            )
            axis.set_yticks(
                np.arange(1, count) * depth_rows_per_patient - 0.5, minor=True)
            axis.grid(which="minor", axis="y", color="white", linewidth=1)
            axis.set_ylabel("Rows per patient: " + ", ".join(depth_row_names))
        elif bin_count > 1 and not is_reference:
            axis.set_xticks(np.arange(len(labels)),
                            [name[0].upper() for _ in channels for name in bins],
                            fontsize=7)
            centers = np.arange(len(channels)) * bin_count + (bin_count - 1) / 2
            electrode_axis = axis.secondary_xaxis("top")
            electrode_axis.set_xticks(centers, channels)
            electrode_axis.tick_params(axis="x", labelsize=7, length=0, pad=3)
            axis.set_xlabel("Electrode (top) / bin (bottom)")
        else:
            axis.set_xticks(np.arange(len(labels)), labels, rotation=90)
        if not is_depth_panel:
            axis.set_yticks(positions, patients)
            axis.set_yticks(np.arange(count + 1) - 0.5, minor=True)
            axis.grid(which="minor", axis="y", color="white", linewidth=1)
        axis.tick_params(which="minor", left=False)
        axis.set_title(heading, loc="left", weight="bold",
                   pad=32 if bin_count > 1 and not is_reference else 12)
        for patient_index, values_row in enumerate(values):
            channel_index = values_row.argmax()
            axis.text(channel_index, patient_index, f"{values_row[channel_index]:.2f}",
                      ha="center", va="center", fontsize=8,
                      color="white" if values_row[channel_index] > 0.55 else "black")
    figure.colorbar(joint_image, ax=heatmap_axes[:2],
                    label="Joint probability mass", shrink=0.65, pad=0.02)
    if depth_image is not None:
        figure.colorbar(depth_image, ax=depth_axis,
                        label="Global-depth probability", shrink=0.65, pad=0.02)
    if reference_image is not None:
        figure.colorbar(reference_image, ax=reference_axis,
                        label="Channel marginal", shrink=0.65, pad=0.02)
    agreement = figure.add_subplot(grid[1, 0])
    matches = predicted_top == target_top
    agreement.barh(positions, overlap, color=["#168b82" if match else "#ba6357" for match in matches])
    for index, value in enumerate(overlap):
        agreement.text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9)
    agreement.set_yticks(positions, [
        f"{patient}  {label_at(target_top[index])} > {label_at(predicted_top[index])}"
        for index, patient in enumerate(patients)
    ])
    agreement.set_xlim(0, 1.12)
    agreement.set_xticks([0, 0.25, 0.5, 0.75, 1])
    agreement.invert_yaxis()
    agreement.set_title("Distribution overlap", loc="left", weight="bold", pad=12)
    agreement.set_xlabel("Higher is better | row label: target > prediction")
    category_label = "channel/bin" if bin_count > 1 else "channel"
    agreement.legend(handles=[Patch(color="#168b82", label=f"Top {category_label} matches"),
                              Patch(color="#ba6357", label=f"Top {category_label} differs")],
                     loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    side_axis = figure.add_subplot(grid[1, 1])
    categories = [
        ("Left", "#168b82", [index for index, name in enumerate(channels) if name[-1:] in "13579"]),
        ("Midline", "#999999", [index for index, name in enumerate(channels) if name.casefold().endswith("z")]),
        ("Right", "#dc7452", [index for index, name in enumerate(channels) if name[-1:] in "02468"]),
    ]
    assigned = {index for _, _, indices in categories for index in indices}
    if len(assigned) != len(channels):
        categories.append(("Other", "#738ca5", [index for index in range(len(channels)) if index not in assigned]))
    for values, offset in [(channel_targets, -0.18), (channel_predictions, 0.18)]:
        left_edge = np.zeros(count)
        for _, color, indices in categories:
            mass = values[:, indices].sum(axis=1)
            side_axis.barh(positions + offset, mass, height=0.30, left=left_edge, color=color)
            left_edge += mass
    side_axis.set_yticks(positions, patients)
    side_axis.invert_yaxis()
    side_axis.set_xlim(0, 1)
    side_axis.set_title("Left / midline / right", loc="left", weight="bold", pad=12)
    side_axis.set_xlabel("Target above; prediction below")
    side_axis.legend(handles=[Patch(color=color, label=name) for name, color, _ in categories],
                     ncol=len(categories), loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    if bin_summary_column is not None:
        bin_axis = figure.add_subplot(grid[1, bin_summary_column])
        bin_colors = ["#168b82", "#d2a84a", "#dc7452"]
        for values, offset in [(bin_targets, -0.18), (bin_predictions, 0.18)]:
            left_edge = np.zeros(count)
            for bin_index, color in enumerate(bin_colors[:len(bin_labels)]):
                mass = values[:, bin_index]
                bin_axis.barh(positions + offset, mass, height=0.30,
                              left=left_edge, color=color)
                left_edge += mass
        bin_axis.set_yticks(positions, patients)
        bin_axis.invert_yaxis()
        bin_axis.set_xlim(0, 1)
        bin_axis.set_title("Per-bin mass", loc="left", weight="bold", pad=12)
        bin_axis.set_xlabel("Target above; prediction below")
        bin_axis.legend(
            handles=[Patch(color=color, label=name)
                     for color, name in zip(bin_colors, bin_labels)],
            ncol=len(bin_labels), loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=False, fontsize=9,
        )
    depth_overlap = None
    if depth_predictions is not None:
        depth_overlap = np.minimum(depth_predictions, depth_targets).sum(axis=1)
        depth_matches = depth_predictions.argmax(axis=1) == depth_targets.argmax(axis=1)
        depth_agreement = figure.add_subplot(grid[1, depth_agreement_column])
        depth_agreement.barh(
            positions, depth_overlap,
            color=["#168b82" if match else "#ba6357" for match in depth_matches],
        )
        for index, value in enumerate(depth_overlap):
            depth_agreement.text(value + 0.015, index, f"{value:.2f}",
                                 va="center", fontsize=9)
        depth_agreement.set_yticks(positions, patients)
        depth_agreement.invert_yaxis()
        depth_agreement.set_xlim(0, 1.12)
        depth_agreement.set_xticks([0, 0.25, 0.5, 0.75, 1])
        depth_agreement.set_title("Global-depth overlap", loc="left", weight="bold", pad=12)
        depth_agreement.set_xlabel("Higher is better")
        depth_agreement.legend(
            handles=[Patch(color="#168b82", label="Top depth bin matches"),
                     Patch(color="#ba6357", label="Top depth bin differs")],
            loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9,
        )
    if reference is not None:
        drift_axis = figure.add_subplot(grid[1, reference_drift_column])
        if reference.ndim == 3:
            drift = 0.5 * np.abs(targets - reference).sum(axis=(1, 2))
        else:
            drift = 0.5 * np.abs(channel_targets - reference).sum(axis=1)
        drift_axis.barh(positions, drift, color="#b25362")
        for index, value in enumerate(drift):
            drift_axis.text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9)
        drift_axis.set_yticks(positions, patients)
        drift_axis.invert_yaxis()
        drift_axis.set_xlim(0, 1.12)
        drift_axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        reference_level = "joint" if reference.ndim == 3 else "channel"
        drift_axis.set_title(f"CSV target vs reference {reference_level} GT",
                     loc="left", weight="bold", pad=12)
        drift_axis.set_xlabel("Total variation: 0 = identical; 1 = disjoint")
    agreement_name = "top channel/bin" if bin_count > 1 else "top-channel"
    depth_summary = f" | mean depth overlap {depth_overlap.mean():.2f}" if depth_overlap is not None else ""
    figure.suptitle(f"{title}\n{count} patients | {agreement_name} matches {matches.sum()}/{count}"
                   f" | mean overlap {overlap.mean():.2f}{depth_summary}", fontsize=15, weight="bold")
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", type=Path,
                        help="Prediction CSV or folder of fold runs containing test_predictions.csv.")
    parser.add_argument("--gt-json", type=Path, help="Optional patient-to-channel GT JSON for target auditing.")
    parser.add_argument("--depth-gt-json", type=Path,
                        help="Optional patient-to-global-depth GT JSON for factorized target auditing.")
    parser.add_argument("--output", type=Path, help="Output PNG path; defaults to data/tmp/<CSV stem>_diagnostic.png.")
    parser.add_argument("--sort", choices=["worst", "patient", "input"], default="worst")
    parser.add_argument("--patients-per-page", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    if args.patients_per_page < 1 or args.dpi < 1:
        parser.error("--patients-per-page and --dpi must be positive.")
    try:
        (patients, channels, bins, predictions, targets, reference,
         depth_bin_names, depth_predictions, depth_targets,
         depth_reference) = load_predictions(args.csv_path, args.gt_json,
                                             args.depth_gt_json)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    overlap = np.minimum(predictions, targets).sum(axis=(1, 2))
    order = np.argsort(overlap, kind="stable") if args.sort == "worst" else (
        np.argsort(patients, kind="stable") if args.sort == "patient" else np.arange(len(patients)))
    output = args.output or (Path(__file__).resolve().parents[2] / "data" / "tmp"
                             / f"{args.csv_path.stem}_diagnostic.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    page_count = (len(patients) + args.patients_per_page - 1) // args.patients_per_page
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for page in range(page_count):
            selected = order[page * args.patients_per_page:(page + 1) * args.patients_per_page]
            page_output = output if page_count == 1 else output.with_name(f"{output.stem}_page{page + 1:02d}{output.suffix}")
            title = args.csv_path.stem + (f" | page {page + 1}/{page_count}" if page_count > 1 else "")
            plot_page(
                [patients[index] for index in selected], channels, bins,
                predictions[selected], targets[selected],
                reference[selected] if reference is not None else None,
                title, page_output, args.dpi,
                depth_bin_names,
                depth_predictions[selected] if depth_predictions is not None else None,
                depth_targets[selected] if depth_targets is not None else None,
                depth_reference[selected] if depth_reference is not None else None,
            )
            print(f"Saved: {page_output}")
        matches = (predictions.reshape(len(patients), -1).argmax(axis=1)
               == targets.reshape(len(patients), -1).argmax(axis=1)).sum()
        category = "channel/bin" if len(bins) > 1 else "channel"
        print(f"All {len(patients)} patients: top-{category} matches {matches}/{len(patients)}, "
            f"mean overlap {overlap.mean():.3f}")
        if depth_predictions is not None:
            depth_overlap = np.minimum(depth_predictions, depth_targets).sum(axis=1)
            print(f"Global depth: mean overlap {depth_overlap.mean():.3f}, "
                  f"top-bin matches "
                  f"{(depth_predictions.argmax(axis=1) == depth_targets.argmax(axis=1)).sum()}/{len(patients)}")
            if depth_reference is not None:
                drift = 0.5 * np.abs(depth_targets - depth_reference).sum(axis=1)
                print(f"Global-depth CSV target vs reference GT: mean total variation {drift.mean():.3f}")


if __name__ == "__main__":
    main()