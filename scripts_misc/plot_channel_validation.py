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


def validate_distribution(values, name, patients):
    invalid = (~np.isfinite(values).all(axis=1)
               | (values < 0).any(axis=1)
               | ~np.isclose(values.sum(axis=1), 1.0, atol=1e-5))
    if invalid.any():
        invalid_patients = [patient for patient, bad in zip(patients, invalid) if bad]
        raise ValueError(f"{name} must be finite, nonnegative distributions summing to one: {invalid_patients}")


def load_predictions(csv_path, gt_path=None):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or "patient_id" not in fields:
        raise ValueError("CSV must contain patient_id and at least one patient row.")
    if len(fields) != len(set(fields)):
        raise ValueError("CSV contains duplicate column names.")
    prediction_columns = {name[5:].casefold(): name for name in fields if name.startswith("pred_")}
    target_columns = {name[7:].casefold(): name for name in fields if name.startswith("target_")}
    if not prediction_columns or prediction_columns.keys() != target_columns.keys():
        raise ValueError("CSV must contain matching pred_<channel> and target_<channel> columns.")
    if len(prediction_columns) != sum(name.startswith("pred_") for name in fields):
        raise ValueError("Prediction channel names must be unique ignoring case.")
    if len(target_columns) != sum(name.startswith("target_") for name in fields):
        raise ValueError("Target channel names must be unique ignoring case.")
    channels = [name[5:] for name in prediction_columns.values()]
    patients = [row["patient_id"] for row in rows]
    if any(not patient.strip() for patient in patients) or len(set(patients)) != len(patients):
        raise ValueError("Patient IDs must be nonempty and unique.")
    predictions = np.array([[float(row[name]) for name in prediction_columns.values()] for row in rows])
    targets = np.array([[float(row[target_columns[name]]) for name in prediction_columns] for row in rows])
    validate_distribution(predictions, "Predictions", patients)
    validate_distribution(targets, "Targets", patients)
    reference = None
    if gt_path is not None:
        with gt_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        reference_rows = []
        for patient in patients:
            if patient not in raw or not isinstance(raw[patient], dict):
                raise ValueError(f"Reference GT lacks a channel distribution for {patient}.")
            values = {}
            for name, value in raw[patient].items():
                key = name.casefold()
                if key not in prediction_columns:
                    raise ValueError(f"Reference channel {name!r} for {patient} is absent from CSV.")
                values[key] = values.get(key, 0.0) + float(value)
            reference_rows.append([values.get(name, 0.0) for name in prediction_columns])
        reference = np.array(reference_rows)
        validate_distribution(reference, "Reference GT", patients)
    return patients, channels, predictions, targets, reference


def plot_page(patients, channels, predictions, targets, reference, title, output, dpi):
    count = len(patients)
    positions = np.arange(count)
    overlap = np.minimum(predictions, targets).sum(axis=1)
    predicted_top = predictions.argmax(axis=1)
    target_top = targets.argmax(axis=1)
    columns = 3 if reference is not None else 2
    figure = plt.figure(figsize=(6.3 * columns, max(8, 3.8 + 0.57 * count)), layout="constrained")
    grid = figure.add_gridspec(2, columns, height_ratios=[1.15, 1])
    heatmap_axes = []
    panels = [(targets, "CSV target"), (predictions, "Prediction")]
    if reference is not None:
        panels.append((reference, "Reference GT"))
    for column, (values, heading) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        heatmap_axes.append(axis)
        image = axis.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(np.arange(len(channels)), channels, rotation=90)
        axis.set_yticks(positions, patients)
        axis.set_yticks(np.arange(count + 1) - 0.5, minor=True)
        axis.grid(which="minor", axis="y", color="white", linewidth=1)
        axis.tick_params(which="minor", left=False)
        axis.set_title(heading, loc="left", weight="bold", pad=12)
        for patient_index, values_row in enumerate(values):
            channel_index = values_row.argmax()
            axis.text(channel_index, patient_index, f"{values_row[channel_index]:.2f}",
                      ha="center", va="center", fontsize=8,
                      color="white" if values_row[channel_index] > 0.55 else "black")
    figure.colorbar(image, ax=heatmap_axes, label="Channel probability", shrink=0.65, pad=0.02)
    agreement = figure.add_subplot(grid[1, 0])
    matches = predicted_top == target_top
    agreement.barh(positions, overlap, color=["#168b82" if match else "#ba6357" for match in matches])
    for index, value in enumerate(overlap):
        agreement.text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9)
    agreement.set_yticks(positions, [
        f"{patient}  {channels[target_top[index]]} > {channels[predicted_top[index]]}"
        for index, patient in enumerate(patients)
    ])
    agreement.set_xlim(0, 1.12)
    agreement.set_xticks([0, 0.25, 0.5, 0.75, 1])
    agreement.invert_yaxis()
    agreement.set_title("Distribution overlap", loc="left", weight="bold", pad=12)
    agreement.set_xlabel("Higher is better | row label: target > prediction")
    agreement.legend(handles=[Patch(color="#168b82", label="Top channel matches"),
                              Patch(color="#ba6357", label="Top channel differs")],
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
    for values, offset in [(targets, -0.18), (predictions, 0.18)]:
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
    if reference is not None:
        drift_axis = figure.add_subplot(grid[1, 2])
        drift = 0.5 * np.abs(targets - reference).sum(axis=1)
        drift_axis.barh(positions, drift, color="#b25362")
        for index, value in enumerate(drift):
            drift_axis.text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9)
        drift_axis.set_yticks(positions, patients)
        drift_axis.invert_yaxis()
        drift_axis.set_xlim(0, 1.12)
        drift_axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        drift_axis.set_title("CSV target vs reference GT", loc="left", weight="bold", pad=12)
        drift_axis.set_xlabel("Total variation: 0 = identical; 1 = disjoint")
    figure.suptitle(f"{title}\n{count} patients | top-channel matches {matches.sum()}/{count}"
                   f" | mean overlap {overlap.mean():.2f}", fontsize=15, weight="bold")
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="CSV containing patient_id, pred_<channel>, target_<channel>.")
    parser.add_argument("--gt-json", type=Path, help="Optional patient-to-channel GT JSON for target auditing.")
    parser.add_argument("--output", type=Path, help="Output PNG path; defaults to data/tmp/<CSV stem>_diagnostic.png.")
    parser.add_argument("--sort", choices=["worst", "patient", "input"], default="worst")
    parser.add_argument("--patients-per-page", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    if args.patients_per_page < 1 or args.dpi < 1:
        parser.error("--patients-per-page and --dpi must be positive.")
    try:
        patients, channels, predictions, targets, reference = load_predictions(args.csv_path, args.gt_json)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    overlap = np.minimum(predictions, targets).sum(axis=1)
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
            plot_page([patients[index] for index in selected], channels, predictions[selected], targets[selected],
                      reference[selected] if reference is not None else None, title, page_output, args.dpi)
            print(f"Saved: {page_output}")
    matches = (predictions.argmax(axis=1) == targets.argmax(axis=1)).sum()
    print(f"All {len(patients)} patients: top-channel matches {matches}/{len(patients)}, mean overlap {overlap.mean():.3f}")


if __name__ == "__main__":
    main()