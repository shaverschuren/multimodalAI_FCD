"""Compare the MRI-only and multimodal late-fusion evaluation results.

The script writes patient-level bootstrap summaries, paired-comparison results,
and one publication figure. Continuous paired outcomes use sign-flip permutation
tests; binary outcomes also report an exact binomial sign test. Permutation
p-values are reported and shown without multiplicity correction.

Usage:
    python evaluate/plot_evaluate_maps_results.py --mri-json MRI.json \\
        --multimodal-json MULTIMODAL.json --output-dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binomtest


# Statistical settings used for the saved comparison results.
BOOTSTRAP_REPLICATES = 10_000
PERMUTATIONS = 10_000
CI_LEVEL = 0.95
RANDOM_SEED = 42

MRI_COLOR = "#A1C5CB"
MULTIMODAL_COLOR = "#0066B3"
DIFF_POINT_COLOR = "#E4B8B4"
DIFF_MEAN_COLOR = "#A6192E"
NEUTRAL_COLOR = "#666666"

BAR_WIDTH = 0.34
DIFF_CATEGORY_SPACING = 1.25
ERRORBAR_CAPSIZE = 4
RIGHT_COLUMN_WIDTH_RATIO = 3.45 / 3.2
OUTER_COLUMN_GAP_RATIO = 0.08
MIDDLE_COLUMN_GAP_RATIO = 0.016
SIGNIFICANCE_BRACKET_HALF_WIDTH_PT = 9.5
SIGNIFICANCE_BRACKET_HEIGHT_PT = 5.5
SIGNIFICANCE_BRACKET_GAP_PT = 5.0
SIGNIFICANCE_LABEL_CLEARANCE_PT = 12.0


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _harmonic_f1(precision: Optional[float], recall: Optional[float]) -> float:
    if precision is None or recall is None:
        return 0.0
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denominator)


def _load_subject_rows(json_path: Path, run_name: str) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows: List[Dict[str, Any]] = []
    for entry in payload.get("per_subject", []):
        fixed = entry.get("fixed_threshold_metrics", {})
        voxel = fixed.get("voxel", {})
        cluster = fixed.get("cluster", {})
        detection = cluster.get("detection", {})
        pinpointing = cluster.get("pinpointing", {})

        det_precision = _as_float(detection.get("cluster_precision"))
        det_recall = _as_float(detection.get("cluster_sensitivity"))
        pin_precision = _as_float(pinpointing.get("cluster_precision"))
        pin_recall = _as_float(pinpointing.get("cluster_sensitivity"))
        voxel_precision = _as_float(voxel.get("voxel_precision"))
        voxel_recall = _as_float(voxel.get("voxel_recall"))
        voxel_dice = _as_float(voxel.get("voxel_dice"))
        if voxel_dice is None and (voxel_precision is None or voxel_recall is None):
            voxel_dice = 0.0

        det_f1 = _as_float(detection.get("cluster_f1"))
        if det_f1 is None:
            det_f1 = _harmonic_f1(det_precision, det_recall)
        pin_f1 = _as_float(pinpointing.get("cluster_f1"))
        if pin_f1 is None:
            pin_f1 = _harmonic_f1(pin_precision, pin_recall)

        rows.append(
            {
                "run": run_name,
                "subject_id": entry.get("subject_id"),
                "is_control": bool(entry.get("is_control", False)),
                "voxel_dice": voxel_dice,
                "voxel_precision": voxel_precision,
                "voxel_recall": voxel_recall,
                "cluster_det_precision": det_precision,
                "cluster_det_recall": det_recall,
                "cluster_det_f1": det_f1,
                "cluster_pin_precision": pin_precision,
                "cluster_pin_recall": pin_recall,
                "cluster_pin_f1": pin_f1,
                "subject_detected": 1.0 if bool(detection.get("subject_detected", False)) else 0.0,
                "subject_pinpointed": 1.0 if bool(pinpointing.get("subject_pinpointed", False)) else 0.0,
                "n_pred_clusters": _as_float(cluster.get("n_pred_clusters")),
                "n_fp_det_clusters": _as_float(detection.get("n_fp_pred_clusters")),
                "n_fp_pin_clusters": _as_float(pinpointing.get("n_fp_pred_clusters")),
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    seed: int = RANDOM_SEED,
) -> Optional[List[float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if values.size == 1:
        value = float(values[0])
        return [value, value]

    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        bootstrap_sample = values[rng.integers(0, values.size, size=values.size)]
        if statistic == "median":
            samples[index] = float(np.median(bootstrap_sample))
        else:
            samples[index] = float(np.mean(bootstrap_sample))

    alpha = (1.0 - CI_LEVEL) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return [float(lower), float(upper)]


def _sign_flip_pvalue(differences: np.ndarray, seed: int = RANDOM_SEED) -> Optional[float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return None

    observed_mean = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    permuted_means = np.empty(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size, replace=True)
        permuted_means[index] = float(np.mean(differences * signs))

    exceedances = np.sum(np.abs(permuted_means) >= abs(observed_mean))
    return float((exceedances + 1) / (PERMUTATIONS + 1))


def _exact_binomial_sign_test(baseline: np.ndarray, comparator: np.ndarray) -> Dict[str, Any]:
    baseline = np.asarray(baseline, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    valid = np.isfinite(baseline) & np.isfinite(comparator)
    baseline = baseline[valid]
    comparator = comparator[valid]

    gained = int(np.sum((baseline == 0.0) & (comparator == 1.0)))
    lost = int(np.sum((baseline == 1.0) & (comparator == 0.0)))
    if gained + lost == 0:
        return {"test": "unavailable", "p_value": None}

    p_value = binomtest(gained, gained + lost, p=0.5, alternative="two-sided").pvalue
    return {"test": "sign_test_binomial", "p_value": float(p_value)}


def _patient_bootstrap_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "voxel_dice",
        "voxel_precision",
        "voxel_recall",
        "cluster_det_precision",
        "cluster_det_recall",
        "cluster_det_f1",
        "cluster_pin_precision",
        "cluster_pin_recall",
        "cluster_pin_f1",
        "n_fp_det_clusters",
        "n_fp_pin_clusters",
    ]
    zero_fill_metrics = set(metrics) - {"n_fp_det_clusters", "n_fp_pin_clusters"}
    rows: List[Dict[str, Any]] = []

    for run_name, run_subjects in subjects.groupby("run"):
        controls = run_subjects[run_subjects["is_control"]]
        for metric in metrics:
            values = run_subjects[metric]
            if metric in zero_fill_metrics:
                values = values.fillna(0.0)
            values = values.dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue

            ci = _bootstrap_ci(values)
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "n_subjects": int(values.size),
                    "unit": "subject",
                    "ci_method": "patient_bootstrap",
                }
            )

        detection = run_subjects["subject_detected"].fillna(0.0).to_numpy(dtype=float)
        pinpointing = run_subjects["subject_pinpointed"].fillna(0.0).to_numpy(dtype=float)
        false_positive = (
            controls["n_pred_clusters"].fillna(0.0).to_numpy(dtype=float) > 0.0
        ).astype(float)
        for metric, values in [
            ("subject_detected", detection),
            ("subject_pinpointed", pinpointing),
            ("false_positive_subject_rate", false_positive),
            ("specificity", 1.0 - false_positive),
        ]:
            if values.size == 0:
                continue
            ci = _bootstrap_ci(values)
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "n_subjects": int(values.size),
                    "unit": "subject",
                    "ci_method": "patient_bootstrap",
                }
            )

    return pd.DataFrame(rows)


def _paired_comparison(joined: pd.DataFrame, baseline: str, comparator: str) -> pd.DataFrame:
    continuous_metrics = [
        "voxel_precision",
        "voxel_recall",
        "voxel_dice",
        "cluster_det_precision",
        "cluster_det_recall",
        "cluster_det_f1",
        "cluster_pin_precision",
        "cluster_pin_recall",
        "cluster_pin_f1",
        "n_pred_clusters",
        "n_fp_det_clusters",
        "n_fp_pin_clusters",
    ]
    binary_metrics = ["subject_detected", "subject_pinpointed"]
    rows: List[Dict[str, Any]] = []

    def add_comparison(metric: str, binary: bool = False) -> None:
        baseline_values = joined[f"{metric}_a"]
        comparator_values = joined[f"{metric}_b"]
        differences = (comparator_values - baseline_values).dropna().to_numpy(dtype=float)
        if differences.size == 0:
            return

        mean_ci = _bootstrap_ci(differences)
        median_ci = _bootstrap_ci(differences, "median", RANDOM_SEED + 1)
        row: Dict[str, Any] = {
            "baseline": baseline,
            "comparator": comparator,
            "direction": "comparator-baseline",
            "metric": metric,
            "n_paired": int(differences.size),
            "mean_diff": float(np.mean(differences)),
            "median_diff": float(np.median(differences)),
            "mean_diff_ci_lower": mean_ci[0],
            "mean_diff_ci_upper": mean_ci[1],
            "median_diff_ci_lower": median_ci[0] if not binary else None,
            "median_diff_ci_upper": median_ci[1] if not binary else None,
            "p_value_signflip_mean": _sign_flip_pvalue(differences),
            "analysis": "paired_binary" if binary else "paired_continuous",
        }
        if binary:
            test = _exact_binomial_sign_test(
                baseline_values.to_numpy(dtype=float),
                comparator_values.to_numpy(dtype=float),
            )
            row["mcnemar_test"] = test["test"]
            row["mcnemar_p"] = test["p_value"]
        rows.append(row)

    for metric in continuous_metrics:
        add_comparison(metric)
    for metric in binary_metrics:
        add_comparison(metric, binary=True)

    controls = joined[(joined["is_control_a"] == True) & (joined["is_control_b"] == True)]
    if not controls.empty:
        baseline_fp = (controls["n_pred_clusters_a"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        comparator_fp = (controls["n_pred_clusters_b"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        differences = comparator_fp - baseline_fp
        mean_ci = _bootstrap_ci(differences)
        test = _exact_binomial_sign_test(baseline_fp, comparator_fp)
        rows.append(
            {
                "baseline": baseline,
                "comparator": comparator,
                "direction": "comparator-baseline",
                "metric": "false_positive_subject_indicator_controls",
                "n_paired": int(differences.size),
                "mean_diff": float(np.mean(differences)),
                "median_diff": float(np.median(differences)),
                "mean_diff_ci_lower": mean_ci[0],
                "mean_diff_ci_upper": mean_ci[1],
                "median_diff_ci_lower": None,
                "median_diff_ci_upper": None,
                "p_value_signflip_mean": _sign_flip_pvalue(differences),
                "mcnemar_test": test["test"],
                "mcnemar_p": test["p_value"],
                "analysis": "paired_binary_controls",
            }
        )

    return pd.DataFrame(rows)


def _mean_summary(values: np.ndarray, seed: int) -> Optional[Tuple[float, float, float]]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    ci = _bootstrap_ci(values, seed=seed)
    return float(np.mean(values)), ci[0], ci[1]


def _cluster_f1_ci(
    precision: np.ndarray,
    recall: np.ndarray,
    seed: int,
) -> Optional[Tuple[float, float]]:
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    valid = np.isfinite(precision) & np.isfinite(recall)
    precision = precision[valid]
    recall = recall[valid]
    if precision.size == 0:
        return None
    if precision.size == 1:
        estimate = _harmonic_f1(float(precision[0]), float(recall[0]))
        return estimate, estimate

    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, precision.size, size=precision.size)
        estimates[index] = _harmonic_f1(
            float(np.mean(precision[sample])),
            float(np.mean(recall[sample])),
        )
    alpha = (1.0 - CI_LEVEL) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _bar_summaries(cases: pd.DataFrame, run_order: Sequence[str]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    summaries: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    simple_metrics = [
        ("voxel_precision", "voxel_precision", RANDOM_SEED + 20),
        ("voxel_recall", "voxel_recall", RANDOM_SEED + 20),
        ("voxel_dice", "voxel_dice", RANDOM_SEED + 20),
    ]
    for run in run_order:
        run_cases = cases[cases["run"] == run]
        for metric, column, seed in simple_metrics:
            summary = _mean_summary(run_cases[column].dropna().to_numpy(dtype=float), seed)
            if summary is not None:
                summaries[(run, metric)] = summary

    for domain_index, prefix in enumerate(["cluster_det", "cluster_pin"]):
        for run_index, run in enumerate(run_order):
            run_cases = cases[cases["run"] == run]
            precision_column = f"{prefix}_precision"
            recall_column = f"{prefix}_recall"
            paired = run_cases[[precision_column, recall_column]].dropna()
            if paired.empty:
                continue

            precision = paired[precision_column].to_numpy(dtype=float)
            recall = paired[recall_column].to_numpy(dtype=float)
            precision_seed = RANDOM_SEED + 21 + domain_index
            recall_seed = precision_seed + 1
            f1_seed = precision_seed + 2
            p_ci = _bootstrap_ci(precision, seed=precision_seed)
            r_ci = _bootstrap_ci(recall, seed=recall_seed)
            f1_ci = _cluster_f1_ci(precision, recall, f1_seed)
            summaries[(run, f"{prefix}_precision")] = (
                float(np.mean(precision)), p_ci[0], p_ci[1]
            )
            summaries[(run, f"{prefix}_recall")] = (
                float(np.mean(recall)), r_ci[0], r_ci[1]
            )
            summaries[(run, f"{prefix}_f1")] = (
                _harmonic_f1(float(np.mean(precision)), float(np.mean(recall))),
                f1_ci[0],
                f1_ci[1],
            )

    return summaries


def _draw_bar_panel(
    ax: plt.Axes,
    summaries: Dict[Tuple[str, str], Tuple[float, float, float]],
    run_order: Sequence[str],
    metrics: Sequence[str],
    ylim: Tuple[float, float],
    ylabel: Optional[str] = None,
) -> None:
    positions = np.arange(len(metrics), dtype=float)
    for run_index, run in enumerate(run_order):
        color = MRI_COLOR if run_index == 0 else MULTIMODAL_COLOR
        offset = (run_index - 0.5) * BAR_WIDTH
        for metric_index, metric in enumerate(metrics):
            summary = summaries.get((run, metric))
            if summary is None:
                continue
            estimate, lower, upper = summary
            x = positions[metric_index] + offset
            ax.bar(
                x,
                estimate,
                width=BAR_WIDTH,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=2,
            )
            ax.errorbar(
                x,
                estimate,
                yerr=[[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]],
                fmt="none",
                ecolor="black",
                capsize=ERRORBAR_CAPSIZE,
                linewidth=1.0,
                zorder=3,
            )

    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(metrics) - 1 + 0.6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def _p_to_stars(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _draw_difference_panel(
    ax: plt.Axes,
    joined: pd.DataFrame,
    comparisons: pd.DataFrame,
    metrics: Sequence[str],
    seed: int,
    secondary_metrics: Sequence[str] = (),
    ylabel: Optional[str] = None,
) -> Optional[plt.Axes]:
    secondary = set(secondary_metrics)
    primary_order = [metric for metric in metrics if metric not in secondary]
    secondary_order = [metric for metric in metrics if metric in secondary]
    positions: Dict[str, float] = {
        metric: float(index) for index, metric in enumerate(primary_order)
    }
    separator = None
    if secondary_order:
        start = float(len(primary_order) - 1) + DIFF_CATEGORY_SPACING if primary_order else 0.0
        separator = float(len(primary_order) - 1) + DIFF_CATEGORY_SPACING / 2.0 if primary_order else None
        for index, metric in enumerate(secondary_order):
            positions[metric] = start + index

    secondary_ax = ax.twinx() if secondary_order else None
    rng = np.random.default_rng(seed)
    comparison_by_metric = comparisons.set_index("metric")
    significance_markers: List[Tuple[plt.Axes, float, float, str]] = []
    for index, metric in enumerate(metrics):
        left = joined[f"{metric}_a"]
        right = joined[f"{metric}_b"]
        differences = (right - left).dropna().to_numpy(dtype=float)
        if differences.size == 0:
            continue

        target_ax = secondary_ax if metric in secondary else ax
        x = positions[metric]
        jitter = rng.uniform(-0.12 * DIFF_CATEGORY_SPACING, 0.12 * DIFF_CATEGORY_SPACING, size=differences.size)
        target_ax.scatter(
            np.full(differences.size, x) + jitter,
            differences,
            color=DIFF_POINT_COLOR,
            edgecolors="none",
            alpha=0.65,
            s=20,
            zorder=2,
        )

        mean_difference = float(np.mean(differences))
        ci = _bootstrap_ci(differences, seed=seed + index)
        target_ax.errorbar(
            [x],
            [mean_difference],
            yerr=[[max(0.0, mean_difference - ci[0])], [max(0.0, ci[1] - mean_difference)]],
            fmt="D",
            color=DIFF_MEAN_COLOR,
            ecolor=DIFF_MEAN_COLOR,
            capsize=ERRORBAR_CAPSIZE,
            markersize=6,
            zorder=4,
        )

        p_value = comparison_by_metric.loc[metric, "p_value_signflip_mean"]
        if pd.notna(p_value) and float(p_value) < 0.05:
            stars = _p_to_stars(float(p_value))
            significance_markers.append((target_ax, x, float(ci[1]), stars))

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    if secondary_ax is not None:
        secondary_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
        secondary_ax.set_ylim(-5.0, 5.0)
        secondary_ax.set_ylabel("FP clusters per subject")
        secondary_ax.grid(False)
    if separator is not None:
        ax.axvline(separator, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.0, alpha=0.5)

    ordered_metrics = list(metrics)
    x_positions = [positions[metric] for metric in ordered_metrics]
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    if secondary_ax is not None:
        secondary_ax.set_xlim(ax.get_xlim())
    ax.set_ylim(-0.5, 0.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for marker_ax, x, ci_upper, stars in significance_markers:
        center_x = marker_ax.transAxes.inverted().transform(
            marker_ax.transData.transform((x, ci_upper))
        )[0]
        ci_upper_y = marker_ax.transAxes.inverted().transform(
            marker_ax.transData.transform((x, ci_upper))
        )[1]
        half_width = (
            SIGNIFICANCE_BRACKET_HALF_WIDTH_PT
            * marker_ax.figure.dpi
            / 72.0
            / marker_ax.bbox.width
        )
        height = (
            SIGNIFICANCE_BRACKET_HEIGHT_PT
            * marker_ax.figure.dpi
            / 72.0
            / marker_ax.bbox.height
        )
        gap = (
            SIGNIFICANCE_BRACKET_GAP_PT
            * marker_ax.figure.dpi
            / 72.0
            / marker_ax.bbox.height
        )
        label_clearance = (
            SIGNIFICANCE_LABEL_CLEARANCE_PT
            * marker_ax.figure.dpi
            / 72.0
            / marker_ax.bbox.height
        )
        bottom = max(0.01, min(ci_upper_y + gap, 1.0 - height - label_clearance))
        marker_ax.plot(
            [center_x - half_width, center_x - half_width,
             center_x + half_width, center_x + half_width],
            [bottom, bottom + height, bottom + height, bottom],
            color="black",
            linewidth=1.0,
            zorder=5,
            transform=marker_ax.transAxes,
        )
        marker_ax.text(
            center_x,
            bottom + height + label_clearance * 0.1,
            stars,
            ha="center",
            va="bottom",
            color="black",
            fontsize=10,
            zorder=6,
            transform=marker_ax.transAxes,
        )
    return secondary_ax


def _plot_comparison(
    subjects: pd.DataFrame,
    joined: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    run_order = list(subjects["run"].drop_duplicates())
    summaries = _bar_summaries(subjects[~subjects["is_control"]], run_order)
    cases = subjects[~subjects["is_control"]]

    rate_values: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for run_index, run in enumerate(run_order):
        run_cases = cases[cases["run"] == run]
        values_by_metric = {
            "detection_rate": run_cases["subject_detected"].dropna().to_numpy(dtype=float),
            "pinpointing_rate": run_cases["subject_pinpointed"].dropna().to_numpy(dtype=float),
        }
        for metric, values in values_by_metric.items():
            summary = _mean_summary(values, RANDOM_SEED + 23)
            if summary is not None:
                rate_values[(run, metric)] = summary

        count_summary = _mean_summary(
            run_cases["n_fp_det_clusters"].dropna().to_numpy(dtype=float),
            RANDOM_SEED + 24,
        )
        if count_summary is not None:
            rate_values[(run, "n_fp_det_clusters")] = count_summary

    plt.rcParams.update({
        "font.family": "Calibri",
        "font.size": 12,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Calibri",
        "mathtext.it": "Calibri:italic",
        "mathtext.bf": "Calibri:bold",
    })
    figure = plt.figure(figsize=(14, 6.5))
    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=[
            1.0,
            OUTER_COLUMN_GAP_RATIO,
            1.0,
            MIDDLE_COLUMN_GAP_RATIO,
            1.0,
            OUTER_COLUMN_GAP_RATIO,
            RIGHT_COLUMN_WIDTH_RATIO,
        ],
        height_ratios=[1.5, 1.0],
        left=0.07,
        right=0.94,
        bottom=0.06,
        top=0.955,
        wspace=0.0,
        hspace=0.20,
    )
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        axes[row, 0] = figure.add_subplot(grid[row, 0])
        for column, grid_column in enumerate((2, 4, 6), start=1):
            axes[row, column] = figure.add_subplot(
                grid[row, grid_column], sharey=axes[row, 0]
            )

    _draw_bar_panel(
        axes[0, 0],
        summaries,
        run_order,
        ["voxel_precision", "voxel_recall", "voxel_dice"],
        (0.0, 1.0),
        ylabel=r"Score/rate [0 — 1]",
    )
    _draw_bar_panel(
        axes[0, 1],
        summaries,
        run_order,
        [
            "cluster_det_precision",
            "cluster_det_recall",
            "cluster_det_f1",
        ],
        (0.0, 1.0),
    )
    _draw_bar_panel(
        axes[0, 2],
        summaries,
        run_order,
        [
            "cluster_pin_precision",
            "cluster_pin_recall",
            "cluster_pin_f1",
        ],
        (0.0, 1.0),
    )
    subject_ax = axes[0, 3]
    rate_metrics = [
        "detection_rate",
        "pinpointing_rate",
    ]
    _draw_bar_panel(subject_ax, rate_values, run_order, rate_metrics, (0.0, 1.0))

    count_x = float(len(rate_metrics) - 1) + DIFF_CATEGORY_SPACING
    subject_ax.axvline(
        float(len(rate_metrics) - 1) + DIFF_CATEGORY_SPACING / 2.0,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )
    count_ax = subject_ax.twinx()
    for run_index, run in enumerate(run_order):
        summary = rate_values.get((run, "n_fp_det_clusters"))
        if summary is None:
            continue
        estimate, lower, upper = summary
        x = count_x + (run_index - 0.5) * BAR_WIDTH
        color = MRI_COLOR if run_index == 0 else MULTIMODAL_COLOR
        count_ax.bar(
            x,
            estimate,
            width=BAR_WIDTH,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=2,
        )
        count_ax.errorbar(
            x,
            estimate,
            yerr=[[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]],
            fmt="none",
            ecolor="black",
            capsize=ERRORBAR_CAPSIZE,
            linewidth=1.0,
            zorder=3,
        )
    subject_ax.set_xlim(-0.6, count_x + 0.6)
    count_ax.set_xlim(subject_ax.get_xlim())
    count_ax.set_ylim(0.0, 5.0)
    count_ax.set_yticks(np.arange(0, 6, 1))
    count_ax.set_ylabel("Number of FP clusters [n]")
    count_ax.patch.set_alpha(0.0)
    count_ax.grid(False)

    _draw_difference_panel(
        axes[1, 0],
        joined,
        comparisons,
        ["voxel_precision", "voxel_recall", "voxel_dice"],
        RANDOM_SEED + 30,
        ylabel=r"Paired difference [-1 — 1]",
    )

    _draw_difference_panel(
        axes[1, 1],
        joined,
        comparisons,
        [
            "cluster_det_precision",
            "cluster_det_recall",
            "cluster_det_f1",
        ],
        RANDOM_SEED + 31,
    )

    _draw_difference_panel(
        axes[1, 2],
        joined,
        comparisons,
        [
            "cluster_pin_precision",
            "cluster_pin_recall",
            "cluster_pin_f1",
        ],
        RANDOM_SEED + 32,
    )

    difference_count_ax = _draw_difference_panel(
        axes[1, 3],
        joined,
        comparisons,
        [
            "subject_detected",
            "subject_pinpointed",
            "n_fp_det_clusters",
        ],
        RANDOM_SEED + 33,
        secondary_metrics=("n_fp_det_clusters",),
    )

    for axis in figure.axes:
        axis.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
    for axis in axes[:, 0]:
        axis.tick_params(axis="y", which="major", left=True, labelleft=True, length=3)
    category_labels = [
        ["Precision", "Recall", "DSC"],
        ["Precision", "Recall", r"$F_1$"],
        ["Precision", "Recall", r"$F_1$"],
        ["Detection", "Pinpointing", "n FP clusters"],
    ]
    for row_axes in axes:
        for column, axis in enumerate(row_axes):
            positions = [0.0, 1.0, count_x] if column == 3 else [0.0, 1.0, 2.0]
            axis.set_xticks(positions, labels=category_labels[column])
            axis.tick_params(axis="x", labelbottom=True, bottom=False, labelsize=12, pad=5)
    count_ax.tick_params(axis="y", which="major", right=True, labelright=True, length=3)
    if difference_count_ax is not None:
        difference_count_ax.set_ylabel("Paired difference [n]")
        difference_count_ax.tick_params(
            axis="y", which="major", right=True, labelright=True, length=3
        )

    legend = [
        Patch(facecolor=MRI_COLOR, edgecolor=NEUTRAL_COLOR, label="MRI"),
        Patch(facecolor=MULTIMODAL_COLOR, edgecolor=NEUTRAL_COLOR, label="Multimodal"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper left", ncol=1, frameon=False)
    # figure.text(
    #     0.5,
    #     0.025,
    #     "Bars: mean with 95% patient-bootstrap CI. Paired differences: multimodal - MRI; "
    #     "points are subjects, error bars are 95% bootstrap CIs, stars use unadjusted sign-flip tests.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=10,
    # )
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def run_comparison(mri_json: Path, multimodal_json: Path, output_dir: Path) -> None:
    mri_name = mri_json.stem
    multimodal_name = multimodal_json.stem
    mri_subjects = _load_subject_rows(mri_json, mri_name)
    multimodal_subjects = _load_subject_rows(multimodal_json, multimodal_name)
    subjects = pd.concat([mri_subjects, multimodal_subjects], ignore_index=True)

    for run_name, run_subjects in subjects.groupby("run"):
        if run_subjects["subject_id"].isna().any():
            raise ValueError(f"{run_name} contains a missing subject ID.")
        if run_subjects["subject_id"].duplicated().any():
            raise ValueError(f"{run_name} contains duplicate subject IDs.")

    mri_ids = set(mri_subjects["subject_id"])
    multimodal_ids = set(multimodal_subjects["subject_id"])
    if mri_ids != multimodal_ids:
        raise ValueError("MRI and multimodal inputs must contain the same subject IDs for paired analysis.")

    joined = mri_subjects.merge(
        multimodal_subjects,
        on="subject_id",
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    if not joined.empty and (joined["is_control_a"] != joined["is_control_b"]).any():
        raise ValueError("MRI and multimodal inputs disagree on control status for a subject.")

    comparisons = _paired_comparison(joined, mri_name, multimodal_name)
    summary = _patient_bootstrap_summary(subjects)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "paired_method_comparison.csv"
    summary_path = output_dir / "summary_patient_bootstrap.csv"
    figure_path = output_dir / "mri_vs_multimodal_comparison.pdf"
    comparisons.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_comparison(subjects, joined, comparisons, figure_path)

    print(f"Saved: {comparison_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MRI-only and multimodal evaluation JSON results."
    )
    parser.add_argument("--mri-json", required=True, type=Path)
    parser.add_argument("--multimodal-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_comparison(args.mri_json, args.multimodal_json, args.output_dir)


if __name__ == "__main__":
    main()
