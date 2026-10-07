"""
Batch wrapper to run preprocess_gt.process_patient() for multiple subjects,
compute global harmonisation threshold from smoothed Photo2Cortex probability maps,
apply the threshold to obtain harmonised masks for all subjects, and produce
plots comparing unharmonised vs harmonised metrics.

Notes:
- For (slightly more) proper documentation, see preprocess_gt.py.

Author: Sjors Verschuren
Date: November 2025
"""

import argparse
import json
import os
import platform
from cycler import cycler
from typing import Any, Callable, Sequence

import nibabel as nib
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import preprocess_gt  # type: ignore

THRESHOLD_BIN_COUNT = 100
THRESHOLD_DOWNSAMPLE_STEP = 2
FIGURE_DPI = 600

PLOT_COLORS = {
    "overlap_unharmonised": "#bacefc",
    "overlap_harmonised": "#0076c0",
    "assd_unharmonised": "#e4b8b4",
    "assd_harmonised": "#ce8080",
    "pr_curve": "#002157",
    "highlight": "#a30234",
    "atlas_pearson": "#a1c5cb",
    "atlas_accuracy": "#f1b682",
    "neutral": "#002f30",
    "grid": "#a1c5cb",
}

plt.rcParams.update({
    "font.family": "Calibri",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.prop_cycle": cycler(color=list(PLOT_COLORS.values())),
    "axes.edgecolor": PLOT_COLORS["neutral"],
    "axes.labelcolor": PLOT_COLORS["neutral"],
    "axes.titlecolor": PLOT_COLORS["neutral"],
    "text.color": PLOT_COLORS["neutral"],
    "xtick.color": PLOT_COLORS["neutral"],
    "ytick.color": PLOT_COLORS["neutral"],
    "grid.color": PLOT_COLORS["grid"],
})


def setup_logger(log_file: str) -> Callable[[str], None]:
    """Return a logger that writes to the terminal and an append-only log file."""

    def logger(message: str) -> None:
        tqdm.write(message)
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"{message}\n")

    return logger


def _write_json(data: Any, output_path: str) -> None:
    """Write a JSON artifact using the pipeline's standard encoding and formatting."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def pr_curve_binned_fast(
    ground_truth: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = THRESHOLD_BIN_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voxel-weighted precision and recall at descending probability bins."""
    truth = np.asarray(ground_truth, dtype=bool).ravel()
    probability = np.asarray(probabilities, dtype=np.float32).ravel()
    if truth.shape != probability.shape:
        raise ValueError("Ground-truth and probability arrays must have matching shapes.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not truth.any():
        raise ValueError("Cannot select an F1 threshold without positive ground-truth voxels.")

    positive_by_bin, bin_edges = np.histogram(
        probability,
        bins=num_bins,
        range=(0.0, 1.0),
        weights=truth.astype(np.uint64),
    )
    voxels_by_bin, _ = np.histogram(probability, bins=bin_edges)

    true_positive = np.cumsum(positive_by_bin[::-1], dtype=np.float64)
    predicted_positive = np.cumsum(voxels_by_bin[::-1], dtype=np.float64)
    total_positive = truth.sum()

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        total_positive,
        out=np.zeros_like(true_positive),
        where=total_positive > 0,
    )
    thresholds = bin_edges[-2::-1]
    return precision, recall, thresholds


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    output_path: str,
) -> None:
    """Save the binned precision-recall curve and its maximum-F1 operating point."""
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.argmax(f1))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(recall, precision, linewidth=2, color=PLOT_COLORS["pr_curve"])
    axis.scatter(
        recall[best_index],
        precision[best_index],
        s=60,
        color=PLOT_COLORS["highlight"],
        zorder=5,
    )
    axis.annotate(
        f"F1={f1[best_index]:.2f}",
        (recall[best_index], precision[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
        color=PLOT_COLORS["highlight"],
    )
    axis.axvline(
        recall[best_index],
        ymax=precision[best_index],
        color=PLOT_COLORS["highlight"],
        linestyle="--",
        alpha=0.5,
    )
    axis.axhline(
        precision[best_index],
        xmax=recall[best_index],
        color=PLOT_COLORS["highlight"],
        linestyle="--",
        alpha=0.5,
    )
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Precision–recall curve")
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def compute_metrics_between_masks(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    affine: np.ndarray,
) -> dict[str, float]:
    """Compute overlap and surface-distance metrics for two binary masks."""
    ground_truth = np.asarray(gt_mask, dtype=np.uint8)
    prediction = np.asarray(pred_mask, dtype=np.uint8)
    return {
        "dice": float(preprocess_gt.dice_coef(ground_truth, prediction)),
        "jaccard": float(preprocess_gt.jaccard_index(ground_truth, prediction)),
        "F1 (B=GT)": float(preprocess_gt.fbeta_score(ground_truth, prediction, beta=1.0)),
        "precision (B=GT)": float(preprocess_gt.precision(ground_truth, prediction)),
        "recall (B=GT)": float(preprocess_gt.recall(ground_truth, prediction)),
        "rVD (B=GT)": float(preprocess_gt.relative_volume_difference(ground_truth, prediction)),
        "hausdorff_mm": float(
            preprocess_gt.hausdorff_distance_mm(ground_truth, prediction, affine)
        ),
        "assd_mm": float(preprocess_gt.assd_mm(ground_truth, prediction, affine)),
    }


def _sample_threshold_voxels(
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the joint nonzero bounding box and sample every second voxel per axis."""
    if probabilities.shape != ground_truth.shape or probabilities.ndim != 3:
        raise ValueError("Probability map and ground-truth mask must share a 3D voxel grid.")

    nonzero = (probabilities > 0) | (ground_truth > 0)
    coordinates = np.argwhere(nonzero)
    if coordinates.size:
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        bounds = tuple(slice(start, stop) for start, stop in zip(lower, upper))
        probabilities = probabilities[bounds]
        ground_truth = ground_truth[bounds]

    stride = (slice(None, None, THRESHOLD_DOWNSAMPLE_STEP),) * 3
    return probabilities[stride].ravel(), ground_truth[stride].ravel()


def plot_comparison_with_atlas_and_pr(
    unharmonised: Sequence[dict[str, Any]],
    harmonised: Sequence[dict[str, Any]],
    pr_points: dict[str, Sequence[float]],
    output_plot: str
) -> None:
    """
    Creates a 3-panel composite figure:
      (1) Full-width intertwined violins for Dice/Precision/Recall + ASSD
      (2) Bottom-left precision–recall curve
      (3) Bottom-right atlas similarity bars (unharmonised only)
    """

    if not unharmonised:
        print("No comparison data provided — skipping plot.")
        return

    # ----------------------------------------------------
    # Build DataFrames
    # ----------------------------------------------------
    df_u = pd.DataFrame(unharmonised)
    df_h = pd.DataFrame(harmonised) if harmonised else pd.DataFrame()

    metrics = ["dice", "precision (B=GT)", "recall (B=GT)"]
    metric_labels = ["Dice / F1", "Precision", "Recall"]
    assd_key = "assd_mm"

    label_map = dict(zip(metrics, metric_labels))

    frames: list[pd.DataFrame] = []

    # ---- Overlap metrics (U vs H) ----
    for df, src in [(df_u, "Unharmonised"), (df_h, "Harmonised")]:
        if df.empty:
            continue

        available = [m for m in metrics if m in df.columns]
        if not available:
            continue
        melted = df[available].melt(var_name="Metric", value_name="Score")
        melted["Metric"] = melted["Metric"].map(label_map)
        melted["Source"] = src
        melted["Type"] = "Overlap"
        frames.append(melted)

    # ---- ASSD ----
    for df, src in [(df_u, "Unharmonised"), (df_h, "Harmonised")]:
        if df.empty or assd_key not in df.columns:
            continue

        df_assd = pd.DataFrame({
            "Metric": ["ASSD"] * len(df),
            "Score": df[assd_key],
            "Source": src,
            "Type": "ASSD"
        })
        frames.append(df_assd)

    if not frames:
        raise ValueError("Comparison records do not contain any plottable metrics.")
    df_all = pd.concat(frames, ignore_index=True)

    # ----------------------------------------------------
    # Atlas Similarity (unharm only, separate panel)
    # ----------------------------------------------------
    atlas_cols = [
        c for c in ["atlas_pearsonr_hemisphere", "atlas_pearsonr_lobe", "atlas_pearsonr_gyrus"]
        if c in df_u.columns
    ]

    # ----------------------------------------------------
    # Figure Layout
    # ----------------------------------------------------
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 2, height_ratios=[2.5, 2.0], width_ratios=[1, 1], hspace=0.2, wspace=0.15)

    # Top full width
    ax1 = fig.add_subplot(gs[0, :])

    # Bottom-left PR
    ax_pr = fig.add_subplot(gs[1, 0])

    # Bottom-right atlas
    ax_atlas = fig.add_subplot(gs[1, 1])

    # ----------------------------------------------------
    # TOP PANEL: intertwined violins + ASSD
    # ----------------------------------------------------
    df_overlap = df_all[df_all["Type"] == "Overlap"]

    # Set consistent colors
    palette_overlap = {
        "Unharmonised": PLOT_COLORS["overlap_unharmonised"],
        "Harmonised": PLOT_COLORS["overlap_harmonised"],
    }

    sns.violinplot(
        x="Metric",
        y="Score",
        hue="Source",
        data=df_overlap,
        ax=ax1,
        cut=0,
        inner="quartile",
        dodge=True,
        width=0.6,
        palette=palette_overlap
    )

    ax1.set_ylabel("Score (0–1)")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, color=PLOT_COLORS["grid"])
    ax1.set_xlabel("")
    ax1.set_title(
        f"Photo2Cortex vs Post-op MRI Voxel-wise Overlap Metrics (n={len(df_u)})"
    )

    # Add legend
    handles_ax1 = [
        plt.Line2D([0], [0], color=palette_overlap["Unharmonised"], lw=6),
        plt.Line2D([0], [0], color=palette_overlap["Harmonised"], lw=6)
    ]
    labels_ax1 = ["Unharmonised", "Harmonised"]
    ax1.legend(handles_ax1, labels_ax1, loc="upper left", frameon=False, fontsize=12)

    # Mean diamond overlay
    x_positions = {cat: i for i, cat in enumerate(df_overlap["Metric"].unique())}
    for metric in x_positions:
        for src in palette_overlap:
            vals = df_overlap[
                (df_overlap["Metric"] == metric) &
                (df_overlap["Source"] == src)
            ]["Score"]
            if len(vals) > 0:
                xpos = x_positions[metric] + (-0.15 if src == "Unharmonised" else 0.15)
                ax1.scatter(xpos, vals.mean(), color=PLOT_COLORS["neutral"], s=60, marker="D", zorder=10)

    # ASSD on twin axis
    if "ASSD" in df_all["Metric"].values:
        df_assd = df_all[df_all["Type"] == "ASSD"]
        ax2 = ax1.twinx()

        palette_assd = {
            "Unharmonised": PLOT_COLORS["assd_unharmonised"],
            "Harmonised": PLOT_COLORS["assd_harmonised"],
        }

        sns.violinplot(
            x="Metric",
            y="Score",
            hue="Source",
            data=df_assd,
            ax=ax2,
            cut=0,
            inner="quartile",
            dodge=True,
            width=0.6,
            palette=palette_assd
        )

        ax2.set_ylabel("ASSD (mm)")
        ax2.set_ylim(0, df_assd["Score"].max() * 1.15)

        # Add legend
        handles_ax2 = [
            plt.Line2D([0], [0], color=palette_assd["Unharmonised"], lw=6),
            plt.Line2D([0], [0], color=palette_assd["Harmonised"], lw=6)
        ]
        labels_ax2 = ["Unharmonised", "Harmonised"]
        ax2.legend(handles_ax2, labels_ax2, loc="upper right", frameon=False, fontsize=12)

        # compute ASSD x-position
        categories = list(df_overlap["Metric"].unique()) + ["ASSD"]
        assd_x = categories.index("ASSD")

        for src in palette_assd:
            vals = df_assd[df_assd["Source"] == src]["Score"]
            if len(vals) > 0:
                xpos = assd_x + (-0.15 if src == "Unharmonised" else 0.15)
                ax2.scatter(xpos, vals.mean(), color=PLOT_COLORS["neutral"], s=70, marker="D", zorder=10)

        ax1.axvline(assd_x - 0.5, color=PLOT_COLORS["neutral"], linestyle="--", alpha=0.7)

        # ax1.legend([], [], frameon=False)
        # ax2.legend([], [], frameon=False)

    # ----------------------------------------------------
    # BOTTOM LEFT PANEL: PR curve
    # ----------------------------------------------------
    if pr_points and "precision" in pr_points and "recall" in pr_points:
        ax_pr.plot(
            pr_points["recall"],
            pr_points["precision"],
            linewidth=2.,
            alpha=0.9,
            color=PLOT_COLORS["pr_curve"]
        )

        # annotate max F1 point
        prec = np.array(pr_points["precision"])
        rec = np.array(pr_points["recall"])
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        max_idx = f1.argmax()
        ax_pr.scatter(rec[max_idx], prec[max_idx], s=60, color=PLOT_COLORS["highlight"], zorder=5)
        ax_pr.text(rec[max_idx]+0.02, prec[max_idx]+0.02, f"F1={f1[max_idx]:.2f}", color=PLOT_COLORS["highlight"])
        ax_pr.vlines(rec[max_idx], 0, prec[max_idx], colors=PLOT_COLORS["highlight"], linestyles='dashed', alpha=0.5)
        ax_pr.hlines(prec[max_idx], 0, rec[max_idx], colors=PLOT_COLORS["highlight"], linestyles='dashed', alpha=0.5)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Harmonisation - Precision–Recall Curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.grid(True, linestyle="--", alpha=0.4, color=PLOT_COLORS["grid"])

    # ----------------------------------------------------
    # BOTTOM RIGHT PANEL: Atlas similarity metrics (unharmonised only)
    # ----------------------------------------------------
    if atlas_cols:
        # Prepare data for intertwined bar chart
        atlas_metrics = ["atlas_pearsonr_hemisphere", "atlas_pearsonr_lobe", "atlas_pearsonr_gyrus"]
        top_region_metrics = ["atlas_top_region_same_hemisphere", "atlas_top_region_same_lobe", "atlas_top_region_same_gyrus"]
        
        categories = ["Hemisphere", "Lobe", "Gyrus"]
        x_pos = np.arange(len(categories))
        width = 0.35
        
        # Extract values for each metric type
        pearson_vals = []
        top_region_vals = []
        
        for atlas_col, top_col in zip(atlas_metrics, top_region_metrics):
            # Pearson correlation
            if atlas_col in df_u.columns:
                pearson_vals.append(df_u[atlas_col].mean())
            else:
                pearson_vals.append(0)
            
            # Top region accuracy
            if top_col in df_u.columns:
                top_region_vals.append(df_u[top_col].mean())
            else:
                top_region_vals.append(0)
        
        # Create twin axis for accuracy
        ax_atlas_twin = ax_atlas.twinx()
        
        # Left axis - Pearson correlation
        # Calculate confidence intervals for Pearson correlations
        pearson_cis = []
        for atlas_col in atlas_metrics:
            if atlas_col in df_u.columns:
                vals = df_u[atlas_col].dropna()
                if len(vals) > 1:
                    ci = 1.96 * vals.std() / np.sqrt(len(vals))  # 95% CI
                    pearson_cis.append(ci)
                else:
                    pearson_cis.append(0)
            else:
                pearson_cis.append(0)
        # Draw bars
        ax_atlas.bar(
            x_pos - width / 2,
            pearson_vals,
            width,
            label="Pearson Correlation",
            color=PLOT_COLORS["atlas_pearson"],
            edgecolor=PLOT_COLORS["neutral"],
            yerr=pearson_cis,
            capsize=5,
            error_kw={"ecolor": PLOT_COLORS["neutral"], "capthick": 1},
        )
        
        # Right axis - Top region accuracy
        ax_atlas_twin.bar(
            x_pos + width / 2,
            top_region_vals,
            width,
            label="Top Region Accuracy",
            color=PLOT_COLORS["atlas_accuracy"],
            edgecolor=PLOT_COLORS["neutral"],
        )
        
        ax_atlas.set_xlabel(' ')
        ax_atlas.set_ylabel('Pearson Correlation (0-1)')
        ax_atlas_twin.set_ylabel('Top Region Accuracy (0-1)')
        ax_atlas.set_title('Atlas-Based Similarity Metrics')
        ax_atlas.set_xticks(x_pos)
        ax_atlas.set_xticklabels(categories)
        ax_atlas.set_ylim(0, 1)
        ax_atlas_twin.set_ylim(0, 1)
        
        # Combine legends
        lines1, labels1 = ax_atlas.get_legend_handles_labels()
        lines2, labels2 = ax_atlas_twin.get_legend_handles_labels()
        ax_atlas.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, fontsize=12)
        
        ax_atlas.grid(axis="y", linestyle="--", alpha=0.4, color=PLOT_COLORS["grid"])

    # ----------------------------------------------------
    # Save Figure
    # ----------------------------------------------------
    fig.savefig(output_plot, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Combined plot saved to: {output_plot}")


def main(
    selection_csv: str,
    postopmri_dir: str,
    fs_dir: str,
    manual_mask_dir: str,
    atlas_lut: str,
    output_base_dir: str,
    reprocess: bool = False,
    plot_comparisons: bool = True,
) -> None:
    os.makedirs(output_base_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_base_dir, "preprocessing_log.txt"))

    df_sel = pd.read_csv(selection_csv)
    patient_ids = df_sel["Participant Id"].unique().tolist()
    run_metadata = {
        "inputs": {
            "selection_csv": os.path.abspath(selection_csv),
            "postoperative_mri_masks": os.path.abspath(postopmri_dir),
            "freesurfer_subjects": os.path.abspath(fs_dir),
            "manual_masks": os.path.abspath(manual_mask_dir),
            "atlas_lookup_table": os.path.abspath(atlas_lut),
        },
        "patient_count": len(patient_ids),
        "reprocess_existing_reports": reprocess,
        "comparison_plots": plot_comparisons,
        "threshold_method": {
            "objective": "maximum pooled voxel-weighted F1",
            "probability_bins": THRESHOLD_BIN_COUNT,
            "sample_stride_per_axis": THRESHOLD_DOWNSAMPLE_STEP,
            "threshold_rule": "probability >= optimal_threshold",
            "crop_region": "joint nonzero bounding box of probability map and ground-truth mask",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "nibabel": nib.__version__,
        },
    }
    _write_json(
        run_metadata,
        os.path.join(output_base_dir, "preprocessing_run_metadata.json"),
    )

    results: list[dict[str, Any]] = []
    comparisons_unharm: list[dict[str, Any]] = []
    comparisons_harmon: list[dict[str, Any]] = []
    top_hemis_unharm: dict[str, dict[str, float]] = {}
    top_lobes_unharm: dict[str, dict[str, float]] = {}
    sampled_ground_truth: list[np.ndarray] = []
    sampled_probabilities: list[np.ndarray] = []
    threshold_sample_patient_ids: list[str] = []
    pr_points: dict[str, np.ndarray] = {}

    # Iterate through patients
    for pid in tqdm(patient_ids, desc="Processing patients"):
        logger(f"[{pid}] ----- Start processing -----")
        # Paths expected by preprocess_gt
        postop_mri_mask = os.path.join(postopmri_dir, f"{pid}_resection_mask.nii.gz")
        aparc_aseg = os.path.join(fs_dir, pid, "mri", "aparc.a2009s+aseg.mgz")
        pic2mri_mask = os.path.join(fs_dir, pid, "pic2mri_output", f"pic2mri_resection_mask_final.nii.gz")
        manual_mask = os.path.join(manual_mask_dir, f"{pid}_manual_resection.nii.gz")
        if not os.path.exists(pic2mri_mask):
            pic2mri_mask = os.path.join(fs_dir, pid, "pic2mri_output", f"pic2mri_resection_mask.nii.gz")
        output_dir = os.path.join(output_base_dir, pid)

        # Check if already processed
        report_json = os.path.join(output_dir, f"{pid}_processing_report.json")
        if os.path.exists(report_json) and not reprocess:
            logger(f"[{pid}] Output directory already exists and not reprocessing. Loading existing report.")
            with open(report_json, encoding="utf-8") as report_file:
                result = json.load(report_file)
        else:
            os.makedirs(output_dir, exist_ok=True)
            try:
                result = preprocess_gt.process_patient(
                    patient_id=pid,
                    mri_mask=postop_mri_mask,
                    pic_mask=pic2mri_mask,
                    manual_mask=manual_mask,
                    atlas=aparc_aseg,
                    atlas_lut=atlas_lut,
                    outdir=output_dir,
                    logger=logger
                )
            except Exception as e:
                logger(f"\033[91mError processing {pid}: {e}\033[0m")
                result = None

        if result is None:
            logger(f"[{pid}] No result for this patient — skipping.")
            continue

        results.append(result)

        # Collect atlas top hemi/lobe info (unharmonised)
        atlas_labeling = result.get('atlas_labeling_gt', {})
        lobe_counts = {}
        hemi_counts = {}
        if atlas_labeling:
            for region in atlas_labeling.get('region_counts', []):
                hemi = region.get('hemisphere', '')
                lobe = region.get('lobe', '')
                if hemi:
                    hemi_counts[hemi] = hemi_counts.get(hemi, 0) + region.get('count', 0)
                if hemi and lobe:
                    lobe_id = f"{hemi}_{lobe}"
                    lobe_counts[lobe_id] = lobe_counts.get(lobe_id, 0) + region.get('count', 0)
            if hemi_counts:
                hemi_counts = {k: v / sum(hemi_counts.values()) for k, v in hemi_counts.items()}
                top_hemis_unharm[pid] = hemi_counts
            if lobe_counts:
                lobe_counts = {k: v / sum(lobe_counts.values()) for k, v in lobe_counts.items()}
                top_lobes_unharm[pid] = lobe_counts
        else:
            logger(f"\033[33m[{pid}] No atlas labeling information found in report.\033[0m")

        # Collect unharmonised comparisons (if present)
        if 'comparison' in result:
            comparisons_unharm.append({'patient_id': pid, **result['comparison']})

        # Collect smoothed prob map and GT for global threshold estimation
        if result.get('chosen_mask_reason', '') == 'mri_mask' and 'pic2mri_smooth_path' in result:
            smooth_path = result.get('pic2mri_smooth_path', None)
            gt_path = result.get('written_nifti', None)
            if smooth_path and os.path.exists(smooth_path) and gt_path and os.path.exists(gt_path):
                try:
                    # Get smoothed Photo2Cortex and GT post-op MRI data
                    sm_img = nib.load(smooth_path)
                    gt_img = nib.load(gt_path)
                    if not np.allclose(sm_img.affine, gt_img.affine):
                        raise ValueError("Probability map and ground-truth mask affines do not match.")
                    sm_data = sm_img.get_fdata(dtype=np.float32)
                    gt_data = gt_img.get_fdata(dtype=np.float32) > 0
                    sampled_prob, sampled_gt = _sample_threshold_voxels(sm_data, gt_data)
                    sampled_probabilities.append(sampled_prob)
                    sampled_ground_truth.append(sampled_gt)
                    threshold_sample_patient_ids.append(pid)
                except Exception as e:
                    logger(f"[{pid}] Could not collect threshold samples: {e}")

    # The threshold maximizes voxel-weighted F1 on the pooled, downsampled samples.
    optimal_threshold = None
    if sampled_ground_truth:
        truth = np.concatenate(sampled_ground_truth)
        probability = np.concatenate(sampled_probabilities)
        precision, recall, thresholds = pr_curve_binned_fast(
            truth,
            probability,
            num_bins=THRESHOLD_BIN_COUNT,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        best_index = int(np.argmax(f1))
        optimal_threshold = float(thresholds[best_index])
        pr_points = {"precision": precision, "recall": recall}

        # Save threshold
        os.makedirs(output_base_dir, exist_ok=True)
        threshold_metadata = {
            "optimal_threshold": optimal_threshold,
            "selection_metric": "pooled_voxel_weighted_f1",
            "probability_bins": THRESHOLD_BIN_COUNT,
            "sample_stride_per_axis": THRESHOLD_DOWNSAMPLE_STEP,
            "sampled_patients": threshold_sample_patient_ids,
            "sampled_voxels": int(truth.size),
            "positive_sampled_voxels": int(truth.sum()),
            "threshold_rule": "probability >= optimal_threshold",
            "tie_breaking": "highest threshold",
        }
        _write_json(
            threshold_metadata,
            os.path.join(output_base_dir, "optimal_threshold.json"),
        )
        logger(f"Optimal harmonisation threshold = {optimal_threshold:.6f}")

        pr_plot_path = os.path.join(output_base_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(precision, recall, pr_plot_path)
        logger(f"Saved precision-recall plot to: {pr_plot_path}")
    else:
        logger("No threshold samples collected; harmonised masks were not produced.")

    # Apply threshold to all smoothed maps to produce harmonised masks
    for pid in tqdm(patient_ids, desc="Applying threshold to produce harmonised masks"):

        # Read report, set paths
        output_dir = os.path.join(output_base_dir, pid)
        report_json = os.path.join(output_dir, f"{pid}_processing_report.json")
        if not os.path.exists(report_json):
            continue
        with open(report_json, encoding="utf-8") as report_file:
            report = json.load(report_file)
        
        smooth_path = report.get('pic2mri_smooth_path', None)
        gt_path = report.get('written_nifti', None)

        # Apply threshold if possible
        if smooth_path and os.path.exists(smooth_path) and optimal_threshold is not None:
            # Get smoothed data
            sm_img = nib.load(smooth_path)
            sm_data = sm_img.get_fdata()
            # Threshold
            harmonised_bin = (sm_data >= optimal_threshold).astype(np.uint8)
            # Save harmonised mask
            harmon_out = os.path.join(output_dir, f"{pid}_pic2mri_harmonised.nii.gz")
            nib.save(nib.Nifti1Image(harmonised_bin.astype(np.uint8), sm_img.affine), harmon_out)

            # If Photo2Cortex is the ground truth mask, also save as ground truth mask
            if report.get('chosen_mask_reason', '') == 'pic2mri':
                gt_harmon_out = os.path.join(output_dir, f"{pid}_gt_mask_harmonised.nii.gz")
                nib.save(nib.Nifti1Image(harmonised_bin.astype(np.uint8), sm_img.affine), gt_harmon_out)
                report['written_nifti_harmonised'] = gt_harmon_out

            # If the post-op MRI mask is available, compute harmonised metrics
            if report.get('chosen_mask_reason', '') == 'mri_mask':
                # Get MRI (GT) data
                gt_img = nib.load(gt_path)
                gt_data = gt_img.get_fdata().astype(np.uint8)
                # Compute metrics
                metrics = compute_metrics_between_masks(gt_data, harmonised_bin, gt_img.affine)
                comparisons_harmon.append({"patient_id": pid, **metrics})
                # Save harmonised comparison per patient
                harmon_comp_path = os.path.join(output_dir, f"{pid}_comparison_harmonised.json")
                _write_json(metrics, harmon_comp_path)
                # Update processing report with harmonised info
                report['harmonised_mask_path'] = harmon_out
                report['harmonised_threshold'] = optimal_threshold
                report['harmonised_comparison_path'] = harmon_comp_path
                _write_json(report, report_json)

    # Save summary JSONs
    _write_json(results, os.path.join(output_base_dir, "processing_summary.json"))
    if comparisons_unharm:
        _write_json(
            comparisons_unharm,
            os.path.join(output_base_dir, "comparison_summary_unharmonised.json"),
        )
    if comparisons_harmon:
        _write_json(
            comparisons_harmon,
            os.path.join(output_base_dir, "comparison_summary_harmonised.json"),
        )
    if top_hemis_unharm:
        _write_json(top_hemis_unharm, os.path.join(output_base_dir, "gt_hemi.json"))
    if top_lobes_unharm:
        _write_json(top_lobes_unharm, os.path.join(output_base_dir, "gt_lobe.json"))

    # Plot comparisons (two-panel)
    if plot_comparisons:
        plot_file = os.path.join(output_base_dir, "comparison_plot_extended.png")
        plot_comparison_with_atlas_and_pr(
            unharmonised=comparisons_unharm,
            harmonised=comparisons_harmon,
            pr_points=pr_points,
            output_plot=plot_file
        )


def parse_args() -> argparse.Namespace:
    """Parse paths and behavior flags for one ground-truth preprocessing run."""
    parser = argparse.ArgumentParser(
        description="Preprocess selected ground-truth masks and estimate a global threshold."
    )
    data_dir = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data"
    parser.add_argument(
        "--selection-csv",
        default=os.path.join(data_dir, "selection", "selected_summary.csv"),
        help="CSV containing the 'Participant Id' column.",
    )
    parser.add_argument(
        "--postopmri-dir",
        default=os.path.join(data_dir, "masks_postop_mri"),
        help="Directory containing post-operative MRI masks.",
    )
    parser.add_argument(
        "--fs-dir",
        default=os.path.join(data_dir, "dataset_fs"),
        help="FreeSurfer subject directory.",
    )
    parser.add_argument(
        "--manual-mask-dir",
        default=os.path.join(data_dir, "manual_segs"),
        help="Directory containing manual resection masks.",
    )
    parser.add_argument(
        "--atlas-lut",
        default=r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\ext\FreeSurfer\FreeSurferColorLUT.txt",
        help="FreeSurfer color lookup table.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(data_dir, "preprocessing", "gt"),
        help="Directory for reports, masks, and figures.",
    )
    parser.add_argument("--reprocess", action="store_true", help="Recompute reports even when they already exist.")
    parser.add_argument(
        "--no-comparison-plots",
        action="store_true",
        help="Skip the comparison figure while still saving metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        selection_csv=args.selection_csv,
        postopmri_dir=args.postopmri_dir,
        fs_dir=args.fs_dir,
        manual_mask_dir=args.manual_mask_dir,
        atlas_lut=args.atlas_lut,
        output_base_dir=args.output_dir,
        reprocess=args.reprocess,
        plot_comparisons=not args.no_comparison_plots,
    )