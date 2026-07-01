"""
Batch wrapper to run preprocess_gt.process_patient() for multiple subjects,
compute global harmonisation threshold from smoothed pic2mri probability maps,
apply the threshold to obtain harmonised masks for all subjects, and produce
plots comparing unharmonised vs harmonised metrics.

Notes:
- For (slightly more) proper documentation, see preprocess_gt.py.

Author: Sjors Verschuren
Date: November 2025
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from tqdm import tqdm
import nibabel as nib
import preprocess_gt  # type: ignore

def setup_logger(log_file: str):
    """
    Create a logger function that writes to both tqdm.write and a log file.
    """
    def logger(message: str):
        tqdm.write(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    return logger

def pr_curve_binned_fast(gt_bool: np.ndarray, probs: np.ndarray, num_bins: int = 200):
    """
    Vectorized, memory- and speed-efficient binned PR curve.
    """
    gt = gt_bool.astype(np.uint8).ravel()
    p = probs.astype(np.float16).ravel()

    # bin indices 0..num_bins-1
    bins = np.linspace(0, 1, num_bins + 1)
    inds = np.digitize(p, bins) - 1

    # TP per bin
    TP_per_bin = np.bincount(inds, weights=gt, minlength=num_bins)
    # Total per bin
    total_per_bin = np.bincount(inds, minlength=num_bins)

    # Sweep from high → low probability bins
    TP_cumsum = np.cumsum(TP_per_bin[::-1])
    total_cumsum = np.cumsum(total_per_bin[::-1])
    FP_cumsum = total_cumsum - TP_cumsum
    FN_cumsum = gt.sum() - TP_cumsum

    precision = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-8)
    recall = TP_cumsum / (TP_cumsum + FN_cumsum + 1e-8)
    thresholds = bins[::-1][1:]

    return precision, recall, thresholds

def plot_precision_recall_curve(prec: np.ndarray, rec: np.ndarray, output_path: str):
    """ Plot PR curve and save to file. """
    # Figure
    plt.figure(figsize=(6, 6))
    # Plot
    plt.plot(rec, prec, lw=2, color="darkblue")
    # annotate max F1 point
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    max_idx = f1.argmax()
    plt.scatter(rec[max_idx], prec[max_idx], s=60, color="red", zorder=5)
    plt.text(rec[max_idx]+0.02, prec[max_idx]+0.02, f"F1={f1[max_idx]:.2f}", color="red")
    plt.vlines(rec[max_idx], 0, prec[max_idx], colors='red', linestyles='dashed', alpha=0.5)
    plt.hlines(prec[max_idx], 0, rec[max_idx], colors='red', linestyles='dashed', alpha=0.5)
    # Visuals
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Save
    plt.savefig(output_path, dpi=300)
    plt.close()

def compute_metrics_between_masks(gt_mask: np.ndarray, pred_mask: np.ndarray, affine: np.ndarray):
    # Expect binary uint8 or bool arrays
    gt = gt_mask.astype(np.uint8)
    pred = pred_mask.astype(np.uint8)
    return {
        'dice': float(preprocess_gt.dice_coef(gt, pred)),
        'jaccard': float(preprocess_gt.jaccard_index(gt, pred)),
        'F1 (B=GT)': float(preprocess_gt.fbeta_score(gt, pred, beta=1.0)),
        'precision (B=GT)': float(preprocess_gt.precision(gt, pred)),
        'recall (B=GT)': float(preprocess_gt.recall(gt, pred)),
        'rVD (B=GT)': float(preprocess_gt.relative_volume_difference(gt, pred)),
        'hausdorff_mm': float(preprocess_gt.hausdorff_distance_mm(gt, pred, affine)),
        'assd_mm': float(preprocess_gt.assd_mm(gt, pred, affine))
    }

def plot_comparison_with_atlas_and_pr(
    unharmonised: list,
    harmonised: list,
    pr_points: dict,
    output_plot: str
):
    """
    Creates a 3-panel composite figure:
      (1) Full-width intertwined violins for Dice/Jaccard/Precision/Recall + ASSD
      (2) Bottom-left precision–recall curve
      (3) Bottom-right atlas similarity violins (unharmonised only)
    """

    if not unharmonised:
        print("No comparison data provided — skipping plot.")
        return

    # ----------------------------------------------------
    # Build DataFrames
    # ----------------------------------------------------
    df_u = pd.DataFrame(unharmonised)
    df_h = pd.DataFrame(harmonised) if harmonised else pd.DataFrame()

    metrics = ["dice", "jaccard", "precision (B=GT)", "recall (B=GT)"]
    metric_labels = ["Dice / F1", "Jaccard (IoU)", "Precision", "Recall"]
    assd_key = "assd_mm"

    label_map = dict(zip(metrics, metric_labels))

    frames = []

    # ---- Overlap metrics (U vs H) ----
    for df, src in [(df_u, "Unharmonised"), (df_h, "Harmonised")]:
        if df.empty:
            continue

        available = [m for m in metrics if m in df.columns]
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

    df_all = pd.concat(frames, ignore_index=True)

    # ----------------------------------------------------
    # Atlas Similarity (unharm only, separate panel)
    # ----------------------------------------------------
    atlas_cols = [
        c for c in ["atlas_pearsonr_hemisphere", "atlas_pearsonr_lobe", "atlas_pearsonr_gyrus"]
        if c in df_u.columns
    ]

    df_atlas = pd.DataFrame()
    if atlas_cols:
        df_atlas = df_u[atlas_cols].melt(var_name="Metric", value_name="Score")
        df_atlas["Metric"] = df_atlas["Metric"].map({
            "atlas_pearsonr_hemisphere": "Hemisphere Similarity",
            "atlas_pearsonr_lobe": "Lobar Similarity",
            "atlas_pearsonr_gyrus": "Gyral Similarity"
        })

    # ----------------------------------------------------
    # Figure Layout
    # ----------------------------------------------------
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 2, height_ratios=[2.5, 2.0], width_ratios=[1, 1], hspace=0.1, wspace=0.1)

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
    palette_overlap = {"Unharmonised": "#B2D6EC", "Harmonised": "skyblue"}

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
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_xlabel("")
    ax1.set_title(
        f"Pic2MRI vs Post-op MRI Voxel-wise Overlap Metrics (n={len(df_u)})"
    )

    # Add legend
    handles_ax1 = [
        plt.Line2D([0], [0], color=palette_overlap["Unharmonised"], lw=6),
        plt.Line2D([0], [0], color=palette_overlap["Harmonised"], lw=6)
    ]
    labels_ax1 = ["Unharmonised", "Harmonised"]
    ax1.legend(handles_ax1, labels_ax1, loc="upper left", frameon=False, fontsize=11)

    # Mean diamond overlay
    x_positions = {cat: i for i, cat in enumerate(df_overlap["Metric"].unique())}
    for metric in x_positions:
        for src, color in palette_overlap.items():
            vals = df_overlap[
                (df_overlap["Metric"] == metric) &
                (df_overlap["Source"] == src)
            ]["Score"]
            if len(vals) > 0:
                xpos = x_positions[metric] + (-0.15 if src == "Unharmonised" else 0.15)
                ax1.scatter(xpos, vals.mean(), color="gray", s=60, marker="D", zorder=10)

    # ASSD on twin axis
    if "ASSD" in df_all["Metric"].values:
        df_assd = df_all[df_all["Type"] == "ASSD"]
        ax2 = ax1.twinx()

        palette_assd = {"Unharmonised": "#E4BABA", "Harmonised": "lightcoral"}

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
        ax2.legend(handles_ax2, labels_ax2, loc="upper right", frameon=False, fontsize=11)

        # compute ASSD x-position
        categories = list(df_overlap["Metric"].unique()) + ["ASSD"]
        assd_x = categories.index("ASSD")

        for src, color in palette_assd.items():
            vals = df_assd[df_assd["Source"] == src]["Score"]
            if len(vals) > 0:
                xpos = assd_x + (-0.15 if src == "Unharmonised" else 0.15)
                ax2.scatter(xpos, vals.mean(), color="gray", s=70, marker="D", zorder=10)

        ax1.axvline(assd_x - 0.5, color="gray", linestyle="--", alpha=0.7)

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
            color="darkblue"
        )

        # annotate max F1 point
        prec = np.array(pr_points["precision"])
        rec = np.array(pr_points["recall"])
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        max_idx = f1.argmax()
        ax_pr.scatter(rec[max_idx], prec[max_idx], s=60, color="red", zorder=5)
        ax_pr.text(rec[max_idx]+0.02, prec[max_idx]+0.02, f"F1={f1[max_idx]:.2f}", color="red")
        ax_pr.vlines(rec[max_idx], 0, prec[max_idx], colors='red', linestyles='dashed', alpha=0.5)
        ax_pr.hlines(prec[max_idx], 0, rec[max_idx], colors='red', linestyles='dashed', alpha=0.5)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Harmonisation - Precision–Recall Curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.grid(True, linestyle="--", alpha=0.4)

    # ----------------------------------------------------
    # BOTTOM RIGHT PANEL: Atlas similarity violins (unharm only)
    # ----------------------------------------------------
    if not df_atlas.empty:
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
        bars1 = ax_atlas.bar(x_pos - width/2, pearson_vals, width,
                    label='Pearson Correlation', color='#B2D6EC', edgecolor='gray',
                    yerr=pearson_cis, capsize=5, 
                    error_kw={'ecolor': 'gray', 'capthick': 1})
        
        # Right axis - Top region accuracy
        bars2 = ax_atlas_twin.bar(x_pos + width/2, top_region_vals, width, 
                                 label='Top Region Accuracy', color='#E4BABA', edgecolor='gray')
        
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
        ax_atlas.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, fontsize=11)
        
        ax_atlas.grid(axis="y", linestyle="--", alpha=0.4)

    # ----------------------------------------------------
    # Save Figure
    # ----------------------------------------------------
    # plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Combined plot saved to: {output_plot}")

def main(selection_csv: str, postopmri_dir: str, fs_dir:str, manual_mask_dir: str, atlas_lut:str, output_base_dir: str, reprocess=False, plot_comparisons=True):

    logger = setup_logger(os.path.join(output_base_dir, "preprocessing_log.txt"))

    df_sel = pd.read_csv(selection_csv)
    patient_ids = df_sel["Participant Id"].unique().tolist()

    results = []
    comparisons_unharm = []
    comparisons_harmon = []
    top_hemis_unharm = {}
    top_lobes_unharm = {}

    # Accumulators for global threshold computation
    global_gt_flat = []
    global_prob_flat = []
    atlas_affine_example = None

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
            try:
                result = json.load(open(report_json, 'r'))
            except Exception as e:
                logger(f"[{pid}] Failed to load existing report: {e}. Reprocessing.")
                result = None
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
                    # Get smoothed pic2mri and GT post-op MRI data
                    sm_img = nib.load(smooth_path)
                    gt_img = nib.load(gt_path)
                    sm_data = sm_img.get_fdata().astype(np.float16)
                    gt_data = gt_img.get_fdata().astype(np.uint8)

                    # Crop to relevant region (remove zero padding)
                    # Find bounding box where either smoothed or GT has non-zero values
                    # We do this for memory efficiency, otherwise we consider all voxels in all masks...
                    combined_nonzero = (sm_data > 0) | (gt_data > 0)
                    nonzero_coords = np.argwhere(combined_nonzero)
                    if len(nonzero_coords) > 0:
                        # Get bounding box
                        min_coords = nonzero_coords.min(axis=0)
                        max_coords = nonzero_coords.max(axis=0)
                        
                        # Crop both arrays to bounding box
                        sm_data = sm_data[
                            min_coords[0]:max_coords[0]+1,
                            min_coords[1]:max_coords[1]+1,
                            min_coords[2]:max_coords[2]+1
                        ]
                        gt_data = gt_data[
                            min_coords[0]:max_coords[0]+1,
                            min_coords[1]:max_coords[1]+1,
                            min_coords[2]:max_coords[2]+1
                        ]
                    
                    # Flatten arrays and downsample by 2, again for memory efficiency
                    sm_data = sm_data[::2, ::2, ::2].flatten()
                    gt_data = gt_data[::2, ::2, ::2].flatten()

                    # Save affine example for later metric computations
                    if atlas_affine_example is None:
                        atlas_affine_example = gt_img.affine

                    global_prob_flat.append(sm_data)
                    global_gt_flat.append(gt_data)
                except Exception as e:
                    logger(f"[{pid}] Error loading smooth/gt for threshold collection: {e}")

    # If we have collected probabilities, compute global threshold (max F1 on PR curve)
    OPT_THR = None
    if global_gt_flat:
        print("Computing optimal harmonisation threshold from collected smoothed probabilities...")
        gt_all = np.concatenate(global_gt_flat).astype(np.uint8)
        prob_all = np.concatenate(global_prob_flat).astype(np.float16)

        # Compute PR curve
        print("Calculating precision-recall curve...", flush=True, end=' ')
        prec, rec, thr = pr_curve_binned_fast(gt_all, prob_all, num_bins=100)
        print("Done.")

        # F1 for each threshold (precision_recall_curve returns arrays such that len(thr) = len(prec)-1)
        f1_vals = 2 * (prec * rec) / (prec + rec + 1e-8)
        # thr corresponds to thresholds between prec/rec entries; choose best index from thr array
        # We pick index of maximum F1 excluding the last prec/rec pair where thr is not defined
        # Align f1_vals[1:] with thr
        if len(thr) > 0:
            best_idx = int(np.argmax(f1_vals[:-1]))  # ignore the final prec/rec that has no thr
            OPT_THR = float(thr[best_idx])
        else:
            # fallback: choose 0.5
            OPT_THR = 0.5

        # Save threshold
        os.makedirs(output_base_dir, exist_ok=True)
        with open(os.path.join(output_base_dir, "optimal_threshold.json"), "w") as f:
            json.dump({"optimal_threshold": OPT_THR}, f, indent=2)
        print(f"Optimal harmonisation threshold = {OPT_THR:.6f}")

        # Plot PR curve
        print("Plotting precision-recall curve...", flush=True, end=' ')
        pr_plot_path = os.path.join(output_base_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(prec, rec, pr_plot_path)
        print("Done. Saved to:", pr_plot_path)
    else:
        print("No smoothed probability maps collected — skipping threshold estimation.")
        OPT_THR = None

    # Apply threshold to all smoothed maps to produce harmonised masks
    for pid in tqdm(patient_ids, desc="Applying threshold to produce harmonised masks"):

        # Read report, set paths
        output_dir = os.path.join(output_base_dir, pid)
        report_json = os.path.join(output_dir, f"{pid}_processing_report.json")
        if not os.path.exists(report_json):
            continue
        try:
            report = json.load(open(report_json, 'r'))
        except Exception:
            continue
        
        smooth_path = report.get('pic2mri_smooth_path', None)
        gt_path = report.get('written_nifti', None)

        # Apply threshold if possible
        if smooth_path and os.path.exists(smooth_path) and OPT_THR is not None:
            # Get smoothed data
            sm_img = nib.load(smooth_path)
            sm_data = sm_img.get_fdata()
            # Threshold
            harmonised_bin = (sm_data > OPT_THR).astype(np.uint8)
            # Save harmonised mask
            harmon_out = os.path.join(output_dir, f"{pid}_pic2mri_harmonised.nii.gz")
            nib.save(nib.Nifti1Image(harmonised_bin.astype(np.uint8), sm_img.affine), harmon_out)

            # If pic2mri is the ground truth mask, also save as ground truth mask
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
                mets = compute_metrics_between_masks(gt_data, harmonised_bin, gt_img.affine)
                comparisons_harmon.append({'patient_id': pid, **mets})
                # Save harmonised comparison per patient
                harmon_comp_path = os.path.join(output_dir, f"{pid}_comparison_harmonised.json")
                with open(harmon_comp_path, 'w') as f:
                    json.dump(mets, f, indent=2)
                # Update processing report with harmonised info
                report['harmonised_mask_path'] = harmon_out
                report['harmonised_threshold'] = OPT_THR
                report['harmonised_comparison_path'] = harmon_comp_path
                with open(report_json, 'w') as f:
                    json.dump(report, f, indent=2)

    # Save summary JSONs
    os.makedirs(output_base_dir, exist_ok=True)
    with open(os.path.join(output_base_dir, "processing_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    if comparisons_unharm:
        with open(os.path.join(output_base_dir, "comparison_summary_unharmonised.json"), 'w') as f:
            json.dump(comparisons_unharm, f, indent=2)
    if comparisons_harmon:
        with open(os.path.join(output_base_dir, "comparison_summary_harmonised.json"), 'w') as f:
            json.dump(comparisons_harmon, f, indent=2)
    if top_hemis_unharm:
        with open(os.path.join(output_base_dir, "gt_hemi.json"), 'w') as f:
            json.dump(top_hemis_unharm, f, indent=2)
    if top_lobes_unharm:
        with open(os.path.join(output_base_dir, "gt_lobe.json"), 'w') as f:
            json.dump(top_lobes_unharm, f, indent=2)

    # Plot comparisons (two-panel)
    if plot_comparisons:
        plot_file = os.path.join(output_base_dir, "comparison_plot_extended.png")
        plot_comparison_with_atlas_and_pr(
            unharmonised=comparisons_unharm,
            harmonised=comparisons_harmon,
            pr_points={"precision": prec.tolist(), "recall": rec.tolist()} if global_gt_flat else {},
            output_plot=plot_file
        )

if __name__ == "__main__":

    # Define paths (example; update to your local paths)
    selection_csv = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\selection\\selected_summary.csv"
    postopmri_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\masks_postop_mri"
    fs_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_fs"
    manual_mask_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\manual_segs"
    atlas_lut = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\ext\\FreeSurfer\\FreeSurferColorLUT.txt"
    output_base_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\gt"

    main(selection_csv, postopmri_dir, fs_dir, manual_mask_dir, atlas_lut, output_base_dir)