"""
Plot cross-validation results from eeg_spike_mil_regression_training.

Loads all validation.json files from the fold subdirectories inside a run
directory and combines them into one dataset for plotting.

Usage
-----
    python plot_mil_regression_validation.py <run_dir> [--output <path>]

    <run_dir>  Directory that contains the fold subdirectories
               (eeg_mil_regression_fold*).
    --output   Optional path for the saved figure (PNG/SVG/…).
               If omitted, the figure is shown interactively.
"""

import argparse
import json
import os
from glob import glob

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Colour scheme (mirrors preprocess_gt_loop.py)
# ---------------------------------------------------------------------------

DIST_COLORS = ["#8FC0DD", "#C1DDEE", "#C1DDEE", "#C1DDEE"]   # blue family (Total darkest)
SIGMA_COLORS = ["#D88C8C", "#E3A7A7", "#E3A7A7", "#E3A7A7"]  # red family (Total darkest)

DIST_COLS = ["euclidean_mm", "euclidean_x_mm", "euclidean_y_mm", "euclidean_z_mm"]
DIST_TITLES = ["Total", "X", "Y", "Z"]

SIGMA_COLS = ["sigma_total_mm", "sigma_x_mm", "sigma_y_mm", "sigma_z_mm"]
SIGMA_TITLES = ["Total", "X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fold_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_all_folds(run_dir: str) -> pd.DataFrame:
    """Glob all fold subdirectories, load validation.json, return combined DF."""
    pattern = os.path.join(run_dir, "*_fold*", "validation.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No validation.json files found under '{run_dir}'.\n"
            f"  Searched: {pattern}"
        )

    rows = []
    for fpath in files:
        fold_data = load_fold_json(fpath)
        # New format: {"summary": {...}, "cases": {...}}.
        # Legacy format: {"patient_id": {...}, ...}.
        case_entries = fold_data.get("cases", fold_data)

        if not isinstance(case_entries, dict):
            raise TypeError(
                f"Invalid validation format in {fpath}: expected a dict of cases."
            )

        for pid, entry in case_entries.items():
            dist_mm = entry.get("coord_euclidean_mm", entry.get("euclidean_mm"))
            dist_norm = entry.get("coord_euclidean_norm", entry.get("euclidean_norm"))
            if dist_mm is None or dist_norm is None:
                raise KeyError(
                    f"Missing distance field in {fpath} for patient '{pid}'. "
                    "Expected coord_euclidean_* or legacy euclidean_* fields."
                )

            # Per-sample mm scale (how many mm equals one normalised unit).
            scale_mm = dist_mm / dist_norm if dist_norm > 0 else 0.0

            pred = np.array(entry["pred_mu"], dtype=float)
            gt = np.array(entry["gt_mu"], dtype=float)
            sigma = np.array(entry.get("sigma", entry.get("pred_sigma", [np.nan, np.nan, np.nan])), dtype=float)

            diff = np.abs(pred - gt)

            # Fallback scale for legacy/new mixed files where normalized distance is absent.
            if scale_mm == 0.0:
                diff_norm = float(np.linalg.norm(pred - gt))
                scale_mm = dist_mm / diff_norm if diff_norm > 0 else 0.0

            rows.append(
                {
                    "patient_id": pid,
                    "fold": os.path.basename(os.path.dirname(fpath)),
                    # --- distance from mu to gt ---
                    "euclidean_mm": dist_mm,
                    "euclidean_x_mm": diff[0] * scale_mm,
                    "euclidean_y_mm": diff[1] * scale_mm,
                    "euclidean_z_mm": diff[2] * scale_mm,
                    # --- sigma ---
                    "sigma_total_mm": float(np.linalg.norm(sigma)) * scale_mm,
                    "sigma_x_mm": sigma[0] * scale_mm,
                    "sigma_y_mm": sigma[1] * scale_mm,
                    "sigma_z_mm": sigma[2] * scale_mm,
                    # --- gt position magnitude (for scatter) ---
                    "gt_norm_mm": float(np.linalg.norm(gt)) * scale_mm,
                }
            )

    print(f"Loaded {len(rows)} samples from {len(files)} fold(s).")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violin(ax, df, col, color, title, ylabel=None):
    """Single violin with an inner box and a median-diamond overlay."""
    sns.violinplot(
        data=df,
        y=col,
        color=color,
        inner="box",
        ax=ax,
        cut=0,
        linewidth=0.8,
    )
    # median diamond (matches preprocess_gt_loop style)
    ax.scatter(0, df[col].median(), color="gray", s=60, marker="D", zorder=10)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if ylabel else "mm")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _violin_panel(ax, df, cols, titles, colors, panel_title, ylabel):
    """Render multiple violins side-by-side inside a single axis."""
    panel_df = pd.DataFrame(
        {
            "Component": np.repeat(titles, len(df)),
            "Value": np.concatenate([df[c].values for c in cols]),
        }
    )

    clipped_frames = []
    outlier_frames = []
    for title in titles:
        comp_vals = panel_df.loc[panel_df["Component"] == title, "Value"]
        p99 = float(np.percentile(comp_vals, 99))

        comp_clipped = comp_vals.copy()
        comp_clipped[comp_clipped > p99] = p99
        clipped_frames.append(pd.DataFrame({"Component": title, "Value": comp_clipped.values}))

        comp_outliers = comp_vals[comp_vals > p99]
        if not comp_outliers.empty:
            outlier_frames.append(
                pd.DataFrame({"Component": title, "Value": comp_outliers.values})
            )

    panel_clipped_df = pd.concat(clipped_frames, ignore_index=True)

    palette = {title: color for title, color in zip(titles, colors)}

    sns.violinplot(
        data=panel_clipped_df,
        x="Component",
        y="Value",
        hue="Component",
        palette=palette,
        inner="box",
        cut=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )

    # Show values above each component's 99th percentile as tiny jittered dots.
    if outlier_frames:
        outlier_df = pd.concat(outlier_frames, ignore_index=True)
        comp_to_idx = {title: i for i, title in enumerate(titles)}
        rng = np.random.default_rng(0)
        x_pos = np.array([comp_to_idx[c] for c in outlier_df["Component"]], dtype=float)
        jitter = rng.normal(loc=0.0, scale=0.03, size=len(outlier_df))
        ax.scatter(
            x_pos + jitter,
            outlier_df["Value"].values,
            s=7,
            color="#2f2f2f",
            alpha=0.75,
            linewidths=0,
            zorder=11,
        )

    medians = panel_df.groupby("Component", sort=False)["Value"].median().reindex(titles)
    for i, m in enumerate(medians.values):
        ax.scatter(i, m, color="gray", s=52, marker="D", zorder=10)

    ax.set_title(panel_title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, output_path=None):
    fig = plt.figure(figsize=(13.6, 7.4))
    fig.suptitle("10-fold cross-validation results – EEG spike MIL Regression", fontsize=14, y=0.995, fontweight="bold")

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.08],
        hspace=0.22,
        wspace=0.16,
    )

    # -----------------------------------------------------------------------
    # Left-top – μ→GT distance violins (single panel)
    # -----------------------------------------------------------------------
    ax_dist = fig.add_subplot(gs[0, 0])
    _violin_panel(
        ax=ax_dist,
        df=df,
        cols=DIST_COLS,
        titles=DIST_TITLES,
        colors=DIST_COLORS,
        panel_title="μ→GT Distance Components",
        ylabel="μ→GT distance (mm)",
    )

    # -----------------------------------------------------------------------
    # Left-bottom – sigma violins (single panel)
    # -----------------------------------------------------------------------
    ax_sigma = fig.add_subplot(gs[1, 0])
    _violin_panel(
        ax=ax_sigma,
        df=df,
        cols=SIGMA_COLS,
        titles=SIGMA_TITLES,
        colors=SIGMA_COLORS,
        panel_title="Uncertainty (σ) Components",
        ylabel="σ (mm)",
    )

    # Keep both violin panels on the same y-scale for direct comparison.
    y_max = max(ax_dist.get_ylim()[1], ax_sigma.get_ylim()[1])
    ax_dist.set_ylim(0, y_max)
    ax_sigma.set_ylim(0, y_max)

    # -----------------------------------------------------------------------
    # Right – scatter: μ→GT distance vs. ||σ|| with linear fit and varying spread
    # -----------------------------------------------------------------------
    ax_sc = fig.add_subplot(gs[:, 1])

    x = df["euclidean_mm"].values
    y = df["sigma_total_mm"].values

    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    # Seaborn computes a varying confidence band over x (instead of fixed-width spread).
    sns.regplot(
        x=x,
        y=y,
        ax=ax_sc,
        ci=95,
        n_boot=2000,
        scatter_kws={
            "color": "#7AB6D9",
            "alpha": 0.55,
            "edgecolor": "none",
            "s": 28,
            "zorder": 2,
        },
        line_kws={
            "color": "#174A7E",
            "linewidth": 2,
            "zorder": 3,
        },
    )

    ax_sc.set_xlabel("μ→GT distance (mm)")
    ax_sc.set_ylabel("σ vector norm (mm)")
    ax_sc.set_title("Uncertainty vs. Error")
    max_xy = max(float(np.nanmax(x)), float(np.nanmax(y)))
    max_xy = max_xy * 1.03 if max_xy > 0 else 1.0
    ax_sc.set_xlim(0, max_xy)
    ax_sc.set_ylim(0, max_xy)
    ax_sc.set_aspect("equal", adjustable="box")
    ax_sc.grid(linestyle="--", alpha=0.4)

    fit_label = f"Linear fit: y = {slope:.3f}x + {intercept:.1f}  ($R^2$={r_value**2:.3f})"
    ax_sc.plot([], [], color="#174A7E", linewidth=2, label=fit_label)
    ax_sc.plot([], [], color="#174A7E", alpha=0.25, linewidth=8,
               label="95% confidence band")
    ax_sc.scatter([], [], color="#7AB6D9", alpha=0.55, s=28, label="Samples")
    ax_sc.legend(frameon=False, fontsize=9, loc="best")

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.93)

    # Make the scatter axis square and align its top/bottom with the violin stack.
    pos_top = ax_dist.get_position()
    pos_bottom = ax_sigma.get_position()
    stack_top = pos_top.y1
    stack_bottom = pos_bottom.y0
    stack_height = stack_top - stack_bottom

    fig_w, fig_h = fig.get_size_inches()
    square_width = stack_height * (fig_h / fig_w)

    left_column_right = max(pos_top.x1, pos_bottom.x1)
    gap = 0.03
    right_limit = 0.99
    x0_candidate = right_limit - square_width
    x0 = max(left_column_right + gap, x0_candidate)
    ax_sc.set_position([x0, stack_bottom, square_width, stack_height])

    # -----------------------------------------------------------------------
    # Save or show
    # -----------------------------------------------------------------------

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-validation results from eeg_spike_mil_regression_training."
    )
    parser.add_argument(
        "run_dir",
        help="Root directory containing the eeg_mil_regression_fold* sub-folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for the figure (e.g. results.png). "
             "If omitted, the plot is shown interactively.",
    )
    args = parser.parse_args()

    df = load_all_folds(args.run_dir)
    plot_results(df, output_path=args.output)


if __name__ == "__main__":
    main()
