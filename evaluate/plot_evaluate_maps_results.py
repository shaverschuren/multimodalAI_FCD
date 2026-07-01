"""
Plot evaluate_maps JSON outputs in single-run and compare-run modes.

This module reads evaluation JSON files produced by evaluate/evaluate_maps.py and
creates fixed-threshold and variable-threshold summary plots.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - optional dependency guard
    binomtest = None

try:
    from scipy.stats import kruskal, mannwhitneyu
except Exception:  # pragma: no cover - optional dependency guard
    kruskal = None
    mannwhitneyu = None

try:
    _sm_tables = importlib.import_module("statsmodels.stats.contingency_tables")
    mcnemar = getattr(_sm_tables, "mcnemar", None)
except Exception:  # pragma: no cover - optional dependency guard
    mcnemar = None


# Color scheme aligned with scripts_misc/plot/plot_mil_regression_validation.py
BLUE_DARK = "#A9C6D5"  # "#174A7E"
BLUE_MAIN = "#0076A8"  # "#8FC0DD"
BLUE_LIGHT = "#C1DDEE"
RED_MAIN = "#C91400"
RED_LIGHT = "#E3A7A7"
RED_TRUE = "#C00000"

RUN_COLORS = [BLUE_DARK, BLUE_MAIN, RED_MAIN, RED_LIGHT, BLUE_LIGHT]

BOXPLOT_FIG_WIDTH = 12
BOXPLOT_FIG_HEIGHT = 6
BOXPLOT_WIDTH_COMPARE = 0.7
BOXPLOT_WIDTH_SINGLE = 0.7
SIDE_LEGEND_X_ANCHOR = 1.12
COMBINED_LEGEND_Y_ANCHOR = 1.03

# Keep the gray plotting box (axes area) fixed and independent from margin changes.
# Values are [left, bottom, width, height] in figure coordinates.
BAR_AXES_RECT = [0.12, 0.14, 0.40, 0.74]
SUBJECT_BAR_AXES_RECT = [0.12, 0.14, 0.40 * 17./16., 0.74]
PAIRED_DIFF_AXES_RECT = [0.12, 0.14, 0.40, 0.74 * 3./4.]
PAIRED_DIFF_AXES_RECT_SECONDARY = [0.12, 0.14, 0.40 * 17./16., 0.74 * 3./4.]

# Errorbar cap sizes (points)
BAR_ERRORBAR_CAPSIZE = 4
DIFF_ERRORBAR_CAPSIZE = 5
AXES_BORDER_LINEWIDTH = 1.2

COMPACT_GRID_FIG_WIDTH = 16
COMPACT_GRID_TOP_ROW_HEIGHT = 3.6
COMPACT_GRID_BOTTOM_ROW_RATIO = 0.8
COMPACT_GRID_WSPACE = 0.10
COMPACT_GRID_HSPACE = 0.10

GRID_2X4_FIGSIZE = (16, 7.5)
GRID_2X4_LEFT = 0.04
GRID_2X4_RIGHT = 0.985
GRID_2X4_BOTTOM = 0.08
GRID_2X4_TOP = 0.96
GRID_2X4_WSPACE = 0.025
GRID_2X4_HSPACE = 0.06
GRID_2X4_BOTTOM_TO_TOP_HEIGHT_RATIO = 0.8

# Keep category-to-category spacing identical across plots.
PLOT_CATEGORY_SPACING = 1.25


@dataclass
class RunEvaluation:
    label: str
    json_path: Path
    payload: Dict[str, Any]


def _load_run(json_path: Path, label: Optional[str] = None) -> RunEvaluation:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    run_label = label or json_path.stem
    return RunEvaluation(label=run_label, json_path=json_path, payload=payload)


def _set_axes_border_width(ax: plt.Axes, linewidth: float = AXES_BORDER_LINEWIDTH) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def _set_compact_y_ticks(
    ax: plt.Axes,
    ylim: Optional[Tuple[float, float]] = None,
    n_ticks: int = 6,
) -> None:
    if ylim is None:
        lo, hi = ax.get_ylim()
    else:
        lo, hi = ylim
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return
    ax.set_yticks(np.linspace(float(lo), float(hi), int(n_ticks)))


def _strip_axes_decorations(
    ax: plt.Axes,
    keep_y_ticks: bool = False,
    y_tick_side: Optional[str] = None,
) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    if not keep_y_ticks:
        ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(
        axis="y",
        which="both",
        left=bool(keep_y_ticks and y_tick_side == "left"),
        right=bool(keep_y_ticks and y_tick_side == "right"),
        labelleft=False,
        labelright=False,
        length=4 if keep_y_ticks else 0,
    )


def _strip_axis_for_grid(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
    )


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in {"true", "1", "yes", "y"}:
            return True
        if vv in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_fold_id(path: Any) -> Optional[str]:
    if path is None:
        return None
    for part in Path(str(path)).parts:
        if re.fullmatch(r"fold_\d+", part, re.IGNORECASE):
            return part.lower()
    return None


def _harmonic_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return 0.0
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _recompute_cluster_f1_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cluster_det_precision" in out.columns and "cluster_det_recall" in out.columns:
        out["cluster_det_f1"] = [
            _harmonic_f1(_as_float(p), _as_float(r))
            for p, r in zip(out["cluster_det_precision"], out["cluster_det_recall"])
        ]
    if "cluster_pin_precision" in out.columns and "cluster_pin_recall" in out.columns:
        out["cluster_pin_f1"] = [
            _harmonic_f1(_as_float(p), _as_float(r))
            for p, r in zip(out["cluster_pin_precision"], out["cluster_pin_recall"])
        ]
    return out


def _extract_subject_rows(run: RunEvaluation) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for entry in run.payload.get("per_subject", []):
        fm = entry.get("fixed_threshold_metrics", {})
        vox = fm.get("voxel", {})
        cluster = fm.get("cluster", {})
        det = cluster.get("detection", {})
        pin = cluster.get("pinpointing", {})

        det_prec = _as_float(det.get("cluster_precision"))
        det_rec = _as_float(det.get("cluster_sensitivity"))
        pin_prec = _as_float(pin.get("cluster_precision"))
        pin_rec = _as_float(pin.get("cluster_sensitivity"))
        voxel_prec = _as_float(vox.get("voxel_precision"))
        voxel_rec = _as_float(vox.get("voxel_recall"))
        voxel_dice = _as_float(vox.get("voxel_dice"))
        if voxel_dice is None and (voxel_prec is None or voxel_rec is None):
            voxel_dice = 0.0
        det_f1 = _as_float(det.get("cluster_f1"))
        if det_f1 is None:
            det_f1 = _harmonic_f1(det_prec, det_rec)
        pin_f1 = _as_float(pin.get("cluster_f1"))
        if pin_f1 is None:
            pin_f1 = _harmonic_f1(pin_prec, pin_rec)

        is_control = bool(entry.get("is_control", False))
        gt_empty = _as_bool(fm.get("gt_empty"))

        rows.append(
            {
                "run": run.label,
                "fold_id": entry.get("fold_id") or _extract_fold_id(entry.get("prediction_path")),
                "split_role": entry.get("split_role"),
                "subject_id": entry.get("subject_id"),
                "is_control": is_control,
                "gt_empty": gt_empty,
                "voxel_dice": voxel_dice,
                "voxel_precision": voxel_prec,
                "voxel_recall": voxel_rec,
                "voxel_tp": _as_float(vox.get("tp")),
                "voxel_fp": _as_float(vox.get("fp")),
                "voxel_fn": _as_float(vox.get("fn")),
                "voxel_tn": _as_float(vox.get("tn")),
                "cluster_det_precision": det_prec,
                "cluster_det_recall": det_rec,
                "cluster_det_f1": det_f1,
                "cluster_pin_precision": pin_prec,
                "cluster_pin_recall": pin_rec,
                "cluster_pin_f1": pin_f1,
                "subject_detected": 1.0 if bool(det.get("subject_detected", False)) else 0.0,
                "subject_pinpointed": 1.0 if bool(pin.get("subject_pinpointed", False)) else 0.0,
                "n_pred_clusters": _as_float(cluster.get("n_pred_clusters")),
                "n_fp_det_clusters": _as_float(det.get("n_fp_pred_clusters")),
                "n_fp_pin_clusters": _as_float(pin.get("n_fp_pred_clusters")),
            }
        )

    return pd.DataFrame(rows)


def _aggregate_distribution_units(subject_df: pd.DataFrame, by_subject: bool) -> pd.DataFrame:
    if subject_df.empty or by_subject:
        return _recompute_cluster_f1_columns(subject_df)

    unit_df = subject_df.copy()
    unit_df["fold_id"] = unit_df["fold_id"].fillna("__no_fold__")

    metric_cols = [
        "subject_detected",
        "subject_pinpointed",
        "n_pred_clusters",
        "n_fp_det_clusters",
        "n_fp_pin_clusters",
        "voxel_dice",
        "voxel_precision",
        "voxel_recall",
        "cluster_det_precision",
        "cluster_det_recall",
        "cluster_pin_precision",
        "cluster_pin_recall",
    ]

    grouped = (
        unit_df.groupby(["run", "fold_id", "is_control"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    grouped = _recompute_cluster_f1_columns(grouped)
    grouped["subject_id"] = grouped["fold_id"]
    grouped["gt_empty"] = np.nan
    return grouped


def _subject_summary_distribution(subject_df: pd.DataFrame, by_subject: bool) -> pd.DataFrame:
    if subject_df.empty:
        return pd.DataFrame(columns=["run", "metric", "value"])

    out_rows: List[Dict[str, Any]] = []
    if by_subject:
        unit_groups = [(None, g) for _, g in subject_df.groupby("run")]
    else:
        fold_df = subject_df.copy()
        fold_df["fold_id"] = fold_df["fold_id"].fillna("__no_fold__")
        unit_groups = list(fold_df.groupby(["run", "fold_id"]))

    for key, unit_df in unit_groups:
        if by_subject:
            run_name = str(unit_df["run"].iloc[0])
        else:
            run_name = str(key[0])

        cases = unit_df[~unit_df["is_control"]]
        controls = unit_df[unit_df["is_control"]]

        if by_subject:
            for _, row in cases.iterrows():
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "detection_rate",
                        "value": float(row["subject_detected"]),
                    }
                )
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "pinpointing_rate",
                        "value": float(row["subject_pinpointed"]),
                    }
                )

            for _, row in controls.iterrows():
                fp_subject = 1.0 if float(row["n_pred_clusters"]) > 0.0 else 0.0
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "false_positive_subject_rate",
                        "value": fp_subject,
                    }
                )
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "specificity",
                        "value": 1.0 - fp_subject,
                    }
                )
        else:
            if not cases.empty:
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "detection_rate",
                        "value": float(cases["subject_detected"].mean()),
                    }
                )
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "pinpointing_rate",
                        "value": float(cases["subject_pinpointed"].mean()),
                    }
                )

            if not controls.empty:
                fp_subject_rate = float((controls["n_pred_clusters"] > 0).mean())
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "false_positive_subject_rate",
                        "value": fp_subject_rate,
                    }
                )
                out_rows.append(
                    {
                        "run": run_name,
                        "metric": "specificity",
                        "value": 1.0 - fp_subject_rate,
                    }
                )

    return pd.DataFrame(out_rows)


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 12345,
) -> Optional[List[float]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    if vals.size == 1:
        x = float(vals[0])
        return [x, x]
    rng = np.random.default_rng(seed)
    n = vals.size
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = vals[rng.integers(0, n, size=n)]
        if statistic == "median":
            boot[i] = float(np.median(sample))
        else:
            boot[i] = float(np.mean(sample))
    alpha = (1.0 - float(ci_level)) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


def _sign_flip_permutation_pvalue(
    differences: np.ndarray,
    n_permutations: int,
    seed: int,
) -> Optional[float]:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    n = diffs.size
    perms = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        perms[i] = float(np.mean(diffs * signs))
    p = float((np.sum(np.abs(perms) >= abs(obs)) + 1) / (len(perms) + 1))
    return p


def _mcnemar_or_sign_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[valid]
    bb = bb[valid]
    out: Dict[str, Any] = {"test": None, "p_value": None}
    if aa.size == 0:
        return out

    n01 = int(np.sum((aa == 0) & (bb == 1)))
    n10 = int(np.sum((aa == 1) & (bb == 0)))
    out["discordant_01"] = n01
    out["discordant_10"] = n10

    if mcnemar is not None:
        table = np.array(
            [
                [int(np.sum((aa == 0) & (bb == 0))), n01],
                [n10, int(np.sum((aa == 1) & (bb == 1)))],
            ]
        )
        try:
            res = mcnemar(table, exact=True)
            out["test"] = "mcnemar_exact"
            out["p_value"] = float(res.pvalue)
            return out
        except Exception:
            pass

    if binomtest is not None and (n01 + n10) > 0:
        out["test"] = "sign_test_binomial"
        out["p_value"] = float(binomtest(n01, n01 + n10, p=0.5, alternative="two-sided").pvalue)
        return out

    out["test"] = "unavailable"
    return out


def _summary_patient_bootstrap_rows(
    subject_df: pd.DataFrame,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
    for run_name, run_df in subject_df.groupby("run"):
        controls = run_df[run_df["is_control"]]
        zero_fill_metrics = {
            "voxel_dice",
            "voxel_precision",
            "voxel_recall",
            "cluster_det_precision",
            "cluster_det_recall",
            "cluster_det_f1",
            "cluster_pin_precision",
            "cluster_pin_recall",
            "cluster_pin_f1",
        }
        for metric in metrics:
            series = run_df[metric]
            if metric in zero_fill_metrics:
                series = series.fillna(0.0)
            vals = series.dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "estimate": float(np.mean(vals)),
                    "ci_lower": (_bootstrap_ci(vals, "mean", n_bootstrap, ci_level, seed) or [None, None])[0],
                    "ci_upper": (_bootstrap_ci(vals, "mean", n_bootstrap, ci_level, seed) or [None, None])[1],
                    "n_subjects": int(vals.size),
                    "unit": "subject",
                    "ci_method": "patient_bootstrap",
                }
            )

        pin_vals = run_df["subject_pinpointed"].fillna(0.0).to_numpy(dtype=float)
        det_vals = run_df["subject_detected"].fillna(0.0).to_numpy(dtype=float)
        fp_vals = (controls["n_pred_clusters"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        spec_vals = 1.0 - fp_vals
        for metric_name, vals in [
            ("subject_detected", det_vals),
            ("subject_pinpointed", pin_vals),
            ("false_positive_subject_rate", fp_vals),
            ("specificity", spec_vals),
        ]:
            if vals.size == 0:
                continue
            ci = _bootstrap_ci(vals, "mean", n_bootstrap, ci_level, seed)
            rows.append(
                {
                    "run": run_name,
                    "metric": metric_name,
                    "estimate": float(np.mean(vals)),
                    "ci_lower": (ci or [None, None])[0],
                    "ci_upper": (ci or [None, None])[1],
                    "n_subjects": int(vals.size),
                    "unit": "subject",
                    "ci_method": "patient_bootstrap",
                }
            )

    return pd.DataFrame(rows)


def _validate_compare_inputs(subject_df: pd.DataFrame) -> List[str]:
    warnings_out: List[str] = []
    for run_name, run_df in subject_df.groupby("run"):
        dupes = run_df["subject_id"].duplicated(keep=False)
        if bool(dupes.any()):
            warnings_out.append(
                f"[{run_name}] Duplicate subject IDs detected: {sorted(run_df.loc[dupes, 'subject_id'].astype(str).unique().tolist())[:10]}"
            )
    return warnings_out


def _pair_validation_warnings(
    subject_df: pd.DataFrame,
    baseline: str,
    comparator: str,
    min_paired_warn: int = 20,
) -> List[str]:
    warnings_out: List[str] = []
    a_df = subject_df[subject_df["run"] == baseline].copy()
    b_df = subject_df[subject_df["run"] == comparator].copy()
    a_ids = set(a_df["subject_id"].dropna().astype(str))
    b_ids = set(b_df["subject_id"].dropna().astype(str))
    shared = a_ids & b_ids
    only_a = sorted(a_ids - b_ids)
    only_b = sorted(b_ids - a_ids)

    if not shared:
        warnings_out.append(
            f"[{baseline} vs {comparator}] No overlapping subject IDs; paired comparison is invalid."
        )
        return warnings_out

    if only_a:
        warnings_out.append(
            f"[{baseline} vs {comparator}] Subjects only in baseline (showing up to 10): {only_a[:10]}"
        )
    if only_b:
        warnings_out.append(
            f"[{baseline} vs {comparator}] Subjects only in comparator (showing up to 10): {only_b[:10]}"
        )

    joined = a_df.merge(b_df, on="subject_id", suffixes=("_a", "_b"), how="inner")
    if joined.empty:
        warnings_out.append(
            f"[{baseline} vs {comparator}] Inner join produced zero rows after duplicate filtering."
        )
        return warnings_out

    ctrl_mismatch = joined[joined["is_control_a"] != joined["is_control_b"]]
    if not ctrl_mismatch.empty:
        warnings_out.append(
            f"[{baseline} vs {comparator}] is_control mismatch for {len(ctrl_mismatch)} paired subjects."
        )

    if "fold_id_a" in joined.columns and "fold_id_b" in joined.columns:
        fold_mismatch = joined[
            joined["fold_id_a"].notna() & joined["fold_id_b"].notna() & (joined["fold_id_a"] != joined["fold_id_b"])
        ]
        if not fold_mismatch.empty:
            warnings_out.append(
                f"[{baseline} vs {comparator}] fold_id mismatch for {len(fold_mismatch)} paired subjects. "
                "Allowed if each remains a held-out prediction, but verify provenance."
            )

    if len(joined) < int(min_paired_warn):
        warnings_out.append(
            f"[{baseline} vs {comparator}] Only {len(joined)} paired subjects available (<{min_paired_warn})."
        )
    return warnings_out


def _paired_compare_two_runs(
    run_a: str,
    run_b: str,
    subject_df: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    a_df = subject_df[subject_df["run"] == run_a].copy()
    b_df = subject_df[subject_df["run"] == run_b].copy()
    joined = a_df.merge(b_df, on="subject_id", suffixes=("_a", "_b"), how="inner")
    rows: List[Dict[str, Any]] = []

    metrics_cont = [
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
    metrics_bin = ["subject_detected", "subject_pinpointed"]

    for metric in metrics_cont:
        ca = f"{metric}_a"
        cb = f"{metric}_b"
        if ca not in joined.columns or cb not in joined.columns:
            continue
        diff = (joined[cb] - joined[ca]).dropna().to_numpy(dtype=float)
        if diff.size == 0:
            continue
        mean_ci = _bootstrap_ci(diff, "mean", n_bootstrap, ci_level, seed)
        median_ci = _bootstrap_ci(diff, "median", n_bootstrap, ci_level, seed + 1)
        p_perm = _sign_flip_permutation_pvalue(diff, n_permutations, seed)
        rows.append(
            {
                "baseline": run_a,
                "comparator": run_b,
                "direction": "comparator-baseline",
                "metric": metric,
                "n_paired": int(diff.size),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "mean_diff_ci_lower": (mean_ci or [None, None])[0],
                "mean_diff_ci_upper": (mean_ci or [None, None])[1],
                "median_diff_ci_lower": (median_ci or [None, None])[0],
                "median_diff_ci_upper": (median_ci or [None, None])[1],
                "p_value_signflip_mean": p_perm,
                "analysis": "paired_continuous",
            }
        )

    for metric in metrics_bin:
        ca = f"{metric}_a"
        cb = f"{metric}_b"
        if ca not in joined.columns or cb not in joined.columns:
            continue
        diff = (joined[cb] - joined[ca]).dropna().to_numpy(dtype=float)
        if diff.size == 0:
            continue
        mean_ci = _bootstrap_ci(diff, "mean", n_bootstrap, ci_level, seed)
        p_perm = _sign_flip_permutation_pvalue(diff, n_permutations, seed)
        mcn = _mcnemar_or_sign_test(joined[ca].to_numpy(dtype=float), joined[cb].to_numpy(dtype=float))
        rows.append(
            {
                "baseline": run_a,
                "comparator": run_b,
                "direction": "comparator-baseline",
                "metric": metric,
                "n_paired": int(diff.size),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "mean_diff_ci_lower": (mean_ci or [None, None])[0],
                "mean_diff_ci_upper": (mean_ci or [None, None])[1],
                "median_diff_ci_lower": None,
                "median_diff_ci_upper": None,
                "p_value_signflip_mean": p_perm,
                "mcnemar_test": mcn.get("test"),
                "mcnemar_p": mcn.get("p_value"),
                "analysis": "paired_binary",
            }
        )

    # Controls binary FP indicator
    ctrl = joined[(joined["is_control_a"] == True) & (joined["is_control_b"] == True)].copy()
    if not ctrl.empty:
        fp_a = (ctrl["n_pred_clusters_a"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        fp_b = (ctrl["n_pred_clusters_b"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float)
        diff = fp_b - fp_a
        mean_ci = _bootstrap_ci(diff, "mean", n_bootstrap, ci_level, seed)
        p_perm = _sign_flip_permutation_pvalue(diff, n_permutations, seed)
        mcn = _mcnemar_or_sign_test(fp_a, fp_b)
        rows.append(
            {
                "baseline": run_a,
                "comparator": run_b,
                "direction": "comparator-baseline",
                "metric": "false_positive_subject_indicator_controls",
                "n_paired": int(diff.size),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "mean_diff_ci_lower": (mean_ci or [None, None])[0],
                "mean_diff_ci_upper": (mean_ci or [None, None])[1],
                "median_diff_ci_lower": None,
                "median_diff_ci_upper": None,
                "p_value_signflip_mean": p_perm,
                "mcnemar_test": mcn.get("test"),
                "mcnemar_p": mcn.get("p_value"),
                "analysis": "paired_binary_controls",
            }
        )

    return pd.DataFrame(rows)


def _extract_curve_df(
    run: RunEvaluation,
    curve_key: str,
    x_key: str,
    y_key: str,
    x_fallback_key: Optional[str] = None,
    clip_x_to_unit: bool = False,
) -> pd.DataFrame:
    curves = run.payload.get("curves", {})
    points = curves.get(curve_key, {}).get("points", [])

    rows: List[Dict[str, Any]] = []
    for p in points:
        thr = _as_float(p.get("threshold"))
        x = _as_float(p.get(x_key))
        if x is None and x_fallback_key is not None:
            x = _as_float(p.get(x_fallback_key))
            if x is not None and clip_x_to_unit:
                x = float(np.clip(x, 0.0, 1.0))

        y = _as_float(p.get(y_key))
        if x is None or y is None:
            continue

        rows.append(
            {
                "run": run.label,
                "threshold": thr,
                "x": x,
                "y": y,
            }
        )

    return pd.DataFrame(rows)


def _p_to_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []

    order = np.argsort(pvals)
    sorted_p = [float(pvals[i]) for i in order]

    adjusted_sorted: List[float] = [0.0] * m
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(1.0, running_max)

    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def _compute_boxplot_significance(
    df: pd.DataFrame,
    metric_col: str,
    value_col: str,
    run_col: str,
    run_order: Sequence[str],
    alpha: float = 0.05,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare runs per metric using non-parametric tests.

    - 2 runs: Mann-Whitney U (two-sided).
    - >2 runs: Kruskal-Wallis omnibus, then pairwise Mann-Whitney U with Holm correction.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    if mannwhitneyu is None:
        return out

    metric_values = [m for m in df[metric_col].dropna().unique().tolist()]
    for metric_name in metric_values:
        metric_df = df[df[metric_col] == metric_name].dropna(subset=[value_col, run_col])
        if metric_df.empty:
            continue

        present_runs = [r for r in run_order if r in set(metric_df[run_col].unique())]
        groups: List[tuple[str, np.ndarray]] = []
        for run_name in present_runs:
            vals = metric_df.loc[metric_df[run_col] == run_name, value_col].dropna().to_numpy(dtype=float)
            if vals.size > 0:
                groups.append((run_name, vals))

        if len(groups) < 2:
            continue

        metric_results: List[Dict[str, Any]] = []
        if len(groups) == 2:
            (ra, va), (rb, vb) = groups
            p = float(mannwhitneyu(va, vb, alternative="two-sided", method="auto").pvalue)
            metric_results.append(
                {
                    "run_a": ra,
                    "run_b": rb,
                    "p_raw": p,
                    "p_adj": p,
                    "test": "Mann-Whitney U",
                    "n_a": int(va.size),
                    "n_b": int(vb.size),
                    "significant": bool(p < alpha),
                }
            )
        else:
            if kruskal is None:
                continue
            omnibus_p = float(kruskal(*[vals for _, vals in groups]).pvalue)
            if omnibus_p < alpha:
                pair_rows: List[Dict[str, Any]] = []
                pvals: List[float] = []
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        ra, va = groups[i]
                        rb, vb = groups[j]
                        p = float(
                            mannwhitneyu(va, vb, alternative="two-sided", method="auto").pvalue
                        )
                        pair_rows.append(
                            {
                                "run_a": ra,
                                "run_b": rb,
                                "p_raw": p,
                                "test": "Mann-Whitney U (post-hoc)",
                                "n_a": int(va.size),
                                "n_b": int(vb.size),
                                "omnibus_test": "Kruskal-Wallis",
                                "omnibus_p": omnibus_p,
                            }
                        )
                        pvals.append(p)

                padj = _holm_adjust(pvals)
                for row, p_adj in zip(pair_rows, padj):
                    row["p_adj"] = float(p_adj)
                    row["significant"] = bool(p_adj < alpha)
                metric_results.extend(pair_rows)

        if metric_results:
            out[str(metric_name)] = metric_results

    return out


def _annotate_significance_brackets(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    value_col: str,
    run_col: str,
    metric_order: Sequence[str],
    run_order: Sequence[str],
    sig_by_metric: Dict[str, List[Dict[str, Any]]],
    group_width: float,
    preferred_side: str = "auto",
    anchor_offset_fraction: float = 0.03,
    fixed_anchor_y: Optional[float] = None,
) -> None:
    if not sig_by_metric:
        return

    n_runs = max(1, len(run_order))
    run_to_idx = {r: i for i, r in enumerate(run_order)}
    x_ticks = list(ax.get_xticks())
    x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    metric_to_x = {label: pos for label, pos in zip(x_labels, x_ticks)}

    base_ylim = ax.get_ylim()
    y_range = max(1e-6, base_ylim[1] - base_ylim[0])
    line_h = 0.018 * y_range
    gap_h = 0.04 * y_range
    text_h = 0.008 * y_range
    required_top = base_ylim[1]
    required_bottom = base_ylim[0]

    # Decide one global side (top/bottom) per plot for visual consistency.
    all_metric_vals = df[df[metric_col].isin(metric_order)][value_col].dropna()
    if all_metric_vals.empty:
        return

    if preferred_side == "above":
        place_top_global = True
    elif preferred_side == "below":
        place_top_global = False
    else:
        global_min = float(all_metric_vals.min())
        global_max = float(all_metric_vals.max())
        space_top = base_ylim[1] - global_max
        space_bottom = global_min - base_ylim[0]
        place_top_global = space_top >= space_bottom

    for m_idx, metric_name in enumerate(metric_order):
        rows = [r for r in sig_by_metric.get(metric_name, []) if "p_adj" in r]
        if not rows:
            continue
        if metric_name not in metric_to_x:
            continue

        metric_vals = df.loc[df[metric_col] == metric_name, value_col].dropna()
        if metric_vals.empty:
            continue

        metric_min = float(metric_vals.min())
        metric_max = float(metric_vals.max())
        n_rows = len(rows)

        # If requested, force a fixed vertical anchor in data coordinates.
        if fixed_anchor_y is not None:
            y = float(fixed_anchor_y)
        # For explicit placement, anchor to axis bounds so location is guaranteed.
        elif preferred_side == "above":
            y = (
                base_ylim[1]
                - (n_rows - 1) * gap_h
                - line_h
                - text_h
                - anchor_offset_fraction * y_range
            )
        elif preferred_side == "below":
            y = (
                base_ylim[0]
                + (n_rows - 1) * gap_h
                + line_h
                + text_h
                + anchor_offset_fraction * y_range
            )
        elif place_top_global:
            y = metric_max + anchor_offset_fraction * y_range
        else:
            y = metric_min - anchor_offset_fraction * y_range

        # Draw shorter-distance comparisons first to reduce crossings.
        rows.sort(
            key=lambda r: abs(run_to_idx.get(str(r.get("run_a")), 0) - run_to_idx.get(str(r.get("run_b")), 0))
        )

        for row in rows:
            ra = str(row.get("run_a"))
            rb = str(row.get("run_b"))
            if ra not in run_to_idx or rb not in run_to_idx:
                continue

            ia = run_to_idx[ra]
            ib = run_to_idx[rb]
            if ia == ib:
                continue

            x_center = float(metric_to_x[metric_name])
            box_w = group_width / n_runs
            x1 = x_center - group_width / 2.0 + (ia + 0.5) * box_w
            x2 = x_center - group_width / 2.0 + (ib + 0.5) * box_w
            if x1 > x2:
                x1, x2 = x2, x1

            p_adj = float(row.get("p_adj", 1.0))
            stars = _p_to_stars(p_adj)
            label = stars if stars != "ns" else f"p={p_adj:.3g}"

            if place_top_global:
                ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=1.0, c="black")
                ax.text(
                    (x1 + x2) / 2.0,
                    y + line_h + text_h,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )
                y += gap_h
            else:
                ax.plot([x1, x1, x2, x2], [y, y - line_h, y - line_h, y], lw=1.0, c="black")
                ax.text(
                    (x1 + x2) / 2.0,
                    y - line_h - text_h,
                    label,
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="black",
                )
                y -= gap_h

        if place_top_global:
            required_top = max(required_top, y + line_h + text_h)
        else:
            required_bottom = min(required_bottom, y - line_h - text_h)

    if required_top > base_ylim[1] or required_bottom < base_ylim[0]:
        ax.set_ylim(required_bottom - 0.01 * y_range, required_top + 0.01 * y_range)


def _print_significance_summary(
    plot_name: str,
    sig_by_metric: Dict[str, List[Dict[str, Any]]],
) -> None:
    if not sig_by_metric:
        print(f"[stats] {plot_name}: no significance results (insufficient data or scipy unavailable).")
        return

    print(
        "[stats] Using non-parametric tests: Mann-Whitney U for 2 runs, "
        "or Kruskal-Wallis + Holm-corrected post-hoc Mann-Whitney U for >2 runs."
    )
    print(f"[stats] {plot_name}")

    for metric_name, rows in sig_by_metric.items():
        for row in rows:
            ra = str(row.get("run_a"))
            rb = str(row.get("run_b"))
            p_adj = float(row.get("p_adj", 1.0))
            stars = _p_to_stars(p_adj)
            sig_txt = "significant" if bool(row.get("significant", False)) else "not significant"
            print(
                f"  - {metric_name}: {ra} vs {rb}, p_adj={p_adj:.4g} ({stars}), {sig_txt}"
            )


def _plot_metric_boxplots(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    metric_name_map: Dict[str, str],
    mode: str,
    title: str,
    out_path: Path,
    bracket_side: str = "auto",
    hue_order: Optional[List[str]] = None,
    single_color: str = BLUE_MAIN,
    compare_palette: Optional[Dict[str, str]] = None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 12345,
) -> None:
    if df.empty:
        return

    melted = df[list(metric_cols) + ["run"]].melt(
        id_vars=["run"],
        value_vars=list(metric_cols),
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        return

    melted["metric"] = melted["metric"].map(metric_name_map)
    metric_order_plot = [
        metric_name_map[c]
        for c in metric_cols
        if metric_name_map[c] in set(melted["metric"].unique())
    ]

    fig, ax = plt.subplots(figsize=(BOXPLOT_FIG_WIDTH, BOXPLOT_FIG_HEIGHT))
    ax.set_position(BAR_AXES_RECT)
    run_order = hue_order or sorted(melted["run"].dropna().unique().tolist())
    is_cluster_triplet = set(metric_cols) in [
        {"cluster_det_precision", "cluster_det_recall", "cluster_det_f1"},
        {"cluster_pin_precision", "cluster_pin_recall", "cluster_pin_f1"},
    ]
    if is_cluster_triplet:
        p_col = "cluster_det_precision" if "cluster_det_precision" in set(metric_cols) else "cluster_pin_precision"
        r_col = "cluster_det_recall" if "cluster_det_recall" in set(metric_cols) else "cluster_pin_recall"
        summary_df = _summarize_cluster_triplet_for_bars(
            wide_df=df,
            run_order=run_order,
            precision_col=p_col,
            recall_col=r_col,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
    else:
        summary_df = _summarize_for_bars(
            melted_df=melted,
            metric_order=metric_order_plot,
            run_order=run_order,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
    _plot_grouped_bars_with_ci(
        ax=ax,
        summary_df=summary_df,
        metric_order=metric_order_plot,
        run_order=run_order,
        mode=mode,
        single_color=single_color,
        compare_palette=compare_palette,
        ylabel="score",
        ylim=(0, 1),
        title=title,
    )
    if mode == "compare":
        ax.legend(title="run", bbox_to_anchor=(SIDE_LEGEND_X_ANCHOR, 1), loc="upper left")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _summarize_for_bars(
    melted_df: pd.DataFrame,
    metric_order: Sequence[str],
    run_order: Sequence[str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric_name in metric_order:
        metric_df = melted_df[melted_df["metric"] == metric_name]
        if metric_df.empty:
            continue
        for run_name in run_order:
            vals = (
                metric_df.loc[metric_df["run"] == run_name, "value"]
                .dropna()
                .to_numpy(dtype=float)
            )
            if vals.size == 0:
                continue
            ci = _bootstrap_ci(
                vals,
                statistic="mean",
                n_boot=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
            rows.append(
                {
                    "metric": metric_name,
                    "run": run_name,
                    "estimate": float(np.mean(vals)),
                    "ci_low": (ci or [None, None])[0],
                    "ci_high": (ci or [None, None])[1],
                    "n": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def _summarize_cluster_triplet_for_bars(
    wide_df: pd.DataFrame,
    run_order: Sequence[str],
    precision_col: str,
    recall_col: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    def _paired_f1_ci(p_vals: np.ndarray, r_vals: np.ndarray, n_boot: int, ci: float, seed_local: int) -> Optional[List[float]]:
        pp = np.asarray(p_vals, dtype=float)
        rr = np.asarray(r_vals, dtype=float)
        valid = np.isfinite(pp) & np.isfinite(rr)
        pp = pp[valid]
        rr = rr[valid]
        if pp.size == 0:
            return None
        if pp.size == 1:
            x = _harmonic_f1(float(pp[0]), float(rr[0]))
            return None if x is None else [float(x), float(x)]
        rng = np.random.default_rng(seed_local)
        n = pp.size
        boot = np.empty(int(n_boot), dtype=float)
        for i in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            p_mean = float(np.mean(pp[idx]))
            r_mean = float(np.mean(rr[idx]))
            f1 = _harmonic_f1(p_mean, r_mean)
            boot[i] = np.nan if f1 is None else float(f1)
        boot = boot[np.isfinite(boot)]
        if boot.size == 0:
            return None
        alpha = (1.0 - float(ci)) / 2.0
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
        return [float(lo), float(hi)]

    rows: List[Dict[str, Any]] = []
    for run_name in run_order:
        run_df = wide_df[wide_df["run"] == run_name]
        if run_df.empty:
            continue
        paired = run_df[[precision_col, recall_col]].dropna()
        if paired.empty:
            continue
        pp = paired[precision_col].to_numpy(dtype=float)
        rr = paired[recall_col].to_numpy(dtype=float)
        p_est = float(np.mean(pp))
        r_est = float(np.mean(rr))
        f_est = _harmonic_f1(p_est, r_est)
        p_ci = _bootstrap_ci(pp, statistic="mean", n_boot=n_bootstrap, ci_level=ci_level, seed=seed)
        r_ci = _bootstrap_ci(rr, statistic="mean", n_boot=n_bootstrap, ci_level=ci_level, seed=seed + 1)
        f_ci = _paired_f1_ci(pp, rr, n_bootstrap, ci_level, seed + 2)
        n = int(paired.shape[0])

        rows.extend(
            [
                {"metric": "Precision", "run": run_name, "estimate": p_est, "ci_low": (p_ci or [None, None])[0], "ci_high": (p_ci or [None, None])[1], "n": n},
                {"metric": "Recall", "run": run_name, "estimate": r_est, "ci_low": (r_ci or [None, None])[0], "ci_high": (r_ci or [None, None])[1], "n": n},
                {"metric": "F1", "run": run_name, "estimate": f_est, "ci_low": (f_ci or [None, None])[0], "ci_high": (f_ci or [None, None])[1], "n": n},
            ]
        )
    return pd.DataFrame(rows)


def _plot_grouped_bars_with_ci(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    metric_order: Sequence[str],
    run_order: Sequence[str],
    mode: str,
    single_color: str,
    compare_palette: Optional[Dict[str, str]],
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    if summary_df.empty:
        if title:
            ax.set_title(f"{title} (no data)")
        _set_axes_border_width(ax)
        return

    x = np.arange(len(metric_order), dtype=float)
    n_runs = max(1, len(run_order))
    width = 0.75 / n_runs if mode == "compare" else 0.55

    for i, run_name in enumerate(run_order):
        run_df = summary_df[summary_df["run"] == run_name]
        if run_df.empty:
            continue
        est = []
        yerr_low = []
        yerr_high = []
        for metric_name in metric_order:
            row = run_df[run_df["metric"] == metric_name]
            if row.empty:
                est.append(np.nan)
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
                continue
            r = row.iloc[0]
            e = float(r["estimate"])
            lo = r["ci_low"]
            hi = r["ci_high"]
            est.append(e)
            if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
            else:
                yerr_low.append(max(0.0, e - float(lo)))
                yerr_high.append(max(0.0, float(hi) - e))

        if mode == "compare":
            offsets = x - 0.375 + (i + 0.5) * width
            color = (compare_palette or {}).get(run_name, RUN_COLORS[i % len(RUN_COLORS)])
            label = run_name
        else:
            offsets = x
            color = single_color
            label = None

        ax.bar(
            offsets,
            est,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            label=label,
            zorder=2,
        )
        ax.errorbar(
            offsets,
            est,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=BAR_ERRORBAR_CAPSIZE,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_order)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _set_axes_border_width(ax)
    ax.grid(axis="y", alpha=0.25)
    if title:
        ax.set_title(title)


def _plot_subject_bars(
    summary_distribution_df: pd.DataFrame,
    case_distribution_df: pd.DataFrame,
    mode: str,
    out_path: Path,
    distribution_unit_label: str,
    hue_order: Optional[List[str]] = None,
    single_color: str = BLUE_MAIN,
    compare_palette: Optional[Dict[str, str]] = None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 12345,
) -> None:
    if summary_distribution_df.empty:
        return

    rate_metrics = ["detection_rate", "pinpointing_rate", "false_positive_subject_rate", "specificity"]

    metric_name_map = {
        "detection_rate": "Detection rate",
        "pinpointing_rate": "Pinpointing rate",
        "false_positive_subject_rate": "False positive subject rate",
        "specificity": "Specificity",
    }

    rate_dist_df = summary_distribution_df.copy()
    rate_dist_df = rate_dist_df[rate_dist_df["metric"].isin(rate_metrics)]
    rate_dist_df["metric"] = rate_dist_df["metric"].map(metric_name_map)
    rate_metric_order = [
        metric_name_map[m]
        for m in rate_metrics
        if metric_name_map[m] in set(rate_dist_df["metric"].unique())
    ]

    fig, ax1 = plt.subplots(figsize=(BOXPLOT_FIG_WIDTH + 1, BOXPLOT_FIG_HEIGHT + 1))
    run_order = hue_order or sorted(summary_distribution_df["run"].dropna().unique().tolist())
    ax2 = ax1.twinx()
    ax1.set_position(SUBJECT_BAR_AXES_RECT)
    ax2.set_position(SUBJECT_BAR_AXES_RECT)
    _set_axes_border_width(ax1)
    _set_axes_border_width(ax2)

    if not rate_dist_df.empty:
        rate_summary = _summarize_for_bars(
            melted_df=rate_dist_df,
            metric_order=rate_metric_order,
            run_order=run_order,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        _plot_grouped_bars_with_ci(
            ax=ax1,
            summary_df=rate_summary,
            metric_order=rate_metric_order,
            run_order=run_order,
            mode=mode,
            single_color=single_color,
            compare_palette=compare_palette,
            ylabel="rate",
            ylim=(0, 1),
            title=f"Fixed threshold: summary (distribution over {distribution_unit_label}s)",
        )
        if mode == "compare":
            ax1.legend(title="run", bbox_to_anchor=(SIDE_LEGEND_X_ANCHOR, 1), loc="upper left")

    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if rate_dist_df.empty:
        ax1.set_title(f"Fixed threshold: summary (distribution over {distribution_unit_label}s)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="both", length=0)
    ax1.grid(axis="y", alpha=0.25)

    rate_count = len(rate_metric_order)
    fp_gap = PLOT_CATEGORY_SPACING
    if rate_count > 0:
        fp_center = float((rate_count - 1) + fp_gap)
        ax1.axvline(
            x=float((rate_count - 1) + fp_gap / 2.0),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
    else:
        fp_center = 0.0
    fp_metric_name = "Number of FP clusters"
    if not case_distribution_df.empty:
        fp_cols = ["n_fp_det_clusters"]
        fp_data = case_distribution_df[["run"] + fp_cols].dropna()
        
        if not fp_data.empty:
            fp_melted = fp_data.melt(
                id_vars=["run"],
                value_vars=fp_cols,
                var_name="metric",
                value_name="value",
            )
            fp_melted["metric"] = fp_metric_name
            fp_summary = _summarize_for_bars(
                melted_df=fp_melted,
                metric_order=[fp_metric_name],
                run_order=run_order,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed + 1,
            )

            n_runs = max(1, len(run_order))
            width = 0.75 / n_runs if mode == "compare" else 0.55
            for i, run_name in enumerate(run_order):
                row = fp_summary[(fp_summary["run"] == run_name) & (fp_summary["metric"] == fp_metric_name)]
                if row.empty:
                    continue
                r = row.iloc[0]
                est = float(r["estimate"])
                lo = r["ci_low"]
                hi = r["ci_high"]
                if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                    yerr = np.array([[0.0], [0.0]])
                else:
                    yerr = np.array(
                        [[max(0.0, est - float(lo))], [max(0.0, float(hi) - est)]],
                        dtype=float,
                    )

                if mode == "compare":
                    x = fp_center - 0.375 + (i + 0.5) * width
                    color = (compare_palette or {}).get(run_name, RUN_COLORS[i % len(RUN_COLORS)])
                else:
                    x = fp_center
                    color = RED_MAIN

                ax2.bar(
                    [x],
                    [est],
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=2,
                )
                ax2.errorbar(
                    [x],
                    [est],
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=BAR_ERRORBAR_CAPSIZE,
                    zorder=3,
                )

    ax2.set_ylabel("false positive cluster count")
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("")
    ax2.set_ylim(0, 5)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.grid(False)

    x_ticks = list(np.arange(rate_count, dtype=float)) + [fp_center]
    x_labels = rate_metric_order + [fp_metric_name]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlim(-0.6, fp_center + 0.6)
    plt.setp(ax1.get_xticklabels(), rotation=18, ha="center", fontsize=12)

    fig.savefig(out_path, dpi=200)
    plt.close()


def _draw_subject_bar_panel(
    ax1: plt.Axes,
    ax2: plt.Axes,
    summary_distribution_df: pd.DataFrame,
    case_distribution_df: pd.DataFrame,
    mode: str,
    run_order: Sequence[str],
    single_color: str,
    compare_palette: Optional[Dict[str, str]],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    stripped: bool = False,
) -> None:
    rate_metrics = ["detection_rate", "pinpointing_rate", "false_positive_subject_rate", "specificity"]
    metric_name_map = {
        "detection_rate": "Detection rate",
        "pinpointing_rate": "Pinpointing rate",
        "false_positive_subject_rate": "False positive subject rate",
        "specificity": "Specificity",
    }

    rate_dist_df = summary_distribution_df.copy()
    rate_dist_df = rate_dist_df[rate_dist_df["metric"].isin(rate_metrics)]
    rate_dist_df["metric"] = rate_dist_df["metric"].map(metric_name_map)
    rate_metric_order = [
        metric_name_map[m]
        for m in rate_metrics
        if metric_name_map[m] in set(rate_dist_df["metric"].unique())
    ]

    _set_axes_border_width(ax1)
    _set_axes_border_width(ax2)

    if not rate_dist_df.empty:
        rate_summary = _summarize_for_bars(
            melted_df=rate_dist_df,
            metric_order=rate_metric_order,
            run_order=run_order,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        _plot_grouped_bars_with_ci(
            ax=ax1,
            summary_df=rate_summary,
            metric_order=rate_metric_order,
            run_order=run_order,
            mode=mode,
            single_color=single_color,
            compare_palette=compare_palette,
            ylabel="rate",
            ylim=(0, 1),
            title=None,
        )

    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis="both", length=0)

    rate_count = len(rate_metric_order)
    fp_gap = PLOT_CATEGORY_SPACING
    if rate_count > 0:
        fp_center = float((rate_count - 1) + fp_gap)
    else:
        fp_center = 0.0

    fp_metric_name = "Number of FP clusters"
    if not case_distribution_df.empty:
        fp_cols = ["n_fp_det_clusters"]
        fp_data = case_distribution_df[["run"] + fp_cols].dropna()
        if not fp_data.empty:
            fp_melted = fp_data.melt(
                id_vars=["run"],
                value_vars=fp_cols,
                var_name="metric",
                value_name="value",
            )
            fp_melted["metric"] = fp_metric_name
            fp_summary = _summarize_for_bars(
                melted_df=fp_melted,
                metric_order=[fp_metric_name],
                run_order=run_order,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed + 1,
            )

            n_runs = max(1, len(run_order))
            width = 0.75 / n_runs if mode == "compare" else 0.55
            for i, run_name in enumerate(run_order):
                row = fp_summary[(fp_summary["run"] == run_name) & (fp_summary["metric"] == fp_metric_name)]
                if row.empty:
                    continue
                r = row.iloc[0]
                est = float(r["estimate"])
                lo = r["ci_low"]
                hi = r["ci_high"]
                if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                    yerr = np.array([[0.0], [0.0]])
                else:
                    yerr = np.array(
                        [[max(0.0, est - float(lo))], [max(0.0, float(hi) - est)]],
                        dtype=float,
                    )

                if mode == "compare":
                    x = fp_center - 0.375 + (i + 0.5) * width
                    color = (compare_palette or {}).get(run_name, RUN_COLORS[i % len(RUN_COLORS)])
                else:
                    x = fp_center
                    color = RED_MAIN

                ax2.bar(
                    [x],
                    [est],
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=2,
                )
                ax2.errorbar(
                    [x],
                    [est],
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=BAR_ERRORBAR_CAPSIZE,
                    zorder=3,
                )

    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(0, 5)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.grid(False)

    x_ticks = list(np.arange(rate_count, dtype=float)) + [fp_center]
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(-0.6, fp_center + 0.6)

    if stripped:
        _strip_axis_for_grid(ax1)
        _strip_axis_for_grid(ax2)


def _draw_paired_differences_panel(
    ax: plt.Axes,
    joined_df: pd.DataFrame,
    metrics: Sequence[str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    secondary_metrics: Optional[Sequence[str]] = None,
    primary_ylim: Optional[Tuple[float, float]] = None,
    secondary_ylim: Optional[Tuple[float, float]] = None,
    stripped: bool = False,
) -> None:
    plot_rows: List[Dict[str, Any]] = []
    for idx, metric in enumerate(metrics):
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in joined_df.columns or col_b not in joined_df.columns:
            continue
        mdf = joined_df[["subject_id", col_a, col_b]].dropna().copy()
        if mdf.empty:
            continue
        diffs = (mdf[col_b] - mdf[col_a]).to_numpy(dtype=float)
        mean_diff = float(np.mean(diffs))
        ci = _bootstrap_ci(
            diffs,
            statistic="mean",
            n_boot=n_bootstrap,
            ci_level=ci_level,
            seed=seed + idx,
        )
        for d in diffs:
            plot_rows.append(
                {
                    "metric": metric,
                    "difference": float(d),
                    "mean_diff": mean_diff,
                    "ci_low": (ci or [None, None])[0],
                    "ci_high": (ci or [None, None])[1],
                }
            )

    secondary_set = set(secondary_metrics or [])
    use_secondary_axis = any(m in secondary_set for m in metrics)
    ax2 = ax.twinx() if use_secondary_axis else None
    _set_axes_border_width(ax)
    if ax2 is not None:
        _set_axes_border_width(ax2)

    if not plot_rows:
        if primary_ylim is not None:
            ax.set_ylim(*primary_ylim)
        if ax2 is not None and secondary_ylim is not None:
            ax2.set_ylim(*secondary_ylim)
        if stripped:
            _strip_axis_for_grid(ax)
            if ax2 is not None:
                _strip_axis_for_grid(ax2)
        return

    pdf = pd.DataFrame(plot_rows)
    metric_order = [m for m in metrics if m in set(pdf["metric"].unique())]
    primary_metrics = [m for m in metric_order if m not in secondary_set]
    secondary_metrics_ordered = [m for m in metric_order if m in secondary_set]

    x_coord_map: Dict[str, float] = {}
    for i, metric in enumerate(primary_metrics):
        x_coord_map[metric] = float(i)
    if secondary_metrics_ordered:
        if primary_metrics:
            secondary_start = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING)
            separator_x = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING / 2.0)
        else:
            secondary_start = 0.0
            separator_x = None
        for i, metric in enumerate(secondary_metrics_ordered):
            x_coord_map[metric] = float(secondary_start + i)
    else:
        separator_x = None

    rng = np.random.default_rng(seed)
    for metric in metric_order:
        mdf = pdf[pdf["metric"] == metric]
        x0 = x_coord_map[metric]
        target_ax = ax2 if (ax2 is not None and metric in secondary_set) else ax
        jitter = rng.uniform(-0.12 * PLOT_CATEGORY_SPACING, 0.12 * PLOT_CATEGORY_SPACING, size=len(mdf))
        target_ax.scatter(
            np.full(len(mdf), x0, dtype=float) + jitter,
            mdf["difference"].to_numpy(dtype=float),
            color=RUN_COLORS[3],
            alpha=0.55,
            s=20,
            zorder=2,
        )
        first = mdf.iloc[0]
        mean_diff = float(first["mean_diff"])
        ci_low = first["ci_low"]
        ci_high = first["ci_high"]
        if ci_low is None or ci_high is None:
            yerr = np.array([[0.0], [0.0]])
        else:
            yerr = np.array([[max(0.0, mean_diff - float(ci_low))], [max(0.0, float(ci_high) - mean_diff)]])
        target_ax.errorbar(
            [x0],
            [mean_diff],
            yerr=yerr,
            fmt="D",
            color=RUN_COLORS[2],
            ecolor=RUN_COLORS[2],
            capsize=DIFF_ERRORBAR_CAPSIZE,
            markersize=7,
            zorder=4,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if ax2 is not None:
        ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    if separator_x is not None:
        ax.axvline(
            x=separator_x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

    x_coords = np.array([x_coord_map[m] for m in metric_order], dtype=float)
    ax.set_xticks(x_coords)
    if len(x_coords) > 0:
        ax.set_xlim(float(np.min(x_coords)) - 0.6, float(np.max(x_coords)) + 0.6)
    if primary_ylim is not None:
        ax.set_ylim(*primary_ylim)
    if ax2 is not None and secondary_ylim is not None:
        ax2.set_ylim(*secondary_ylim)

    if stripped:
        _strip_axis_for_grid(ax)
        if ax2 is not None:
            _strip_axis_for_grid(ax2)


def _plot_variable_curves(
    runs: Sequence[RunEvaluation],
    out_path: Path,
    mode: str,
    run_palette: Dict[str, str],
) -> None:
    def _annotate_best_f1(ax: plt.Axes, df: pd.DataFrame) -> None:
        if df.empty:
            return

        xy = df[["x", "y"]].dropna()
        if xy.empty:
            return

        denom = xy["x"] + xy["y"]
        valid = denom > 0
        if not valid.any():
            return

        f1_vals = pd.Series(np.nan, index=xy.index, dtype=float)
        f1_vals.loc[valid] = 2.0 * xy.loc[valid, "x"] * xy.loc[valid, "y"] / denom.loc[valid]
        best_idx = f1_vals.idxmax()
        best_x = float(xy.loc[best_idx, "x"])
        best_y = float(xy.loc[best_idx, "y"])
        best_f1 = float(f1_vals.loc[best_idx])
        best_thr = _as_float(df.loc[best_idx, "threshold"]) if "threshold" in df.columns else None

        ax.axvline(best_x, color=RED_TRUE, linestyle="--", linewidth=1.0, alpha=0.25)
        ax.axhline(best_y, color=RED_TRUE, linestyle="--", linewidth=1.0, alpha=0.25)
        ax.scatter([best_x], [best_y], s=52, color=RED_TRUE, edgecolor="white", linewidth=0.8, zorder=4)

        x_annot = min(0.98, best_x + 0.03)
        y_annot = min(0.98, best_y + 0.03)
        label_text = (
            f"threshold={best_thr:.3f}, F1={best_f1:.3f}"
            if best_thr is not None
            else f"threshold=NA, F1={best_f1:.3f}"
        )
        ax.text(
            x_annot,
            y_annot,
            label_text,
            fontsize=9,
            color=RED_TRUE,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": RED_TRUE, "alpha": 0.65},
        )

    voxel_frames = [
        _extract_curve_df(run, "voxel_pr", "recall", "precision") for run in runs
    ]
    det_frames = [
        _extract_curve_df(run, "cluster_detection_pr", "recall", "precision")
        for run in runs
    ]
    pin_frames = [
        _extract_curve_df(run, "cluster_pinpoint_pr", "recall", "precision")
        for run in runs
    ]
    fp_burden_frames = [
        _extract_curve_df(
            run,
            "detection_vs_fp_cluster_burden",
            "mean_fp_clusters_per_subject",
            "subject_detection_rate",
        )
        for run in runs
    ]

    voxel_df = pd.concat(voxel_frames, ignore_index=True) if voxel_frames else pd.DataFrame()
    det_df = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    pin_df = pd.concat(pin_frames, ignore_index=True) if pin_frames else pd.DataFrame()
    fp_burden_df = (
        pd.concat(fp_burden_frames, ignore_index=True)
        if fp_burden_frames
        else pd.DataFrame()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [
        (axes[0, 0], voxel_df, "Voxel-level PR", "recall", "precision", BLUE_DARK, True),
        (axes[0, 1], det_df, "Cluster detection PR", "recall", "precision", RED_MAIN, True),
        (axes[1, 0], pin_df, "Cluster pinpoint PR", "recall", "precision", RED_LIGHT, True),
        (
            axes[1, 1],
            fp_burden_df,
            "Detection vs mean FP clusters/patient",
            "mean_fp_clusters_per_subject",
            "detection_rate",
            BLUE_MAIN,
            False,
        ),
    ]

    for ax, df, title, xlabel, ylabel, single_color, annotate_f1 in panels:
        if df.empty:
            ax.set_title(f"{title} (no data)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if title != "Detection vs mean FP clusters/patient":
                ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            continue

        if "threshold" in df.columns:
            df = df.sort_values(by=["run", "threshold"], kind="stable")

        if mode == "compare":
            sns.lineplot(
                data=df,
                x="x",
                y="y",
                hue="run",
                marker="o",
                palette=run_palette,
                estimator=None,
                errorbar=None,
                sort=False,
                ax=ax,
            )
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="run",
                palette=run_palette,
                legend=False,
                s=36,
                ax=ax,
                zorder=3,
            )
            legend = ax.get_legend()
            if legend is not None and title != "Detection vs mean FP clusters/patient":
                legend.remove()
        else:
            sns.lineplot(
                data=df,
                x="x",
                y="y",
                marker="o",
                color=single_color,
                estimator=None,
                errorbar=None,
                sort=False,
                ax=ax,
            )
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                color=single_color,
                s=36,
                legend=False,
                ax=ax,
                zorder=3,
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        if title == "Detection vs mean FP clusters/patient":
            ax.set_xlim(0, 10)
        else:
            ax.set_xlim(0, 1)
            ax.set_aspect("equal", adjustable="box")

        if annotate_f1:
            _annotate_best_f1(ax, df)

        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_combined_boxplot_summary(
    case_df: pd.DataFrame,
    subject_summary_distribution: pd.DataFrame,
    mode: str,
    out_path: Path,
    distribution_unit_label: str,
    hue_order: Optional[List[str]] = None,
    run_palette: Optional[Dict[str, str]] = None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 12345,
) -> None:
    if case_df.empty and subject_summary_distribution.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.ravel()
    legend_handles = None
    legend_labels = None
    run_order = hue_order or sorted(
        pd.concat([case_df[["run"]], subject_summary_distribution[["run"]]], ignore_index=True)
        ["run"]
        .dropna()
        .unique()
        .tolist()
    )

    def _draw_panel(
        ax: plt.Axes,
        source_df: pd.DataFrame,
        metric_cols: Sequence[str],
        metric_name_map: Dict[str, str],
        title: str,
        ylabel: str,
        single_color: str,
        y_limits: Optional[tuple[float, float]] = None,
        bracket_side: str = "auto",
        show_fliers: bool = False,
    ) -> None:
        nonlocal legend_handles, legend_labels
        if source_df.empty:
            ax.set_title(f"{title} (no data)")
            ax.grid(alpha=0.25)
            return

        needed_cols = [c for c in metric_cols if c in source_df.columns]
        if not needed_cols:
            ax.set_title(f"{title} (no data)")
            ax.grid(alpha=0.25)
            return

        melted = source_df[needed_cols + ["run"]].melt(
            id_vars=["run"],
            value_vars=needed_cols,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])

        if melted.empty:
            ax.set_title(f"{title} (no data)")
            ax.grid(alpha=0.25)
            return

        melted["metric"] = melted["metric"].map(metric_name_map)
        metric_order_plot = [
            metric_name_map[c]
            for c in needed_cols
            if metric_name_map[c] in set(melted["metric"].unique())
        ]

        summary_df = _summarize_for_bars(
            melted_df=melted,
            metric_order=metric_order_plot,
            run_order=run_order,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        _plot_grouped_bars_with_ci(
            ax=ax,
            summary_df=summary_df,
            metric_order=metric_order_plot,
            run_order=run_order,
            mode=mode,
            single_color=single_color,
            compare_palette=run_palette,
            ylabel=ylabel,
            ylim=y_limits,
            title=title,
        )
        if mode == "compare":
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        ax.grid(axis="x", alpha=0.25)

    _draw_panel(
        axes[0],
        case_df,
        metric_cols=["voxel_precision", "voxel_recall", "voxel_dice"],
        metric_name_map={
            "voxel_dice": "Dice",
            "voxel_precision": "Precision",
            "voxel_recall": "Recall",
        },
        title="Voxel-level",
        ylabel="score",
        single_color=BLUE_MAIN,
        y_limits=(0, 1),
        bracket_side="auto",
    )

    _draw_panel(
        axes[1],
        case_df,
        metric_cols=["cluster_det_precision", "cluster_det_recall", "cluster_det_f1"],
        metric_name_map={
            "cluster_det_precision": "Precision",
            "cluster_det_recall": "Recall",
            "cluster_det_f1": "F1",
        },
        title="Cluster-level Detection",
        ylabel="score",
        single_color=RED_MAIN,
        y_limits=(0, 1),
        bracket_side="below",
    )

    _draw_panel(
        axes[2],
        case_df,
        metric_cols=["cluster_pin_precision", "cluster_pin_recall", "cluster_pin_f1"],
        metric_name_map={
            "cluster_pin_precision": "Precision",
            "cluster_pin_recall": "Recall",
            "cluster_pin_f1": "F1",
        },
        title="Cluster-level Pinpointing",
        ylabel="score",
        single_color=RED_LIGHT,
        y_limits=(0, 1),
        bracket_side="below",
    )

    subject_rate_df = subject_summary_distribution.copy()
    _draw_panel(
        axes[3],
        subject_rate_df,
        metric_cols=["detection_rate", "pinpointing_rate", "false_positive_subject_rate", "specificity"],
        metric_name_map={
            "detection_rate": "Detection rate",
            "pinpointing_rate": "Pinpointing rate",
            "false_positive_subject_rate": "False positive subject rate",
            "specificity": "Specificity",
        },
        title=f"Subject-level Rates (over {distribution_unit_label}s)",
        ylabel="rate",
        single_color=BLUE_MAIN,
        y_limits=(0, 1),
        bracket_side="auto",
    )

    _draw_panel(
        axes[4],
        case_df,
        metric_cols=["n_fp_det_clusters"],
        metric_name_map={
            "n_fp_det_clusters": "Number of FP clusters",
        },
        title=f"Subject-level FP Clusters (over {distribution_unit_label}s)",
        ylabel="count",
        single_color=RED_MAIN,
        y_limits=(0, 5),
        bracket_side="auto",
        show_fliers=True,
    )

    axes[5].axis("off")

    if mode == "compare" and legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="run",
            loc="upper center",
            bbox_to_anchor=(0.5, COMBINED_LEGEND_Y_ANCHOR),
            ncol=max(1, min(4, len(legend_labels))),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_paired_difference_metric(
    baseline_label: str,
    comparator_label: str,
    metric: str,
    joined_df: pd.DataFrame,
    out_path: Path,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> None:
    col_a = f"{metric}_a"
    col_b = f"{metric}_b"
    if col_a not in joined_df.columns or col_b not in joined_df.columns:
        return
    plot_df = joined_df[["subject_id", col_a, col_b]].dropna().copy()
    if plot_df.empty:
        return
    plot_df["difference"] = plot_df[col_b] - plot_df[col_a]
    diffs = plot_df["difference"].to_numpy(dtype=float)
    mean_diff = float(np.mean(diffs))
    ci = _bootstrap_ci(diffs, "mean", n_bootstrap, ci_level, seed)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_position(PAIRED_DIFF_AXES_RECT)
    _set_axes_border_width(ax)
    jitter = np.random.default_rng(seed).uniform(-0.06, 0.06, size=len(plot_df))
    ax.scatter(np.zeros(len(plot_df)) + jitter, diffs, color=BLUE_MAIN, alpha=0.7, s=25)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.errorbar(
        [0],
        [mean_diff],
        yerr=[[mean_diff - (ci or [mean_diff, mean_diff])[0]], [(ci or [mean_diff, mean_diff])[1] - mean_diff]],
        fmt="D",
        color=RED_TRUE,
        ecolor=RED_TRUE,
        capsize=DIFF_ERRORBAR_CAPSIZE,
        markersize=7,
        label="Mean difference with bootstrap CI",
    )
    ax.set_xlim(-0.3, 0.3)
    ax.set_xticks([0])
    ax.set_xticklabels([metric])
    ax.set_ylabel(f"Paired difference ({comparator_label} - {baseline_label})")
    ax.set_title(f"Paired difference: {metric} (n={len(plot_df)})")
    ax.tick_params(axis="both", length=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_paired_differences_combined(
    baseline_label: str,
    comparator_label: str,
    metrics: Sequence[str],
    joined_df: pd.DataFrame,
    out_path: Path,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    n_permutations: int = 10000,
    secondary_metrics: Optional[Sequence[str]] = None,
    primary_ylim: Optional[Tuple[float, float]] = None,
    secondary_ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    plot_rows: List[Dict[str, Any]] = []
    for idx, metric in enumerate(metrics):
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in joined_df.columns or col_b not in joined_df.columns:
            continue
        mdf = joined_df[["subject_id", col_a, col_b]].dropna().copy()
        if mdf.empty:
            continue
        diffs = (mdf[col_b] - mdf[col_a]).to_numpy(dtype=float)
        mean_diff = float(np.mean(diffs))
        ci = _bootstrap_ci(
            diffs,
            statistic="mean",
            n_boot=n_bootstrap,
            ci_level=ci_level,
            seed=seed + idx,
        )
        for d in diffs:
            plot_rows.append(
                {
                    "metric": metric,
                    "difference": float(d),
                    "mean_diff": mean_diff,
                    "ci_low": (ci or [None, None])[0],
                    "ci_high": (ci or [None, None])[1],
                    "n": int(len(diffs)),
                }
            )

    if not plot_rows:
        return

    pdf = pd.DataFrame(plot_rows)
    metric_order = [m for m in metrics if m in set(pdf["metric"].unique())]
    x_pos = {m: i for i, m in enumerate(metric_order)}
    secondary_set = set(secondary_metrics or [])
    use_secondary_axis = any(m in secondary_set for m in metric_order)

    # Keep the actual graph box size consistent across paired plots.
    # Subject-level plots (with secondary axis) are allowed to be slightly wider.
    if use_secondary_axis:
        fig_w = 6.8
        axes_rect = PAIRED_DIFF_AXES_RECT_SECONDARY
    else:
        fig_w = 6.4
        axes_rect = PAIRED_DIFF_AXES_RECT

    primary_metrics = [m for m in metric_order if m not in secondary_set]
    secondary_metrics_ordered = [m for m in metric_order if m in secondary_set]
    x_coord_map: Dict[str, float] = {}
    for i, metric in enumerate(primary_metrics):
        x_coord_map[metric] = float(i)
    if secondary_metrics_ordered:
        if primary_metrics:
            secondary_start = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING)
            separator_x = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING / 2.0)
        else:
            secondary_start = 0.0
            separator_x = None
        for i, metric in enumerate(secondary_metrics_ordered):
            x_coord_map[metric] = float(secondary_start + i)
    else:
        separator_x = None
    x_coords = np.array([x_coord_map[m] for m in metric_order], dtype=float)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    ax2 = ax.twinx() if use_secondary_axis else None
    ax.set_position(axes_rect)
    _set_axes_border_width(ax)
    if ax2 is not None:
        ax2.set_position(axes_rect)
        _set_axes_border_width(ax2)
    marker_color = RUN_COLORS[2]
    rng = np.random.default_rng(seed)
    sig_annotations: List[Dict[str, Any]] = []
    for metric in metric_order:
        mdf = pdf[pdf["metric"] == metric]
        x0 = x_coord_map[metric]
        target_ax = ax2 if (ax2 is not None and metric in secondary_set) else ax
        jitter = rng.uniform(-0.12 * PLOT_CATEGORY_SPACING, 0.12 * PLOT_CATEGORY_SPACING, size=len(mdf))
        target_ax.scatter(
            np.full(len(mdf), x0, dtype=float) + jitter,
            mdf["difference"].to_numpy(dtype=float),
            color=RUN_COLORS[3],
            alpha=0.55,
            s=20,
            zorder=2,
        )
        first = mdf.iloc[0]
        mean_diff = float(first["mean_diff"])
        ci_low = first["ci_low"]
        ci_high = first["ci_high"]
        diffs = mdf["difference"].to_numpy(dtype=float)
        p_val = _sign_flip_permutation_pvalue(
            differences=diffs,
            n_permutations=n_permutations,
            seed=seed + 1000 + x_pos[metric],
        )
        if p_val is not None and p_val < 0.05:
            stars = _p_to_stars(float(p_val))
            if stars != "ns":
                sig_annotations.append(
                    {
                        "ax": target_ax,
                        "x": x0,
                        "stars": stars,
                        "y_base": float(ci_high) if ci_high is not None else mean_diff,
                    }
                )
        if ci_low is None or ci_high is None:
            yerr = np.array([[0.0], [0.0]])
        else:
            yerr = np.array([[max(0.0, mean_diff - float(ci_low))], [max(0.0, float(ci_high) - mean_diff)]])
        target_ax.errorbar(
            [x0],
            [mean_diff],
            yerr=yerr,
            fmt="D",
            color=marker_color,
            ecolor=marker_color,
            capsize=DIFF_ERRORBAR_CAPSIZE,
            markersize=7,
            zorder=4,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if ax2 is not None:
        ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    if separator_x is not None:
        ax.axvline(
            x=separator_x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
    ax.set_xticks(x_coords)
    ax.set_xticklabels(metric_order, rotation=25, ha="right")
    if len(x_coords) > 0:
        ax.set_xlim(float(np.min(x_coords)) - 0.6, float(np.max(x_coords)) + 0.6)
    ax.set_ylabel(f"Paired difference ({comparator_label} - {baseline_label})")
    if ax2 is not None:
        ax2.set_ylabel(f"Paired difference (secondary axis) ({comparator_label} - {baseline_label})")
    if primary_ylim is not None:
        ax.set_ylim(*primary_ylim)
    if ax2 is not None and secondary_ylim is not None:
        ax2.set_ylim(*secondary_ylim)
    ax.tick_params(axis="both", length=0)
    if ax2 is not None:
        ax2.tick_params(axis="y", length=0)

    # Draw compact significance brackets with stars above significant markers.
    for ann in sig_annotations:
        axis = ann["ax"]
        x0 = float(ann["x"])
        y_base = float(ann["y_base"])
        y_min, y_max = axis.get_ylim()
        y_range = max(1e-6, y_max - y_min)
        h = 0.02 * y_range
        y = min(y_max - 1.8 * h, y_base + 0.06 * y_range)
        w = 0.15
        axis.plot(
            [x0 - w, x0 - w, x0 + w, x0 + w],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.0,
            zorder=5,
        )
        axis.text(
            x0,
            y + h + 0.005 * y_range,
            str(ann["stars"]),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            zorder=6,
        )

    ax.set_title(title or "Paired differences across metrics (mean with bootstrap CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_compact_grouped_bars_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    metric_order: Sequence[str],
    run_order: Sequence[str],
    mode: str,
    single_color: str,
    compare_palette: Optional[Dict[str, str]],
    ylim: Optional[Tuple[float, float]] = None,
    y_tick_side: Optional[str] = None,
) -> None:
    if summary_df.empty or not metric_order:
        _set_axes_border_width(ax)
        _strip_axes_decorations(ax, keep_y_ticks=False)
        return

    x = np.arange(len(metric_order), dtype=float)
    n_runs = max(1, len(run_order))
    width = 0.75 / n_runs if mode == "compare" else 0.55

    for i, run_name in enumerate(run_order):
        run_df = summary_df[summary_df["run"] == run_name]
        if run_df.empty:
            continue

        est = []
        yerr_low = []
        yerr_high = []
        for metric_name in metric_order:
            row = run_df[run_df["metric"] == metric_name]
            if row.empty:
                est.append(np.nan)
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
                continue
            r = row.iloc[0]
            e = float(r["estimate"])
            lo = r["ci_low"]
            hi = r["ci_high"]
            est.append(e)
            if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
            else:
                yerr_low.append(max(0.0, e - float(lo)))
                yerr_high.append(max(0.0, float(hi) - e))

        if mode == "compare":
            offsets = x - 0.375 + (i + 0.5) * width
            color = (compare_palette or {}).get(run_name, RUN_COLORS[i % len(RUN_COLORS)])
        else:
            offsets = x
            color = single_color

        ax.bar(
            offsets,
            est,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        ax.errorbar(
            offsets,
            est,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=BAR_ERRORBAR_CAPSIZE,
            zorder=3,
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(x) > 0:
        ax.set_xlim(float(np.min(x)) - 0.6, float(np.max(x)) + 0.6)
    _set_compact_y_ticks(ax, ylim=ylim)
    _set_axes_border_width(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    _strip_axes_decorations(ax, keep_y_ticks=True, y_tick_side=y_tick_side)


def _plot_compact_subject_bars_panel(
    ax1: plt.Axes,
    ax2: plt.Axes,
    summary_distribution_df: pd.DataFrame,
    case_distribution_df: pd.DataFrame,
    mode: str,
    run_order: Sequence[str],
    single_color: str,
    compare_palette: Optional[Dict[str, str]],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> None:
    rate_metrics = ["detection_rate", "pinpointing_rate", "false_positive_subject_rate", "specificity"]
    metric_name_map = {
        "detection_rate": "Detection rate",
        "pinpointing_rate": "Pinpointing rate",
        "false_positive_subject_rate": "False positive subject rate",
        "specificity": "Specificity",
    }

    rate_dist_df = summary_distribution_df.copy()
    rate_dist_df = rate_dist_df[rate_dist_df["metric"].isin(rate_metrics)]
    rate_dist_df["metric"] = rate_dist_df["metric"].map(metric_name_map)
    rate_metric_order = [
        metric_name_map[m]
        for m in rate_metrics
        if metric_name_map[m] in set(rate_dist_df["metric"].unique())
    ]

    if not rate_dist_df.empty:
        rate_summary = _summarize_for_bars(
            melted_df=rate_dist_df,
            metric_order=rate_metric_order,
            run_order=run_order,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        _plot_compact_grouped_bars_panel(
            ax=ax1,
            summary_df=rate_summary,
            metric_order=rate_metric_order,
            run_order=run_order,
            mode=mode,
            single_color=single_color,
            compare_palette=compare_palette,
            ylim=(0, 1),
        )
    else:
        ax1.set_ylim(0, 1)

    rate_count = len(rate_metric_order)
    fp_gap = PLOT_CATEGORY_SPACING
    fp_center = float((rate_count - 1) + fp_gap) if rate_count > 0 else 0.0
    fp_metric_name = "Number of FP clusters"
    separator_x = float((rate_count - 1) + fp_gap / 2.0) if rate_count > 0 else None

    if not case_distribution_df.empty:
        fp_data = case_distribution_df[["run", "n_fp_det_clusters"]].dropna()
        if not fp_data.empty:
            fp_melted = fp_data.melt(
                id_vars=["run"],
                value_vars=["n_fp_det_clusters"],
                var_name="metric",
                value_name="value",
            )
            fp_melted["metric"] = fp_metric_name
            fp_summary = _summarize_for_bars(
                melted_df=fp_melted,
                metric_order=[fp_metric_name],
                run_order=run_order,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed + 1,
            )
            n_runs = max(1, len(run_order))
            width = 0.75 / n_runs if mode == "compare" else 0.55

            for i, run_name in enumerate(run_order):
                row = fp_summary[(fp_summary["run"] == run_name) & (fp_summary["metric"] == fp_metric_name)]
                if row.empty:
                    continue
                r = row.iloc[0]
                est = float(r["estimate"])
                lo = r["ci_low"]
                hi = r["ci_high"]
                if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                    yerr = np.array([[0.0], [0.0]])
                else:
                    yerr = np.array(
                        [[max(0.0, est - float(lo))], [max(0.0, float(hi) - est)]],
                        dtype=float,
                    )

                if mode == "compare":
                    xpos = fp_center - 0.375 + (i + 0.5) * width
                    color = (compare_palette or {}).get(run_name, RUN_COLORS[i % len(RUN_COLORS)])
                else:
                    xpos = fp_center
                    color = single_color

                ax2.bar(
                    [xpos],
                    [est],
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=2,
                )
                ax2.errorbar(
                    [xpos],
                    [est],
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=BAR_ERRORBAR_CAPSIZE,
                    zorder=3,
                )

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 5)
    ax1.set_xlim(-0.6, fp_center + 0.6)
    ax2.set_xlim(ax1.get_xlim())
    ax2.patch.set_alpha(0.0)
    _set_compact_y_ticks(ax1, ylim=(0, 1))
    _set_compact_y_ticks(ax2, ylim=(0, 5))
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.grid(False)
    if separator_x is not None:
        ax1.axvline(
            x=separator_x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
    _set_axes_border_width(ax1)
    _set_axes_border_width(ax2)
    _strip_axes_decorations(ax1, keep_y_ticks=True, y_tick_side=None)
    _strip_axes_decorations(ax2, keep_y_ticks=True, y_tick_side="right")


def _plot_compact_diff_panel(
    ax: plt.Axes,
    joined_df: pd.DataFrame,
    metrics: Sequence[str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    n_permutations: int = 10000,
    secondary_metrics: Optional[Sequence[str]] = None,
    primary_ylim: Optional[Tuple[float, float]] = None,
    secondary_ylim: Optional[Tuple[float, float]] = None,
    primary_y_tick_side: Optional[str] = None,
    secondary_y_tick_side: Optional[str] = None,
) -> None:
    plot_rows: List[Dict[str, Any]] = []
    for idx, metric in enumerate(metrics):
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in joined_df.columns or col_b not in joined_df.columns:
            continue
        mdf = joined_df[["subject_id", col_a, col_b]].dropna().copy()
        if mdf.empty:
            continue
        diffs = (mdf[col_b] - mdf[col_a]).to_numpy(dtype=float)
        mean_diff = float(np.mean(diffs))
        ci = _bootstrap_ci(
            diffs,
            statistic="mean",
            n_boot=n_bootstrap,
            ci_level=ci_level,
            seed=seed + idx,
        )
        for d in diffs:
            plot_rows.append(
                {
                    "metric": metric,
                    "difference": float(d),
                    "mean_diff": mean_diff,
                    "ci_low": (ci or [None, None])[0],
                    "ci_high": (ci or [None, None])[1],
                }
            )

    if not plot_rows:
        _set_axes_border_width(ax)
        _strip_axes_decorations(ax, keep_y_ticks=False)
        return

    pdf = pd.DataFrame(plot_rows)
    metric_order = [m for m in metrics if m in set(pdf["metric"].unique())]
    secondary_set = set(secondary_metrics or [])
    use_secondary_axis = any(m in secondary_set for m in metric_order)
    ax2 = ax.twinx() if use_secondary_axis else None
    if ax2 is not None:
        ax2.patch.set_alpha(0.0)

    primary_metrics = [m for m in metric_order if m not in secondary_set]
    secondary_metrics_ordered = [m for m in metric_order if m in secondary_set]
    x_coord_map: Dict[str, float] = {}
    for i, metric in enumerate(primary_metrics):
        x_coord_map[metric] = float(i)
    if secondary_metrics_ordered:
        secondary_start = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING) if primary_metrics else 0.0
        separator_x = float((len(primary_metrics) - 1) + PLOT_CATEGORY_SPACING / 2.0) if primary_metrics else None
        for i, metric in enumerate(secondary_metrics_ordered):
            x_coord_map[metric] = float(secondary_start + i)
    else:
        separator_x = None

    x_coords = np.array([x_coord_map[m] for m in metric_order], dtype=float)
    rng = np.random.default_rng(seed)
    metric_rank = {m: i for i, m in enumerate(metric_order)}
    sig_annotations: List[Dict[str, Any]] = []
    for metric in metric_order:
        mdf = pdf[pdf["metric"] == metric]
        x0 = x_coord_map[metric]
        target_ax = ax2 if (ax2 is not None and metric in secondary_set) else ax
        jitter = rng.uniform(-0.12 * PLOT_CATEGORY_SPACING, 0.12 * PLOT_CATEGORY_SPACING, size=len(mdf))
        target_ax.scatter(
            np.full(len(mdf), x0, dtype=float) + jitter,
            mdf["difference"].to_numpy(dtype=float),
            color=RUN_COLORS[3],
            alpha=0.55,
            s=20,
            zorder=2,
        )
        first = mdf.iloc[0]
        mean_diff = float(first["mean_diff"])
        ci_low = first["ci_low"]
        ci_high = first["ci_high"]
        diffs = mdf["difference"].to_numpy(dtype=float)
        p_val = _sign_flip_permutation_pvalue(
            differences=diffs,
            n_permutations=n_permutations,
            seed=seed + 1000 + metric_rank[metric],
        )
        if p_val is not None and p_val < 0.05:
            stars = _p_to_stars(float(p_val))
            if stars != "ns":
                sig_annotations.append(
                    {
                        "ax": target_ax,
                        "x": x0,
                        "stars": stars,
                        "y_base": float(ci_high) if ci_high is not None else mean_diff,
                    }
                )
        if ci_low is None or ci_high is None:
            yerr = np.array([[0.0], [0.0]])
        else:
            yerr = np.array([[max(0.0, mean_diff - float(ci_low))], [max(0.0, float(ci_high) - mean_diff)]])
        target_ax.errorbar(
            [x0],
            [mean_diff],
            yerr=yerr,
            fmt="D",
            color=RUN_COLORS[2],
            ecolor=RUN_COLORS[2],
            capsize=DIFF_ERRORBAR_CAPSIZE,
            markersize=7,
            zorder=4,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if ax2 is not None:
        ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    if separator_x is not None:
        ax.axvline(
            x=separator_x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
    if len(x_coords) > 0:
        ax.set_xlim(float(np.min(x_coords)) - 0.6, float(np.max(x_coords)) + 0.6)
        if ax2 is not None:
            ax2.set_xlim(ax.get_xlim())
    if primary_ylim is not None:
        ax.set_ylim(*primary_ylim)
    if ax2 is not None and secondary_ylim is not None:
        ax2.set_ylim(*secondary_ylim)

    for ann in sig_annotations:
        axis = ann["ax"]
        x0 = float(ann["x"])
        y_base = float(ann["y_base"])
        y_min, y_max = axis.get_ylim()
        y_range = max(1e-6, y_max - y_min)
        h = 0.02 * y_range
        y = min(y_max - 1.8 * h, y_base + 0.06 * y_range)
        w = 0.15
        axis.plot(
            [x0 - w, x0 - w, x0 + w, x0 + w],
            [y, y + h, y + h, y],
            color="black",
            linewidth=1.0,
            zorder=5,
        )
        axis.text(
            x0,
            y + h + 0.005 * y_range,
            str(ann["stars"]),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            zorder=6,
        )

    _set_compact_y_ticks(ax, ylim=primary_ylim)
    if ax2 is not None and secondary_ylim is not None:
        _set_compact_y_ticks(ax2, ylim=secondary_ylim)
    _set_axes_border_width(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    _strip_axes_decorations(ax, keep_y_ticks=True, y_tick_side=primary_y_tick_side)
    if ax2 is not None:
        _set_axes_border_width(ax2)
        _strip_axes_decorations(ax2, keep_y_ticks=True, y_tick_side=secondary_y_tick_side)


def _plot_compact_comparison_grid(
    case_df: pd.DataFrame,
    subject_summary_distribution: pd.DataFrame,
    joined_df: pd.DataFrame,
    run_order: Sequence[str],
    out_path: Path,
    run_palette: Dict[str, str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    n_permutations: int = 10000,
) -> None:
    top_h = COMPACT_GRID_TOP_ROW_HEIGHT
    bottom_h = COMPACT_GRID_TOP_ROW_HEIGHT * COMPACT_GRID_BOTTOM_ROW_RATIO
    fig = plt.figure(figsize=(COMPACT_GRID_FIG_WIDTH, top_h + bottom_h))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, COMPACT_GRID_BOTTOM_ROW_RATIO],
        wspace=COMPACT_GRID_WSPACE,
        hspace=COMPACT_GRID_HSPACE,
        left=0.03,
        right=0.99,
        bottom=0.05,
        top=0.98,
    )

    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(4)]

    voxel_summary = _summarize_for_bars(
        melted_df=case_df[["run", "voxel_precision", "voxel_recall", "voxel_dice"]].melt(
            id_vars=["run"],
            value_vars=["voxel_precision", "voxel_recall", "voxel_dice"],
            var_name="metric",
            value_name="value",
        ).assign(
            metric=lambda d: d["metric"].map(
                {
                    "voxel_precision": "Precision",
                    "voxel_recall": "Recall",
                    "voxel_dice": "Dice",
                }
            )
        ).dropna(subset=["value"]),
        metric_order=["Precision", "Recall", "Dice"],
        run_order=run_order,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    _plot_compact_grouped_bars_panel(
        ax=axes_top[0],
        summary_df=voxel_summary,
        metric_order=["Precision", "Recall", "Dice"],
        run_order=run_order,
        mode="compare",
        single_color=BLUE_MAIN,
        compare_palette=run_palette,
        ylim=(0, 1),
        y_tick_side="left",
    )

    cluster_det_summary = _summarize_cluster_triplet_for_bars(
        wide_df=case_df,
        run_order=run_order,
        precision_col="cluster_det_precision",
        recall_col="cluster_det_recall",
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 1,
    )
    _plot_compact_grouped_bars_panel(
        ax=axes_top[1],
        summary_df=cluster_det_summary,
        metric_order=["Precision", "Recall", "F1"],
        run_order=run_order,
        mode="compare",
        single_color=RED_MAIN,
        compare_palette=run_palette,
        ylim=(0, 1),
        y_tick_side=None,
    )

    cluster_pin_summary = _summarize_cluster_triplet_for_bars(
        wide_df=case_df,
        run_order=run_order,
        precision_col="cluster_pin_precision",
        recall_col="cluster_pin_recall",
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 2,
    )
    _plot_compact_grouped_bars_panel(
        ax=axes_top[2],
        summary_df=cluster_pin_summary,
        metric_order=["Precision", "Recall", "F1"],
        run_order=run_order,
        mode="compare",
        single_color=RED_LIGHT,
        compare_palette=run_palette,
        ylim=(0, 1),
        y_tick_side=None,
    )

    ax_subject_top = axes_top[3]
    ax_subject_top_secondary = ax_subject_top.twinx()
    _plot_compact_subject_bars_panel(
        ax1=ax_subject_top,
        ax2=ax_subject_top_secondary,
        summary_distribution_df=subject_summary_distribution,
        case_distribution_df=case_df,
        mode="compare",
        run_order=run_order,
        single_color=BLUE_MAIN,
        compare_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 3,
    )

    _plot_compact_diff_panel(
        ax=axes_bottom[0],
        joined_df=joined_df,
        metrics=["voxel_precision", "voxel_recall", "voxel_dice"],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 10,
        n_permutations=n_permutations,
        primary_ylim=(-0.5, 0.5),
        primary_y_tick_side="left",
    )
    _plot_compact_diff_panel(
        ax=axes_bottom[1],
        joined_df=joined_df,
        metrics=["cluster_det_precision", "cluster_det_recall", "cluster_det_f1"],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 11,
        n_permutations=n_permutations,
        primary_ylim=(-0.5, 0.5),
        primary_y_tick_side=None,
    )
    _plot_compact_diff_panel(
        ax=axes_bottom[2],
        joined_df=joined_df,
        metrics=["cluster_pin_precision", "cluster_pin_recall", "cluster_pin_f1"],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 12,
        n_permutations=n_permutations,
        primary_ylim=(-0.5, 0.5),
        primary_y_tick_side=None,
    )
    _plot_compact_diff_panel(
        ax=axes_bottom[3],
        joined_df=joined_df,
        metrics=["subject_detected", "subject_pinpointed", "fp_subject_indicator_controls", "n_fp_det_clusters"],
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed + 13,
        n_permutations=n_permutations,
        secondary_metrics=["n_fp_det_clusters"],
        primary_ylim=(-0.5, 0.5),
        secondary_ylim=(-5.0, 5.0),
        primary_y_tick_side=None,
        secondary_y_tick_side="right",
    )

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_evaluate_maps_results(
    json_paths: Sequence[str],
    mode: str,
    output_dir: str,
    distribution_unit: str = "subject",
    labels: Optional[Sequence[str]] = None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    n_permutations: int = 10000,
    comparison_seed: int = 12345,
    baseline_label: Optional[str] = None,
) -> None:
    if mode not in {"single", "compare"}:
        raise ValueError("mode must be 'single' or 'compare'.")

    if mode == "single" and len(json_paths) != 1:
        raise ValueError("single mode expects exactly one JSON path.")
    if mode == "compare" and len(json_paths) < 2:
        raise ValueError("compare mode expects at least two JSON paths.")

    if labels is not None and len(labels) != len(json_paths):
        raise ValueError("labels must have same length as json_paths.")
    if distribution_unit not in {"subject", "fold"}:
        raise ValueError("distribution_unit must be 'subject' or 'fold'.")

    label_list = list(labels) if labels is not None else [None] * len(json_paths)
    runs = [_load_run(Path(p), label=l) for p, l in zip(json_paths, label_list)]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "axes.titlepad": 10.0,
            "axes.labelpad": 7.0,
        },
    )

    subject_frames = [_extract_subject_rows(run) for run in runs]
    subject_df = (
        pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()
    )
    if distribution_unit == "fold":
        warnings.warn(
            "distribution-unit=fold is diagnostic only and not appropriate for primary inference.",
            stacklevel=2,
        )

    distribution_df = _aggregate_distribution_units(
        subject_df,
        by_subject=(distribution_unit == "subject"),
    )
    case_df = (
        distribution_df[~distribution_df["is_control"]].copy()
        if not distribution_df.empty
        else pd.DataFrame()
    )

    run_order = [r.label for r in runs]
    run_palette = {
        run_label: RUN_COLORS[idx % len(RUN_COLORS)]
        for idx, run_label in enumerate(run_order)
    }

    _plot_metric_boxplots(
        case_df,
        metric_cols=["voxel_precision", "voxel_recall", "voxel_dice"],
        metric_name_map={
            "voxel_dice": "Dice",
            "voxel_precision": "Precision",
            "voxel_recall": "Recall",
        },
        mode=mode,
        title="Fixed threshold: voxel-level (per-subject held-out test predictions)",
        out_path=out_root / "fixed_voxel_macro_bars_ci.png",
        bracket_side="auto",
        hue_order=run_order,
        single_color=BLUE_MAIN,
        compare_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed,
    )

    _plot_metric_boxplots(
        case_df,
        metric_cols=[
            "cluster_det_precision",
            "cluster_det_recall",
            "cluster_det_f1",
        ],
        metric_name_map={
            "cluster_det_precision": "Precision",
            "cluster_det_recall": "Recall",
            "cluster_det_f1": "F1",
        },
        mode=mode,
        title="Fixed threshold: cluster-level detection (per-subject held-out test predictions)",
        out_path=out_root / "fixed_cluster_detection_bars_ci.png",
        bracket_side="below",
        hue_order=run_order,
        single_color=RED_MAIN,
        compare_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed + 1,
    )

    _plot_metric_boxplots(
        case_df,
        metric_cols=[
            "cluster_pin_precision",
            "cluster_pin_recall",
            "cluster_pin_f1",
        ],
        metric_name_map={
            "cluster_pin_precision": "Precision",
            "cluster_pin_recall": "Recall",
            "cluster_pin_f1": "F1",
        },
        mode=mode,
        title="Fixed threshold: cluster-level pinpointing (per-subject held-out test predictions)",
        out_path=out_root / "fixed_cluster_pinpoint_bars_ci.png",
        bracket_side="below",
        hue_order=run_order,
        single_color=RED_LIGHT,
        compare_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed + 2,
    )

    subject_summary_distribution = _subject_summary_distribution(
        subject_df,
        by_subject=(distribution_unit == "subject"),
    )
    _plot_subject_bars(
        summary_distribution_df=subject_summary_distribution,
        case_distribution_df=case_df,
        mode=mode,
        out_path=out_root / "fixed_subject_bars.png",
        distribution_unit_label=distribution_unit,
        hue_order=run_order,
        single_color=BLUE_MAIN,
        compare_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed + 3,
    )

    _plot_combined_boxplot_summary(
        case_df=case_df,
        subject_summary_distribution=subject_summary_distribution,
        mode=mode,
        out_path=out_root / "fixed_overall_summary_bars_ci.png",
        distribution_unit_label=distribution_unit,
        hue_order=run_order,
        run_palette=run_palette,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed + 4,
    )

    _plot_variable_curves(
        runs=runs,
        out_path=out_root / "variable_threshold_curves.png",
        mode=mode,
        run_palette=run_palette,
    )

    summary_df = _summary_patient_bootstrap_rows(
        subject_df=subject_df,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=comparison_seed,
    )
    summary_csv = out_root / "summary_patient_bootstrap.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    compare_warnings = _validate_compare_inputs(subject_df)

    if mode == "compare":
        for w in compare_warnings:
            print(f"[warning] {w}")

        run_labels = [r.label for r in runs]
        comparisons: List[pd.DataFrame] = []

        if len(run_labels) == 2:
            baseline = run_labels[0]
            comparator = run_labels[1]
            pair_warnings = _pair_validation_warnings(subject_df, baseline, comparator)
            compare_warnings.extend(pair_warnings)
            for w in pair_warnings:
                print(f"[warning] {w}")
            if any("No overlapping subject IDs" in w for w in pair_warnings):
                raise ValueError("Cannot run paired comparison without overlapping subjects.")

            comp_df = _paired_compare_two_runs(
                run_a=baseline,
                run_b=comparator,
                subject_df=subject_df,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                ci_level=ci_level,
                seed=comparison_seed,
            )
            comparisons.append(comp_df)

            joined = (
                subject_df[subject_df["run"] == baseline]
                .merge(
                    subject_df[subject_df["run"] == comparator],
                    on="subject_id",
                    suffixes=("_a", "_b"),
                    how="inner",
                )
            )
            joined["fp_subject_indicator_controls_a"] = np.where(
                joined["is_control_a"] == True,
                (joined["n_pred_clusters_a"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float),
                np.nan,
            )
            joined["fp_subject_indicator_controls_b"] = np.where(
                joined["is_control_b"] == True,
                (joined["n_pred_clusters_b"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float),
                np.nan,
            )

            _plot_paired_differences_combined(
                baseline_label=baseline,
                comparator_label=comparator,
                metrics=[
                    "voxel_precision",
                    "voxel_recall",
                    "voxel_dice",
                ],
                joined_df=joined,
                out_path=out_root / "paired_differences_voxel.png",
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=comparison_seed,
                n_permutations=n_permutations,
                primary_ylim=(-0.5, 0.5),
                title="Paired differences: voxel metrics",
            )

            _plot_paired_differences_combined(
                baseline_label=baseline,
                comparator_label=comparator,
                metrics=[
                    "cluster_det_precision",
                    "cluster_det_recall",
                    "cluster_det_f1",
                ],
                joined_df=joined,
                out_path=out_root / "paired_differences_cluster_detection.png",
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=comparison_seed + 1,
                n_permutations=n_permutations,
                primary_ylim=(-0.5, 0.5),
                title="Paired differences: cluster detection",
            )

            _plot_paired_differences_combined(
                baseline_label=baseline,
                comparator_label=comparator,
                metrics=[
                    "cluster_pin_precision",
                    "cluster_pin_recall",
                    "cluster_pin_f1",
                ],
                joined_df=joined,
                out_path=out_root / "paired_differences_cluster_pinpointing.png",
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=comparison_seed + 11,
                n_permutations=n_permutations,
                primary_ylim=(-0.5, 0.5),
                title="Paired differences: cluster pinpointing",
            )

            _plot_paired_differences_combined(
                baseline_label=baseline,
                comparator_label=comparator,
                metrics=[
                    "subject_detected",
                    "subject_pinpointed",
                    "fp_subject_indicator_controls",
                    "n_fp_det_clusters",
                ],
                joined_df=joined,
                out_path=out_root / "paired_differences_subject.png",
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=comparison_seed + 2,
                n_permutations=n_permutations,
                primary_ylim=(-0.5, 0.5),
                secondary_metrics=["n_fp_det_clusters"],
                secondary_ylim=(-5.0, 5.0),
                title="Paired differences: subject-level outcomes",
            )

            _plot_compact_comparison_grid(
                case_df=case_df,
                subject_summary_distribution=subject_summary_distribution,
                joined_df=joined,
                run_order=run_order,
                out_path=out_root / "compact_comparison_grid.png",
                run_palette=run_palette,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=comparison_seed + 20,
                n_permutations=n_permutations,
            )
        else:
            if baseline_label is None:
                raise ValueError(
                    "More than two runs supplied. Provide --baseline_label for paired baseline comparisons."
                )
            if baseline_label not in run_labels:
                raise ValueError(f"baseline-label '{baseline_label}' not found in run labels: {run_labels}")
            for comparator in run_labels:
                if comparator == baseline_label:
                    continue
                pair_warnings = _pair_validation_warnings(subject_df, baseline_label, comparator)
                compare_warnings.extend(pair_warnings)
                for w in pair_warnings:
                    print(f"[warning] {w}")
                if any("No overlapping subject IDs" in w for w in pair_warnings):
                    continue

                comp_df = _paired_compare_two_runs(
                    run_a=baseline_label,
                    run_b=comparator,
                    subject_df=subject_df,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    ci_level=ci_level,
                    seed=comparison_seed,
                )
                comparisons.append(comp_df)

                joined = (
                    subject_df[subject_df["run"] == baseline_label]
                    .merge(
                        subject_df[subject_df["run"] == comparator],
                        on="subject_id",
                        suffixes=("_a", "_b"),
                        how="inner",
                    )
                )
                joined["fp_subject_indicator_controls_a"] = np.where(
                    joined["is_control_a"] == True,
                    (joined["n_pred_clusters_a"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float),
                    np.nan,
                )
                joined["fp_subject_indicator_controls_b"] = np.where(
                    joined["is_control_b"] == True,
                    (joined["n_pred_clusters_b"].fillna(0.0).to_numpy(dtype=float) > 0.0).astype(float),
                    np.nan,
                )
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", comparator)
                _plot_paired_differences_combined(
                    baseline_label=baseline_label,
                    comparator_label=comparator,
                    metrics=[
                        "voxel_precision",
                        "voxel_recall",
                        "voxel_dice",
                    ],
                    joined_df=joined,
                    out_path=out_root / f"paired_differences_{safe_name}_vs_baseline_voxel.png",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=comparison_seed,
                    n_permutations=n_permutations,
                    primary_ylim=(-0.5, 0.5),
                    title=f"Paired differences (voxel): {comparator} - {baseline_label}",
                )
                _plot_paired_differences_combined(
                    baseline_label=baseline_label,
                    comparator_label=comparator,
                    metrics=[
                        "cluster_det_precision",
                        "cluster_det_recall",
                        "cluster_det_f1",
                    ],
                    joined_df=joined,
                    out_path=out_root / f"paired_differences_{safe_name}_vs_baseline_cluster_detection.png",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=comparison_seed + 1,
                    n_permutations=n_permutations,
                    primary_ylim=(-0.5, 0.5),
                    title=f"Paired differences (cluster detection): {comparator} - {baseline_label}",
                )
                _plot_paired_differences_combined(
                    baseline_label=baseline_label,
                    comparator_label=comparator,
                    metrics=[
                        "cluster_pin_precision",
                        "cluster_pin_recall",
                        "cluster_pin_f1",
                    ],
                    joined_df=joined,
                    out_path=out_root / f"paired_differences_{safe_name}_vs_baseline_cluster_pinpointing.png",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=comparison_seed + 11,
                    n_permutations=n_permutations,
                    primary_ylim=(-0.5, 0.5),
                    title=f"Paired differences (cluster pinpointing): {comparator} - {baseline_label}",
                )
                _plot_paired_differences_combined(
                    baseline_label=baseline_label,
                    comparator_label=comparator,
                    metrics=[
                        "subject_detected",
                        "subject_pinpointed",
                        "fp_subject_indicator_controls",
                        "n_fp_det_clusters",
                    ],
                    joined_df=joined,
                    out_path=out_root / f"paired_differences_{safe_name}_vs_baseline_subject.png",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=comparison_seed + 2,
                    n_permutations=n_permutations,
                    primary_ylim=(-0.5, 0.5),
                    secondary_metrics=["n_fp_det_clusters"],
                    secondary_ylim=(-5.0, 5.0),
                    title=f"Paired differences (subject): {comparator} - {baseline_label}",
                )

                _plot_compact_comparison_grid(
                    case_df=case_df[case_df["run"].isin([baseline_label, comparator])].copy(),
                    subject_summary_distribution=subject_summary_distribution[
                        subject_summary_distribution["run"].isin([baseline_label, comparator])
                    ].copy(),
                    joined_df=joined,
                    run_order=[baseline_label, comparator],
                    out_path=out_root / f"compact_comparison_grid_{safe_name}_vs_baseline.png",
                    run_palette={
                        baseline_label: run_palette[baseline_label],
                        comparator: run_palette[comparator],
                    },
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=comparison_seed + 20,
                    n_permutations=n_permutations,
                )

        if comparisons:
            comparison_df = pd.concat(comparisons, ignore_index=True)
            if not comparison_df.empty and "p_value_signflip_mean" in comparison_df.columns:
                pvals = comparison_df["p_value_signflip_mean"].fillna(1.0).to_list()
                comparison_df["p_value_holm"] = _holm_adjust([float(p) for p in pvals])
            comparison_csv = out_root / "paired_method_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            print(f"Saved: {comparison_csv}")

        if compare_warnings:
            warning_txt = out_root / "comparison_warnings.txt"
            warning_txt.write_text("\n".join(compare_warnings), encoding="utf-8")
            print(f"Saved: {warning_txt}")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evaluate_maps outputs (single run or compare runs)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "compare"],
        required=True,
        help="single: one JSON, compare: multiple JSONs",
    )
    parser.add_argument(
        "--jsons",
        nargs="+",
        required=True,
        help="Path(s) to evaluate_maps JSON files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional run labels (same length as --jsons).",
    )
    parser.add_argument(
        "--distribution_unit",
        type=str,
        choices=["subject", "fold"],
        default="subject",
        help=(
            "Primary plotting unit. subject is primary; fold is diagnostic only."
        ),
    )
    parser.add_argument(
        "--distribution_by_subject",
        action="store_true",
        help="Deprecated alias for --distribution_unit subject.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Bootstrap replicates for summary and paired CI estimation.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals.",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=10000,
        help="Sign-flip permutation count for paired mean-difference p-values.",
    )
    parser.add_argument(
        "--comparison_seed",
        type=int,
        default=42,
        help="Seed used by paired bootstrap/permutation procedures.",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default=None,
        help="Required when comparing more than two runs; baseline run label.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where plots are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    distribution_unit = args.distribution_unit
    if args.distribution_by_subject:
        distribution_unit = "subject"
    plot_evaluate_maps_results(
        json_paths=args.jsons,
        mode=args.mode,
        output_dir=args.output_dir,
        distribution_unit=distribution_unit,
        labels=args.labels,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        n_permutations=args.n_permutations,
        comparison_seed=args.comparison_seed,
        baseline_label=args.baseline_label,
    )
