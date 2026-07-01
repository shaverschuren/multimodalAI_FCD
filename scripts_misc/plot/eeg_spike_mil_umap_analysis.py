"""
UMAP analysis for exported EEG MIL feature vectors.

This script is for descriptive representation analysis from one selected trained
cross-validation checkpoint (or explicitly allowed mixed checkpoints).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _check_dependencies():
    missing = []
    try:
        import numpy as _np  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import pandas as _pd  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import matplotlib.pyplot as _plt  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import umap as _umap  # noqa: F401
    except ImportError:
        missing.append("umap-learn")
    try:
        from sklearn.preprocessing import StandardScaler as _SS  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise ImportError(
            "Missing required Python packages: "
            f"{joined}. Install with: pip install {joined}"
        )


_check_dependencies()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler, normalize


CANONICAL_LABELS = [
    "Left frontal",
    "Right frontal",
    "Left temporal",
    "Right temporal",
    "Left insular",
    "Right insular",
    "Left parietal",
    "Right parietal",
    "Left occipital",
    "Right occipital",
]

UMAP_COLOR_MAP = {
    "Left frontal": "#63B2D1",
    "Right frontal": "#016097",
    "Left temporal": "#F2B95E",
    "Right temporal": "#B34E02",
    "Left parietal": "#69BE9D",
    "Right parietal": "#018663",
    "Left occipital": "#DFA3C3",
    "Right occipital": "#CF3A67",
    "Unknown": "#9A9A9A",
}

LEGEND_FONT_SIZE = 8
LEGEND_TITLE_FONT_SIZE = 8
LEGEND_MARKER_SIZE = 7


def _normalize_anatomic_label(label: str) -> str:
    s = str(label).strip()
    if not s:
        return "Unknown"

    s_low = s.lower()
    if s_low in {"unknown", "nan", "none"}:
        return "Unknown"

    parts = s_low.split()
    if len(parts) >= 2 and parts[0] in {"left", "right"}:
        lat = parts[0].capitalize()
        lobe = parts[1]
        return f"{lat} {lobe}"

    return s


def _load_artifact(npz_path: Path) -> Dict:
    sidecar = npz_path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing sidecar JSON for feature file: {npz_path}")

    with open(sidecar, "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path, allow_pickle=True)
    return {"npz_path": npz_path, "meta_path": sidecar, "meta": meta, "data": data}


def _decode_strings(arr):
    out = np.asarray(arr)
    if out.dtype.kind in {"S", "O"}:
        return out.astype(str)
    return out


def _require_keys(data: np.lib.npyio.NpzFile, keys: List[str], source: Path):
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {source}: {missing}")


def _preprocess_features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X
    if mode == "l2":
        return normalize(X, norm="l2")
    if mode == "standardize":
        return StandardScaler().fit_transform(X)
    raise ValueError(f"Unsupported preprocess mode: {mode}")


def _compatibility_signature(meta: Dict) -> Tuple:
    return (
        meta.get("feature_space_id"),
        meta.get("checkpoint_sha256"),
        meta.get("model_class"),
        int(meta.get("embedding_dim", -1)),
        meta.get("feature_definition"),
    )


def _build_label_palette(labels: List[str]) -> Dict[str, str]:
    neutral = {
        "Unknown": "#9e9e9e",
        "Bilateral": "#616161",
        "Multilobar": "#bdbdbd",
    }
    obs = list(dict.fromkeys(labels))

    base_colors = plt.get_cmap("tab20").colors
    canonical_present = [l for l in CANONICAL_LABELS if l in obs]
    other_present = [l for l in obs if l not in canonical_present and l not in neutral]

    palette = {}
    idx = 0
    for label in canonical_present + other_present:
        palette[label] = base_colors[idx % len(base_colors)]
        idx += 1
    for label, color in neutral.items():
        if label in obs:
            palette[label] = color

    return palette


def _build_anatomic_palette(labels: List[str]) -> Dict[str, str]:
    categories = list(dict.fromkeys([_normalize_anatomic_label(str(x)) for x in labels]))
    palette = {}
    for cat in categories:
        palette[cat] = UMAP_COLOR_MAP.get(cat, UMAP_COLOR_MAP["Unknown"])
    return palette


def _scatter_by_category(ax, df: pd.DataFrame, xcol: str, ycol: str, ccol: str, palette: Dict[str, str],
                         point_size: float, alpha: float, title: str):
    categories = list(dict.fromkeys(df[ccol].astype(str).tolist()))
    for cat in categories:
        sub = df[df[ccol].astype(str) == cat]
        ax.scatter(
            sub[xcol].values,
            sub[ycol].values,
            s=point_size,
            alpha=alpha,
            color=palette.get(cat, "#9e9e9e"),
            label=cat,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def _marker_for_anatomic_label(label: str) -> str:
    l = _normalize_anatomic_label(str(label)).strip().lower()
    if l.startswith("right "):
        return "^"
    if l.startswith("left "):
        return "o"
    return "o"


def _scatter_patient_by_anatomy(ax, df: pd.DataFrame, xcol: str, ycol: str, ccol: str, palette: Dict[str, str],
                                point_size: float, alpha: float, title: str):
    plot_labels = df[ccol].astype(str).map(_normalize_anatomic_label)
    categories = list(dict.fromkeys(plot_labels.tolist()))
    for cat in categories:
        sub = df[plot_labels == cat]
        ax.scatter(
            sub[xcol].values,
            sub[ycol].values,
            s=point_size,
            alpha=alpha,
            color=palette.get(cat, UMAP_COLOR_MAP["Unknown"]),
            marker=_marker_for_anatomic_label(cat),
            label=cat,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def _save_dataframe_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_hemisphere_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"left", "l"}:
        return "Left"
    if s in {"right", "r"}:
        return "Right"
    return "Unknown"


def _ordered_anatomic_labels(labels: List[str]) -> List[str]:
    desired = [
        "Left frontal", "Right frontal",
        "Left temporal", "Right temporal",
        "Left parietal", "Right parietal",
        "Left occipital", "Right occipital",
        "Unknown",
    ]
    present = set([str(x) for x in labels])
    ordered = [lab for lab in desired if lab in present]
    extras = sorted([lab for lab in present if lab not in desired])
    return ordered + extras


def _ordered_hemisphere_labels(labels: List[str]) -> List[str]:
    desired = ["Left", "Right", "Unknown"]
    present = set([str(x) for x in labels])
    ordered = [lab for lab in desired if lab in present]
    extras = sorted([lab for lab in present if lab not in desired])
    return ordered + extras


def _ordered_split_labels(labels: List[str]) -> List[str]:
    desired = ["train", "val", "validation", "test"]
    present = set([str(x) for x in labels])
    ordered = [lab for lab in desired if lab in present]
    extras = sorted([lab for lab in present if lab not in desired])
    return ordered + extras


def _legend_handles(
    labels: List[str],
    palette: Dict[str, str],
    marker_by_label: Dict[str, str] = None,
) -> List[Line2D]:
    marker_by_label = marker_by_label or {}
    handles = []
    for label in labels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_by_label.get(label, "o"),
                linestyle="",
                markerfacecolor=palette.get(label, UMAP_COLOR_MAP["Unknown"]),
                markeredgecolor="none",
                markersize=LEGEND_MARKER_SIZE,
            )
        )
    return handles


def main():
    parser = argparse.ArgumentParser(description="UMAP analysis of EEG MIL exported features")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--feature_glob", type=str, default="features_*.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--umap_random_state", type=int, default=42)
    parser.add_argument("--preprocess", type=str, choices=["none", "l2", "standardize"], default="l2")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--make_split_diagnostics", action="store_true")
    parser.add_argument("--allow_mixed_feature_spaces", action="store_true")
    args = parser.parse_args()

    # Descriptive analysis note: this UMAP visualizes representations from one selected
    # trained cross-validation checkpoint over a cohort and is not an independent OOF performance analysis.

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    files = sorted(feature_dir.glob(args.feature_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.feature_glob!r} in {feature_dir}")

    artifacts = [_load_artifact(p) for p in files]

    signatures = [_compatibility_signature(a["meta"]) for a in artifacts]
    unique_signatures = list({s for s in signatures})
    mixed_spaces = len(unique_signatures) > 1

    if mixed_spaces and not args.allow_mixed_feature_spaces:
        raise ValueError(
            "Detected multiple incompatible feature spaces across inputs. "
            "Raw latent embeddings from independently trained checkpoints are not guaranteed to be comparable; "
            "mixing may produce geometry driven by checkpoint identity rather than lesion biology. "
            "Use --allow_mixed_feature_spaces only for exploratory analysis."
        )

    if mixed_spaces and args.allow_mixed_feature_spaces:
        print("WARNING: mixed feature spaces enabled; exploratory only.")

    spike_features_list = []
    pooled_features_list = []
    spike_rows = []
    pooled_rows = []
    input_feature_space_ids = []
    observed_labels = set()

    for art in artifacts:
        npz_path = art["npz_path"]
        meta = art["meta"]
        data = art["data"]

        input_feature_space_ids.append(meta.get("feature_space_id"))

        _require_keys(
            data,
            [
                "spike_features",
                "spike_subject_ids",
                "spike_fold",
                "spike_split",
                "spike_original_indices",
                "spike_bag_positions",
                "spike_combined_lobe_labels",
                "pooled_features",
                "pooled_subject_ids",
                "pooled_fold",
                "pooled_split",
                "pooled_n_selected_spikes",
                "pooled_combined_lobe_labels",
            ],
            npz_path,
        )

        spike_features = np.asarray(data["spike_features"], dtype=np.float32)
        pooled_features = np.asarray(data["pooled_features"], dtype=np.float32)

        if spike_features.ndim != 2 or pooled_features.ndim != 2:
            raise ValueError(f"Invalid feature dimensions in {npz_path}")

        spike_subject_ids = _decode_strings(data["spike_subject_ids"])
        spike_fold = np.asarray(data["spike_fold"], dtype=np.int32)
        spike_split = _decode_strings(data["spike_split"])
        spike_original_indices = np.asarray(data["spike_original_indices"], dtype=np.int64)
        spike_bag_positions = np.asarray(data["spike_bag_positions"], dtype=np.int32)
        spike_combined = _decode_strings(data["spike_combined_lobe_labels"])

        spike_lobe = _decode_strings(data["spike_lobe_label_names"]) if "spike_lobe_label_names" in data else np.full(spike_combined.shape, "Unknown")
        spike_lat = _decode_strings(data["spike_laterality_label_names"]) if "spike_laterality_label_names" in data else np.full(spike_combined.shape, "Unknown")

        pooled_subject_ids = _decode_strings(data["pooled_subject_ids"])
        pooled_fold = np.asarray(data["pooled_fold"], dtype=np.int32)
        pooled_split = _decode_strings(data["pooled_split"])
        pooled_combined = _decode_strings(data["pooled_combined_lobe_labels"])
        pooled_lobe = _decode_strings(data["pooled_lobe_label_names"]) if "pooled_lobe_label_names" in data else np.full(pooled_combined.shape, "Unknown")
        pooled_lat = _decode_strings(data["pooled_laterality_label_names"]) if "pooled_laterality_label_names" in data else np.full(pooled_combined.shape, "Unknown")

        if spike_features.shape[0] != spike_subject_ids.shape[0]:
            raise ValueError(f"Spike row mismatch in {npz_path}")
        if pooled_features.shape[0] != pooled_subject_ids.shape[0]:
            raise ValueError(f"Pooled row mismatch in {npz_path}")

        feature_space_id = str(meta.get("feature_space_id", "unknown"))

        for i in range(spike_features.shape[0]):
            row = {
                "subject_id": str(spike_subject_ids[i]),
                "fold": int(spike_fold[i]),
                "split": str(spike_split[i]),
                "combined_lobe_label": str(spike_combined[i]),
                "lobe_label": str(spike_lobe[i]),
                "laterality_label": str(spike_lat[i]),
                "feature_space_id": feature_space_id,
                "original_spike_index": int(spike_original_indices[i]),
                "bag_position": int(spike_bag_positions[i]),
            }
            for optional_key in ["spike_time", "spike_perception_score", "spike_channel"]:
                if optional_key in data:
                    values = _decode_strings(data[optional_key])
                    if values.shape[0] == spike_features.shape[0]:
                        row[optional_key] = values[i]
            spike_rows.append(row)
            observed_labels.add(row["combined_lobe_label"])

        for i in range(pooled_features.shape[0]):
            row = {
                "subject_id": str(pooled_subject_ids[i]),
                "fold": int(pooled_fold[i]),
                "split": str(pooled_split[i]),
                "combined_lobe_label": str(pooled_combined[i]),
                "lobe_label": str(pooled_lobe[i]),
                "laterality_label": str(pooled_lat[i]),
                "feature_space_id": feature_space_id,
            }
            pooled_rows.append(row)
            observed_labels.add(row["combined_lobe_label"])

        spike_features_list.append(spike_features)
        pooled_features_list.append(pooled_features)

    X_spike = np.concatenate(spike_features_list, axis=0)
    X_patient = np.concatenate(pooled_features_list, axis=0)

    if X_spike.shape[1] != X_patient.shape[1]:
        raise ValueError(
            f"Embedding dimensionality mismatch: spike D={X_spike.shape[1]}, patient D={X_patient.shape[1]}"
        )

    X_spike_p = _preprocess_features(X_spike, args.preprocess)
    X_patient_p = _preprocess_features(X_patient, args.preprocess)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_random_state,
    )
    spike_umap = reducer.fit_transform(X_spike_p)

    reducer_patient = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_random_state,
    )
    patient_umap = reducer_patient.fit_transform(X_patient_p)

    spike_df = pd.DataFrame(spike_rows)
    patient_df = pd.DataFrame(pooled_rows)

    spike_df["umap_1"] = spike_umap[:, 0]
    spike_df["umap_2"] = spike_umap[:, 1]
    patient_df["umap_1"] = patient_umap[:, 0]
    patient_df["umap_2"] = patient_umap[:, 1]

    spike_df["combined_lobe_label"] = spike_df["combined_lobe_label"].astype(str).map(_normalize_anatomic_label)
    patient_df["combined_lobe_label"] = patient_df["combined_lobe_label"].astype(str).map(_normalize_anatomic_label)

    output_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = ""
    if mixed_spaces and args.allow_mixed_feature_spaces:
        title_suffix = " | mixed feature spaces - exploratory only"

    anatomic_palette = _build_anatomic_palette(
        spike_df["combined_lobe_label"].astype(str).tolist() + patient_df["combined_lobe_label"].astype(str).tolist()
    )

    ckpt_short = Path(str(artifacts[0]["meta"].get("checkpoint_filename", "checkpoint"))).stem
    fold_short = artifacts[0]["meta"].get("fold", "?")

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    _scatter_by_category(
        ax1,
        spike_df,
        "umap_1",
        "umap_2",
        "combined_lobe_label",
        anatomic_palette,
        point_size=8,
        alpha=0.45,
        title=(
            f"Per-spike feature UMAP | patients={patient_df.shape[0]}, spikes={spike_df.shape[0]}"
            f" | n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, metric={args.umap_metric}"
            f" | {ckpt_short} fold={fold_short}{title_suffix}"
        ),
    )
    legend_labels = ["Frontal", "Temporal", "Parietal", "Occipital", "Unknown"]
    legend_colors = [
        UMAP_COLOR_MAP["Left frontal"],
        UMAP_COLOR_MAP["Left temporal"],
        UMAP_COLOR_MAP["Left parietal"],
        UMAP_COLOR_MAP["Left occipital"],
        UMAP_COLOR_MAP["Unknown"],
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=c,
            markeredgecolor="none",
            markersize=LEGEND_MARKER_SIZE,
        )
        for c in legend_colors
    ]
    ax1.legend(
        legend_handles,
        legend_labels,
        loc="best",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        title="Lobes by color family\nlight=left, dark=right",
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )
    fig1.tight_layout()
    fig1.savefig(output_dir / "umap_spike_level.png", dpi=args.dpi)
    plt.close(fig1)

    hemisphere_palette = {
        "Right": "#0062B2",
        "Left": "#75FFAA",
        "Unknown": "#9A9A9A",
    }
    spike_df["hemisphere"] = spike_df["laterality_label"].astype(str).map(_normalize_hemisphere_label)

    fig1h, ax1h = plt.subplots(figsize=(9, 7))
    _scatter_by_category(
        ax1h,
        spike_df,
        "umap_1",
        "umap_2",
        "hemisphere",
        hemisphere_palette,
        point_size=8,
        alpha=0.45,
        title=(
            f"Per-spike feature UMAP by hemisphere | patients={patient_df.shape[0]}, spikes={spike_df.shape[0]}"
            f" | n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, metric={args.umap_metric}"
            f" | {ckpt_short} fold={fold_short}{title_suffix}"
        ),
    )
    hemi_labels = _ordered_hemisphere_labels(spike_df["hemisphere"].astype(str).tolist())
    ax1h.legend(
        _legend_handles(hemi_labels, hemisphere_palette),
        hemi_labels,
        loc="best",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig1h.tight_layout()
    fig1h.savefig(output_dir / "umap_spike_level_by_hemisphere.png", dpi=args.dpi)
    plt.close(fig1h)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    _scatter_patient_by_anatomy(
        ax2,
        patient_df,
        "umap_1",
        "umap_2",
        "combined_lobe_label",
        anatomic_palette,
        point_size=48,
        alpha=0.85,
        title=(
            f"Per-patient mean-pooled feature UMAP | patients={patient_df.shape[0]}"
            f" | n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, metric={args.umap_metric}"
            f" | {ckpt_short} fold={fold_short}{title_suffix}"
        ),
    )
    patient_labels = _ordered_anatomic_labels(patient_df["combined_lobe_label"].astype(str).tolist())
    patient_markers = {lab: _marker_for_anatomic_label(lab) for lab in patient_labels}
    ax2.legend(
        _legend_handles(patient_labels, anatomic_palette, marker_by_label=patient_markers),
        patient_labels,
        loc="best",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig2.tight_layout()
    fig2.savefig(output_dir / "umap_patient_level.png", dpi=args.dpi)
    plt.close(fig2)

    figc, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    _scatter_by_category(
        axes[0], spike_df, "umap_1", "umap_2", "combined_lobe_label", anatomic_palette,
        point_size=8, alpha=0.45, title="A. Per-spike feature UMAP"
    )
    _scatter_patient_by_anatomy(
        axes[1], patient_df, "umap_1", "umap_2", "combined_lobe_label", anatomic_palette,
        point_size=48, alpha=0.85, title="B. Mean-pooled patient feature UMAP"
    )

    combined_labels = _ordered_anatomic_labels(patient_df["combined_lobe_label"].astype(str).tolist())
    combined_markers = {lab: _marker_for_anatomic_label(lab) for lab in combined_labels}
    figc.legend(
        _legend_handles(combined_labels, anatomic_palette, marker_by_label=combined_markers),
        combined_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    figc.suptitle(
        "UMAP of EEG MIL representations"
        + title_suffix,
        fontsize=12,
    )
    figc.tight_layout(rect=[0, 0.07, 1, 0.95])
    figc.savefig(output_dir / "umap_combined.png", dpi=args.dpi)
    plt.close(figc)

    if args.make_split_diagnostics:
        split_palette = {"train": "#1b9e77", "val": "#d95f02", "validation": "#d95f02", "test": "#7570b3"}

        fig_s, ax_s = plt.subplots(figsize=(9, 7))
        _scatter_by_category(
            ax_s,
            spike_df,
            "umap_1",
            "umap_2",
            "split",
            split_palette,
            point_size=8,
            alpha=0.45,
            title=f"Diagnostic only: per-spike UMAP by split{title_suffix}",
        )
        split_labels = _ordered_split_labels(spike_df["split"].astype(str).tolist())
        ax_s.legend(
            _legend_handles(split_labels, split_palette),
            split_labels,
            loc="best",
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
        fig_s.tight_layout()
        fig_s.savefig(output_dir / "umap_spike_level_by_split.png", dpi=args.dpi)
        plt.close(fig_s)

        fig_p, ax_p = plt.subplots(figsize=(9, 7))
        _scatter_by_category(
            ax_p,
            patient_df,
            "umap_1",
            "umap_2",
            "split",
            split_palette,
            point_size=48,
            alpha=0.85,
            title=f"Diagnostic only: per-patient UMAP by split{title_suffix}",
        )
        split_labels_p = _ordered_split_labels(patient_df["split"].astype(str).tolist())
        ax_p.legend(
            _legend_handles(split_labels_p, split_palette),
            split_labels_p,
            loc="best",
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
        fig_p.tight_layout()
        fig_p.savefig(output_dir / "umap_patient_level_by_split.png", dpi=args.dpi)
        plt.close(fig_p)

    spike_csv_cols = [
        "umap_1",
        "umap_2",
        "subject_id",
        "fold",
        "split",
        "combined_lobe_label",
        "lobe_label",
        "laterality_label",
        "feature_space_id",
        "original_spike_index",
        "bag_position",
    ]
    for optional_col in ["spike_time", "spike_perception_score", "spike_channel"]:
        if optional_col in spike_df.columns:
            spike_csv_cols.append(optional_col)

    patient_csv_cols = [
        "umap_1",
        "umap_2",
        "subject_id",
        "fold",
        "split",
        "combined_lobe_label",
        "lobe_label",
        "laterality_label",
        "feature_space_id",
    ]

    _save_dataframe_csv(spike_df[spike_csv_cols], output_dir / "umap_spike_coordinates.csv")
    _save_dataframe_csv(patient_df[patient_csv_cols], output_dir / "umap_patient_coordinates.csv")

    observed = sorted(observed_labels)
    noncanonical = sorted([l for l in observed if l not in CANONICAL_LABELS])

    metadata = {
        "input_files": [str(a["npz_path"]) for a in artifacts],
        "input_feature_space_ids": sorted(set(input_feature_space_ids)),
        "number_of_patients": int(patient_df.shape[0]),
        "number_of_spikes": int(spike_df.shape[0]),
        "embedding_dim": int(X_spike.shape[1]),
        "preprocessing": args.preprocess,
        "umap_parameters": {
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "random_state": args.umap_random_state,
        },
        "observed_labels": observed,
        "unknown_or_noncanonical_labels": noncanonical,
        "make_split_diagnostics": bool(args.make_split_diagnostics),
        "whether_mixed_feature_spaces_were_allowed": bool(args.allow_mixed_feature_spaces),
        "mixed_feature_spaces_detected": bool(mixed_spaces),
        "interpretation_note": (
            "This is a descriptive UMAP of representations from one selected trained cross-validation checkpoint "
            "run over the full cohort. It is not an out-of-fold latent-space analysis and should not be interpreted "
            "as an independent performance evaluation."
        ),
    }

    with open(output_dir / "umap_analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved UMAP outputs to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
