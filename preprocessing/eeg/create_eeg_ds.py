"""
create_eeg_ds.py
Create EEG dataset from EDF files with channel selection, filtering, and segmentation of spike segments.
Assumes that bad/missing channels have been interpolated and data has been re-referenced earlier.
Requires extracted detection timestamps (integer seconds from start of recording), see extract_event_timestamps.py.
Saves processed segments as NumPy arrays.

For the filtering and segment extraction, we're doing some parallelisation on Snellius. 
Also, for OOM purposes, we're doing chunked FIR filtering to avoid large FFT buffers.

Default bandpasses: 1.0-70Hz and 70-120Hz (gamma, not used further on in the project yet).
Default notch: 50Hz and 60Hz
Default segment length: 1s
Default channels: Standard 10-20 montage subset + F9/F10

Author: Sjors Verschuren
Date: December 2025
"""

import argparse
import datetime
import itertools
import gc
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from mne.filter import create_filter, _overlap_add_filter

try:
    import mkl  # type: ignore
except ImportError:
    print("\033[38;5;208mWARNING: MKL not found, proceeding without MKL optimizations. This is very much not recommended.\033[0m")
    mkl = None

matplotlib.use("Agg")


def _robust_zscore(x, eps=1e-8):
    """Compute robust z-score using median and MAD scaling."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + eps
    return (x - med) / scale


def drop_noisy_segments(
    segments_bands,
    timestamps,
    valid_mask,
    noise_cfg,
):
    """
    Drop noisy segments after filtering and normalization.

    Criteria are inspired by common adjacent EEG practice:
    - Peak-to-peak outlier rejection (MNE-style epoch amplitude rejection)
    - Flat/dropout detection (PREP/PyPREP-style)
    - Extreme-sample fraction in robustly normalized space
    - High-frequency contamination ratio outlier (PREP/PyPREP-style HF noise)
    """
    if len(segments_bands) == 0 or len(segments_bands[0]) == 0:
        return segments_bands, timestamps, {
            "kept": 0,
            "dropped": 0,
            "drop_fraction": 0.0,
            "reason_counts": {
                "ptp": 0,
                "flat": 0,
                "extreme_samples": 0,
                "hf_ratio": 0,
            },
            "detail": {
                "keep_mask": [],
                "drop_mask": [],
                "bad_ptp": [],
                "bad_flat": [],
                "bad_extreme": [],
                "bad_hf": [],
                "ptp_z": [],
                "flat_frac": [],
                "extreme_frac": [],
                "hf_ratio_z": [],
            },
        }

    primary = segments_bands[0]  # (N, C, T)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        # No valid channels to score on; keep everything unchanged.
        return segments_bands, timestamps, {
            "kept": int(primary.shape[0]),
            "dropped": 0,
            "drop_fraction": 0.0,
            "reason_counts": {
                "ptp": 0,
                "flat": 0,
                "extreme_samples": 0,
                "hf_ratio": 0,
            },
            "detail": {
                "keep_mask": [True] * int(primary.shape[0]),
                "drop_mask": [False] * int(primary.shape[0]),
                "bad_ptp": [False] * int(primary.shape[0]),
                "bad_flat": [False] * int(primary.shape[0]),
                "bad_extreme": [False] * int(primary.shape[0]),
                "bad_hf": [False] * int(primary.shape[0]),
                "ptp_z": [0.0] * int(primary.shape[0]),
                "flat_frac": [0.0] * int(primary.shape[0]),
                "extreme_frac": [0.0] * int(primary.shape[0]),
                "hf_ratio_z": [0.0] * int(primary.shape[0]),
            },
        }

    X = primary[:, valid_idx, :]

    # 1) Peak-to-peak amplitude outliers per epoch
    ptp_per_ch = np.ptp(X, axis=2)  # (N, C_valid)
    ptp_med_per_epoch = np.median(ptp_per_ch, axis=1)
    ptp_z = _robust_zscore(ptp_med_per_epoch)
    bad_ptp = ptp_z > noise_cfg["ptp_z_thresh"]

    # 2) Flat/dropout channels within epoch
    flat_ch = ptp_per_ch < noise_cfg["flat_ptp_thresh"]
    flat_frac = np.mean(flat_ch, axis=1)
    bad_flat = flat_frac > noise_cfg["flat_channel_frac_thresh"]

    # 3) Extreme samples in normalized space
    extreme_frac = np.mean(np.abs(X) > noise_cfg["abs_z_thresh"], axis=(1, 2))
    bad_extreme = extreme_frac > noise_cfg["abs_z_frac_thresh"]

    # 4) High-frequency contamination ratio (if high-frequency band exists)
    if len(segments_bands) > 1:
        X_hf = segments_bands[1][:, valid_idx, :]
        rms_low = np.sqrt(np.mean(X ** 2, axis=(1, 2)))
        rms_hf = np.sqrt(np.mean(X_hf ** 2, axis=(1, 2)))
        hf_ratio = rms_hf / (rms_low + 1e-8)
        hf_ratio_z = _robust_zscore(hf_ratio)
        bad_hf = hf_ratio_z > noise_cfg["hf_ratio_z_thresh"]
    else:
        hf_ratio_z = np.zeros(X.shape[0], dtype=np.float32)
        bad_hf = np.zeros(X.shape[0], dtype=bool)

    drop_mask = bad_ptp | bad_flat | bad_extreme | bad_hf
    keep_mask = ~drop_mask

    filtered_bands = [band[keep_mask] for band in segments_bands]
    filtered_timestamps = timestamps[keep_mask]

    reason_counts = {
        "ptp": int(np.sum(bad_ptp)),
        "flat": int(np.sum(bad_flat)),
        "extreme_samples": int(np.sum(bad_extreme)),
        "hf_ratio": int(np.sum(bad_hf)),
    }

    qc_info = {
        "kept": int(np.sum(keep_mask)),
        "dropped": int(np.sum(drop_mask)),
        "drop_fraction": float(np.mean(drop_mask)),
        "reason_counts": reason_counts,
        "detail": {
            "keep_mask": keep_mask.tolist(),
            "drop_mask": drop_mask.tolist(),
            "bad_ptp": bad_ptp.tolist(),
            "bad_flat": bad_flat.tolist(),
            "bad_extreme": bad_extreme.tolist(),
            "bad_hf": bad_hf.tolist(),
            "ptp_z": ptp_z.astype(np.float32).tolist(),
            "flat_frac": flat_frac.astype(np.float32).tolist(),
            "extreme_frac": extreme_frac.astype(np.float32).tolist(),
            "hf_ratio_z": hf_ratio_z.astype(np.float32).tolist(),
        },
    }

    return filtered_bands, filtered_timestamps, qc_info

# ---------------------------------------------------------------------------
# Flat dataset helpers
# ---------------------------------------------------------------------------

def _persyst_time_to_seconds(time_str):
    """
    Convert a Persyst time string like "d1 00:03:04.8699" into seconds from
    recording start.  Mirrors the logic in extract_event_timestamps.py.
    """
    time_str = time_str.strip().strip('"')
    day_part, clock_str = time_str.split(" ", 1)
    day_idx = int(day_part[1:])                              # "d2" -> 2
    clock_seconds = pd.to_timedelta(clock_str).total_seconds()
    day_offset = (day_idx - 1) * 24 * 3600
    return day_offset + clock_seconds


def load_persyst_detection_csv(csv_path):
    """
    Load a Persyst spike-list CSV.

    Expected columns: Time, Channel, Perception  (case-insensitive lookup).
    Returns a DataFrame with columns:
        time_string, timestamp_sec, detected_channel_raw, detected_channel,
        perception
    Rows with unparsable timestamps are dropped; the count is printed.
    """
    df = pd.read_csv(csv_path)

    # Case-insensitive column resolution
    col_lower = {c.lower(): c for c in df.columns}
    time_col       = col_lower.get("time")
    channel_col    = col_lower.get("channel")
    perception_col = col_lower.get("perception")

    if any(c is None for c in (time_col, channel_col, perception_col)):
        raise ValueError(
            f"Detection CSV missing required columns. "
            f"Found: {list(df.columns)}. Expected: Time, Channel, Perception."
        )

    rows, n_bad = [], 0
    for _, row in df.iterrows():
        time_str = str(row[time_col])
        try:
            ts_sec = _persyst_time_to_seconds(time_str)
        except Exception:
            n_bad += 1
            continue
        channel_raw   = str(row[channel_col])
        channel_clean = channel_raw.split("-")[0]
        rows.append({
            "time_string":           time_str,
            "timestamp_sec":         ts_sec,
            "detected_channel_raw":  channel_raw,
            "detected_channel":      channel_clean,
            "perception":            float(row[perception_col]),
        })

    if n_bad:
        print(f"  > Warning: skipped {n_bad} rows with unparsable timestamps.")

    return pd.DataFrame(rows)


def find_band_index(freq_bands, target_band=(1.0, 70.0)):
    """Return the index in freq_bands matching target_band, or -1 if not found."""
    for i, (l, h) in enumerate(freq_bands):
        if abs(l - target_band[0]) < 1e-6 and abs(h - target_band[1]) < 1e-6:
            return i
    return -1


def extract_single_band_segment(X_band, center_sample, win):
    """
    Slice one segment from X_band (n_channels, n_times).

    Returns (segment_float32, start_sample, end_sample), or
            (None, None, None) if the window is out of bounds.
    """
    start = center_sample - win // 2
    end   = start + win
    if start < 0 or end > X_band.shape[1]:
        return None, None, None
    return X_band[:, start:end].astype(np.float32), int(start), int(end)


def sample_non_spike_centers(
    n_times,
    win,
    sfreq,
    detection_center_samples,
    n_segments=1000,
    exclusion_seconds=1.0,
    random_seed=42,
):
    """
    Randomly sample segment center indices that:
      - keep the whole window in bounds, and
      - do not overlap the exclusion zone around any detection.

    Returns a list of valid center sample indices (may be shorter than
    n_segments if there are not enough valid windows).
    """
    half_win   = win // 2
    min_center = half_win
    max_center = n_times - (win - half_win)

    if max_center <= min_center:
        return []

    det_centers     = np.asarray(detection_center_samples, dtype=np.int64)
    excl_half       = int(round((exclusion_seconds / 2.0) * sfreq))

    def _overlaps(start, end):
        if det_centers.size == 0:
            return False
        return bool(np.any(
            (start < det_centers + excl_half) & (end > det_centers - excl_half)
        ))

    rng          = np.random.default_rng(random_seed)
    valid        = []
    max_attempts = n_segments * 100

    for _ in range(max_attempts):
        if len(valid) >= n_segments:
            break
        c     = int(rng.integers(min_center, max_center))
        start = c - half_win
        end   = start + win
        if not _overlaps(start, end):
            valid.append(c)

    return valid


def create_flat_dataset_for_patient(
    X_band,
    sfreq,
    patient_id,
    detection_csv_path,
    output_dir,
    segment_length,
    spike_detection_threshold,
    n_non_spike_segments,
    non_spike_exclusion_seconds,
    random_seed,
):
    """
    Build and save flat 1-70Hz dataset files for one patient:
      {patient_id}_flat_1-70Hz_segments.npy   (N, C, T)
      {patient_id}_flat_1-70Hz_metadata.csv
      {patient_id}_flat_1-70Hz_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    win = int(segment_length * sfreq)

    # --- Load detection CSV ---
    print(f"  > Loading detection CSV: {detection_csv_path}")
    try:
        det_df = load_persyst_detection_csv(detection_csv_path)
    except Exception as e:
        print(f"\033[93m  > Warning: could not load detection CSV: {e}. Skipping flat export.\033[0m")
        return

    if len(det_df) == 0:
        print(f"\033[93m  > Warning: detection CSV is empty for {patient_id}. Skipping flat export.\033[0m")
        return

    n_thresholded  = int((det_df["perception"] >= spike_detection_threshold).sum())
    n_subthreshold = len(det_df) - n_thresholded
    print(f"  > Loaded {len(det_df)} detection rows "
          f"({n_thresholded} thresholded, {n_subthreshold} subthreshold).")

    # --- Detection segments ---
    segments_list, metadata_list, n_oob = [], [], 0

    for _, row in det_df.iterrows():
        center = int(round(float(row["timestamp_sec"]) * sfreq))
        seg, s, e = extract_single_band_segment(X_band, center, win)
        if seg is None:
            n_oob += 1
            continue
        is_thresh = bool(row["perception"] >= spike_detection_threshold)
        segments_list.append(seg)
        metadata_list.append({
            "patient_id":            patient_id,
            "segment_type":          "detection",
            "label":                 1 if is_thresh else 0,
            "is_persyst_detection":  True,
            "is_thresholded_spike":  is_thresh,
            "perception":            float(row["perception"]),
            "detected_channel":      row["detected_channel"],
            "detected_channel_raw":  row["detected_channel_raw"],
            "time_string":           row["time_string"],
            "timestamp_sec":         float(row["timestamp_sec"]),
            "center_sample":         center,
            "start_sample":          s,
            "end_sample":            e,
            "band":                  "1-70Hz",
        })

    if n_oob:
        print(f"  > Skipped {n_oob} out-of-bounds detection segments.")

    n_det_saved     = len(segments_list)
    n_thresh_saved  = sum(1 for m in metadata_list if m["is_thresholded_spike"])
    n_sub_saved     = n_det_saved - n_thresh_saved
    print(f"  > Detection segments saved: {n_det_saved} "
          f"({n_thresh_saved} thresholded, {n_sub_saved} subthreshold).")

    # --- Non-spike segments ---
    det_centers = np.array(
        [int(round(float(ts) * sfreq)) for ts in det_df["timestamp_sec"]],
        dtype=np.int64,
    )

    stable_offset    = sum(ord(c) for c in patient_id)
    non_spike_centers = sample_non_spike_centers(
        n_times=X_band.shape[1],
        win=win,
        sfreq=sfreq,
        detection_center_samples=det_centers,
        n_segments=n_non_spike_segments,
        exclusion_seconds=non_spike_exclusion_seconds,
        random_seed=random_seed + stable_offset,
    )

    if len(non_spike_centers) < n_non_spike_segments:
        print(
            f"\033[93m  > Warning: only found {len(non_spike_centers)} / "
            f"{n_non_spike_segments} valid non-spike windows for {patient_id}.\033[0m"
        )
    else:
        print(f"  > Non-spike segments sampled: {len(non_spike_centers)}.")

    for center in non_spike_centers:
        seg, s, e = extract_single_band_segment(X_band, center, win)
        if seg is None:
            continue
        segments_list.append(seg)
        metadata_list.append({
            "patient_id":            patient_id,
            "segment_type":          "non_spike",
            "label":                 0,
            "is_persyst_detection":  False,
            "is_thresholded_spike":  False,
            "perception":            float("nan"),
            "detected_channel":      "",
            "detected_channel_raw":  "",
            "time_string":           "",
            "timestamp_sec":         float(center / sfreq),
            "center_sample":         center,
            "start_sample":          s,
            "end_sample":            e,
            "band":                  "1-70Hz",
        })

    n_non_spike_saved = len(segments_list) - n_det_saved

    if not segments_list:
        print(f"\033[93m  > Warning: no segments to save for {patient_id}.\033[0m")
        return

    # --- Save outputs ---
    arr = np.stack(segments_list, axis=0)   # (N, C, T)

    seg_path  = output_dir / f"{patient_id}_flat_1-70Hz_segments.npy"
    meta_path = output_dir / f"{patient_id}_flat_1-70Hz_metadata.csv"
    summ_path = output_dir / f"{patient_id}_flat_1-70Hz_summary.json"

    np.save(seg_path, arr)
    print(f"  > Saved flat segments:  {seg_path}  shape={arr.shape}")

    pd.DataFrame(metadata_list).to_csv(meta_path, index=False)
    print(f"  > Saved flat metadata:  {meta_path}  rows={len(metadata_list)}")

    summary = {
        "patient_id":                          patient_id,
        "n_detection_rows_csv":                int(len(det_df)),
        "n_detection_segments_saved":          int(n_det_saved),
        "n_thresholded_spike_segments_saved":  int(n_thresh_saved),
        "n_subthreshold_detection_segments_saved": int(n_sub_saved),
        "n_non_spike_segments_requested":      int(n_non_spike_segments),
        "n_non_spike_segments_saved":          int(n_non_spike_saved),
        "segment_length_sec":                  float(segment_length),
        "sfreq":                               float(sfreq),
        "band":                                "1-70Hz",
        "spike_detection_threshold":           float(spike_detection_threshold),
        "non_spike_exclusion_seconds":         float(non_spike_exclusion_seconds),
    }
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  > Saved flat summary:   {summ_path}")


# ---------------------------------------------------------------------------

def setup_logger(output_dir):
    """Little logging setup to log stdout and stderr to a file (couldn't catch MNE stuff manually)"""
    # Setup logging to file
    log_file = os.path.join(output_dir, f"processing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(output_dir, exist_ok=True)

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_handle = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    print(f"Logging to: {log_file}")

def quality_control_plot(
    segments_bands,
    timestamps,
    channels_to_keep,
    patient_id,
    n_samples=10,
    figsize=(20, 10),
    downsample=1,
    spike_window_sec=(0.25, 0.75),
    plot_spike_window=True,
    freq_bands=None,
    output_path=None
):
    """
    Generalized QC plot for N frequency bands.
    segments_bands : list of arrays, each shaped (N_segments, C, T)
    freq_bands     : list of (l_freq, h_freq) tuples
    """

    import itertools

    if len(segments_bands) == 0 or len(segments_bands[0]) == 0:
        print("No segments to plot")
        return

    N_bands = len(segments_bands)
    N_segments = len(segments_bands[0])
    N_channels = segments_bands[0].shape[1]

    if freq_bands is None:
        freq_bands = [(0, 0) for i in range(N_bands)]

    # Pick samples
    n_total = N_segments
    n_avg = min(max(100, n_samples), n_total)

    idx_avg = np.random.choice(n_total, n_avg, replace=False)
    idx_sample = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    # Downsample
    def ds(x):
        return x[..., ::downsample] if downsample > 1 else x

    segments_bands_ds = [
        ds(seg) for seg in segments_bands
    ]

    # Compute averaged segments for each band
    avg_segments = [
        segments[idx_avg].mean(axis=0)   # shape (C, T)
        for segments in segments_bands_ds
    ]

    # Sampled segments (for individual columns)
    sample_segments = [
        segments[idx_sample]  # shape (n_samples, C, T)
        for segments in segments_bands_ds
    ]

    # Time axis
    sfreq_orig = 256
    sfreq = sfreq_orig / downsample
    T = avg_segments[0].shape[-1]
    time = np.arange(T) / sfreq

    # Plot settings
    spacing = 8
    offsets = np.arange(N_channels)[::-1] * spacing

    # Colors for bands (recycled if needed)
    band_colors = ["black", "darkblue", "skyblue", "green", "orange", "purple", "brown", "cyan"]
    color_cycle = itertools.cycle(band_colors)
    band_color_list = [next(color_cycle) for _ in range(N_bands)]

    # Make figure
    fig, axes = plt.subplots(
        1, min(n_samples, N_segments) + 1,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.0}
    )

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # Spike shading window
    spike_start = int(spike_window_sec[0] * sfreq)
    spike_end   = int(spike_window_sec[1] * sfreq)

    # Tick marks
    major_ticks = np.arange(0, T, int(sfreq / 2))
    major_labels = [f"{t/sfreq:.1f}" for t in major_ticks]
    minor_ticks = np.arange(0, T, int(0.2 * sfreq))

    # Averaged segment (first column)
    ax_avg = axes[0]
    band_idx = 0 # For now, only plot low band average
    avg_seg = avg_segments[band_idx]
    color = band_color_list[band_idx]

    for ch in range(N_channels):
        ax_avg.plot(
            time, -avg_seg[ch] + offsets[ch],
            color=color, alpha=1.0, linewidth=1.0
        )

    # Shading + labels
    if plot_spike_window:
        ax_avg.axvspan(time[spike_start], time[spike_end],
                    color="red", alpha=0.15)

    for ch_idx, y in enumerate(offsets):
        ax_avg.text(time[0] - (time[-1] * 0.03), y,
                    channels_to_keep[ch_idx], ha="right", va="center", fontsize=7)

    # Ticks and grid
    for t in major_ticks:
        ax_avg.axvline(t/sfreq, color="gray", alpha=0.5, linewidth=0.6)
    for t in minor_ticks:
        ax_avg.axvline(t/sfreq, color="gray", alpha=0.25, linewidth=0.4)

    ax_avg.set_title(f"Average from {n_avg} segments", fontsize=7, fontweight="bold")
    ax_avg.set_xlim(time[0], time[-1])
    ax_avg.set_ylim(-spacing, offsets.max() + spacing)
    ax_avg.set_xticks(major_ticks / sfreq)
    ax_avg.set_xticklabels(major_labels)
    ax_avg.set_xlabel("Time (s)")
    ax_avg.set_facecolor((0.95, 1.0, 1.0))

    # Legend
    legend_labels = [f"{l}-{h} Hz" for (l, h) in freq_bands]
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in band_color_list]
    ax_avg.legend(legend_handles, legend_labels, fontsize=6, loc="upper left")

    # Individual segments
    for col, ax in enumerate(axes[1:], start=1):
        sample_idx = idx_sample[col - 1]
        timestamp = timestamps[sample_idx]

        for band_idx in range(N_bands):
            segs = sample_segments[band_idx]
            color = band_color_list[band_idx]
            seg = segs[col - 1]  # shape (C, T)
            alpha = 1.0 if band_idx == 0 else 0.4
            linewidth = 0.7 if band_idx == 0 else 0.4
            for ch in range(N_channels):
                ax.plot(time, -seg[ch] + offsets[ch],
                        color=color, alpha=alpha, linewidth=linewidth)

        # Shading
        if plot_spike_window:
            ax.axvspan(time[spike_start], time[spike_end],
                       color="red", alpha=0.15)

        # Ticks and limits
        for t in major_ticks:
            ax.axvline(t/sfreq, color="gray", alpha=0.5, linewidth=0.6)
        for t in minor_ticks:
            ax.axvline(t/sfreq, color="gray", alpha=0.25, linewidth=0.4)

        ax.set_yticks([])
        ax.set_xlim(time[0], time[-1])
        ax.set_xticks(major_ticks / sfreq)
        ax.set_xticklabels(major_labels)
        ax.set_xlabel("Time (s)")

        # Timestamp as title
        t = int(timestamp)
        days, rem = divmod(t, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        ts_str = f"d{days+1}:{hours:02d}:{minutes:02d}:{seconds:02d}"

        ax.set_title(ts_str, fontsize=8)

    # Final layout
    fig.suptitle(
        f"{patient_id} — {N_bands} frequency bands — Avg + {len(axes)-1} Individual Segments",
        fontsize=14,
        fontweight="bold"
    )
    # Output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def process_patient_edf(
    edf_path, events_json_path, output_dir, patient_id,
    channels_to_keep=[
    "Fp1", "Fp2", "F9", "F10", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
    ],
    freq_bands=[(1.0, 70.0), (70.0, 120.0)], notch_freqs=(50.0, 60.0),
    segment_length=1.0, qc_path=None, qc_dropped_path=None, n_cores=6, drop_noisy=True,
    noise_cfg=None,
    # --- flat dataset options ---
    create_flat_dataset_files=True,
    detection_csv_path=None,
    flat_output_dir=None,
    n_non_spike_segments=1000,
    flat_band=(1.0, 70.0),
    non_spike_exclusion_seconds=1.0,
    spike_detection_threshold=0.5,
    random_seed=42,
):
    """
    Process a single patient's EDF file to create EEG segments.
    """

    print(f"\n\n================================")
    print(f"Processing patient {patient_id}...")
    print(f"EDF path: {edf_path}")
    print(f"Events JSON path: {events_json_path}")
    print(f"Output directory: {output_dir}")
    print("================================\n")
    
    if mkl:
        print(f">>> Setting MKL threads to {n_cores}")
        mkl.set_num_threads(n_cores)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load EDF
    print(f">>> Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=False)
    # Get original sampling frequency
    sfreq = raw.info["sfreq"]

    # Load spike timestamps from JSON
    print(f">>> Loading spike timestamps from JSON: {events_json_path}")
    with open(events_json_path, 'r') as f:
        events_data = json.load(f)
    # Read timestamps (in seconds from start)
    spike_timestamps = np.array(events_data["spikes"])
    # Continue if no spikes
    if len(spike_timestamps) == 0:
        print(f"\033[93mWarning: No spike timestamps found for patient {patient_id}. Skipping.\033[0m")
        return None, None

    # Channel selection
    print(f">>> Selecting channels: {channels_to_keep}")
    # Check for missing channels
    present_channels = raw.info["ch_names"]
    missing_channels = [ch for ch in channels_to_keep if ch not in present_channels]
    if missing_channels:
        print(f"\033[93mWarning: The following channels are missing in the EDF and will be skipped: {missing_channels}\033[0m")
        data_channels = [ch for ch in channels_to_keep if ch in present_channels]
    else:
        data_channels = channels_to_keep
    raw.pick_channels(data_channels)

    # Now load only these channels
    print(">>> Loading data for selected channels...")
    raw.load_data()

    # Handle missing channels by adding zeros and reorder to match channels_to_keep
    if missing_channels:
        print(f">>> Adding zero-filled channels for missing: {missing_channels}")
        # Create zero-filled data for missing channels
        n_timepoints = raw.n_times
        for ch_name in missing_channels:
            # Create info for the missing channel
            info = mne.create_info([ch_name], raw.info['sfreq'], ch_types='eeg')
            # Create zero data
            zero_data = np.zeros((1, n_timepoints))
            # Create raw object for this channel
            zero_raw = mne.io.RawArray(zero_data, info)
            # Add to existing raw
            raw.add_channels([zero_raw], force_update_info=True)

    # Reorder channels to match channels_to_keep order
    print(">>> Reordering channels to match desired order...")
    raw.reorder_channels(channels_to_keep)

    # Custom chunked + parallel FIR filtering for speed and OOM safety
    print(f">>> Starting chunked FIR filtering")
    # Access data array (does not copy)
    X = raw._data                     # shape: (n_channels, n_times)
    sfreq = raw.info["sfreq"]
    # Ensure float32 for memory efficiency
    if X.dtype != np.float32:
        print(" >> Converting data to float32 for memory efficiency...")
        X[:] = X.astype(np.float32)
    # Design FIR kernels
    print(" >> Designing FIR kernels... (shared across calls)")
    # Band-pass filter kernels (FIR)
    fir_band_kernels = []
    for l_freq, h_freq in freq_bands:
        fir_bp = create_filter(
            data=None,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design="firwin",
            phase="zero",
            fir_window="hamming",
            verbose=True,
        )
        fir_band_kernels.append(fir_bp)
    # Notch filter kernels (FIR) --> 50Hz and 60Hz by default
    fir_notches = []
    notch_freqs_arr = np.atleast_1d(notch_freqs)
    for f0 in notch_freqs_arr:
        if f0 is None:
            continue
        # Define params. These are MNE standard, so stuck with these. 
        notch_width = f0 / 200.
        trans_bandwidth = 1.
        notch_low = f0 - notch_width / 2. - trans_bandwidth / 2.
        notch_high = f0 + notch_width / 2. + trans_bandwidth / 2.
        # Create notch filter kernel
        fir_notch = create_filter(
            data=None,
            sfreq=sfreq,
            l_freq=notch_high,  # Yes, this is swapped. This is apparently how MNE wants it.
            h_freq=notch_low,
            filter_length="auto",
            l_trans_bandwidth=trans_bandwidth / 2.,
            h_trans_bandwidth=trans_bandwidth / 2.,
            method="fir",
            iir_params=None,
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
        )

        fir_notches.append(fir_notch)

    if not fir_notches:
        print("\033[38;5;208m >> Warning: No notch filters created.\033[0m")

    # Helper function: apply FIR to one channel in chunks
    # ---------------------------------------------------
    def _apply_fir_chunked_1d(x_1d, fir_kernel, sfreq, chunk_sec=30.0):
        """
        Apply FIR to one channel in chunks to avoid large FFT buffers.
        x_1d: shape (n_samples,), float32/float64
        fir_kernel: 1D FIR coefficients (float64)
        """
        # Make sure each thread uses only one MKL thread
        if mkl:
            mkl.set_num_threads(1)
        # Convert to float64 for filtering and define parameters
        x = x_1d.astype(np.float64, copy=False)
        n_samples = x.shape[0]
        L = len(fir_kernel)
        pad = L - 1              # padding per side
        chunk = int(chunk_sec * sfreq)
        if chunk <= 0:
            chunk = n_samples

        # Output array
        out = np.empty_like(x, dtype=np.float64)
        # Loop over chunks
        start = 0
        while start < n_samples:
            # Define chunk boundaries
            stop = min(start + chunk, n_samples)
            # extend segment to avoid edge effects
            seg_start = max(0, start - pad)
            seg_stop = min(n_samples, stop + pad)
            # Extract segment and filter
            seg = x[seg_start:seg_stop]
            seg_filt = _overlap_add_filter(seg, fir_kernel, n_jobs=1)
            # Slice to original segment (now free of edge effects)
            offset = start - seg_start
            out[start:stop] = seg_filt[offset:offset + (stop - start)]
            # Move on to next chunk
            start = stop

        return out

    # Helper function: apply a FIR kernel to all channels in parallel, chunked
    # ------------------------------------------------------------------
    def apply_fir_chunked_all_channels(X, fir_kernel, sfreq, chunk_sec=30.0, n_jobs=1):
        """
        X: (n_channels, n_times), modified in-place (float32)
        fir_kernel: 1D FIR kernel (float64)
        """
        n_channels, _ = X.shape

        def process_channel(ch):
            x = X[ch]  # float32 view
            y = _apply_fir_chunked_1d(x, fir_kernel, sfreq, chunk_sec)
            return ch, y.astype(np.float32)

        if n_jobs == 1:
            for ch in range(n_channels):
                ch_idx, y = process_channel(ch)
                X[ch_idx, :] = y
        else:
            # Thread-based parallelism across channels. Should share memory. 
            print(f"  > Running chunked FIR across {n_channels} channels with {n_jobs} threads...")
            with parallel_backend("threading", n_jobs=n_jobs):
                results = Parallel()(
                    delayed(process_channel)(ch) for ch in range(n_channels)
                )
            for ch_idx, y in results:
                X[ch_idx, :] = y

        gc.collect()
        return X

    # Apply filters in parallel with 5-min chunks
    n_jobs_channels = min(n_cores, X.shape[0])
    chunk_sec = 300.0

    print(f" >> Applying {len(fir_notches)} notch filter(s) with {chunk_sec:.1f}s chunks...")
    for idx, fir_notch in enumerate(fir_notches, start=1):
        print(f"  > [Notch {idx}/{len(fir_notches)}]")
        X = apply_fir_chunked_all_channels(X, fir_notch, sfreq, chunk_sec=chunk_sec, n_jobs=n_jobs_channels)

    print(f" >> Applying band-pass filters with {chunk_sec:.1f}s chunks...")
    X_bands = []
    for idx, fir_bp in enumerate(fir_band_kernels, start=1):
        print(f"  > [Band-pass {idx}/{len(fir_band_kernels)}]")
        X_band = X.copy()
        X_band = apply_fir_chunked_all_channels(X_band, fir_bp, sfreq, chunk_sec=chunk_sec, n_jobs=n_jobs_channels)
        X_bands.append(X_band)
    print(">>> FIR filtering done")

    if mkl:
        print(f" >> Resetting MKL threads to {n_cores}")
        mkl.set_num_threads(n_cores)
    del X  # free memory
    gc.collect()

    # Normalization: Global robust z-score, normalizing bands separately (EXCLUDING missing channels)
    print(">>> Normalizing bands (global robust z-score)...")

    # Build mask for valid (non-missing) channels
    valid_mask = np.ones(len(channels_to_keep), dtype=bool)
    for ch in missing_channels:
        valid_mask[channels_to_keep.index(ch)] = False

    X_bands_norm = []
    chunk_size = 2_097_152  # = 2^21

    for b_idx, Xb in enumerate(X_bands):
        print(f"  > Band {b_idx+1}: computing statistics (chunked)")

        # Compute median in chunks
        medians = []
        for ch_idx in np.where(valid_mask)[0]:
            x = Xb[ch_idx]
            for start in range(0, x.size, chunk_size):
                chunk = x[start:start+chunk_size].astype(np.float32, copy=False)
                medians.append(np.median(chunk))
        median_b = np.median(np.array(medians, dtype=np.float32))
        print(f"    median={median_b}")

        # Compute MAD in chunks
        mad_vals = []
        for ch_idx in np.where(valid_mask)[0]:
            x = Xb[ch_idx]
            for start in range(0, x.size, chunk_size):
                chunk = x[start:start+chunk_size].astype(np.float32, copy=False)
                mad_vals.append(np.median(np.abs(chunk - median_b)))
        mad_b = np.median(np.array(mad_vals, dtype=np.float32))
        scale_b = mad_b * 1.4826 + 1e-8
        print(f"    MAD={mad_b}, scale={scale_b}")

        # Normalize in chunks
        print(f"  > Band {b_idx+1}: normalizing in-place")

        for ch in range(Xb.shape[0]):
            x = Xb[ch]
            for start in range(0, x.size, chunk_size):
                chunk = x[start:start+chunk_size]
                chunk -= median_b
                chunk /= scale_b

        # Store normalized band
        X_bands_norm.append(Xb.astype(np.float32, copy=False))

    X_bands = X_bands_norm
    del X_bands_norm  # free memory
    gc.collect()

    # Segmentation based on spike timestamps
    print(f">>> Segmenting data into {segment_length}s segments around spikes...")
    win = int(segment_length * sfreq)

    def extract_segment(t):
        if mkl:
            mkl.set_num_threads(1)
        center = int(t * sfreq)
        start = center - win // 2
        stop = start + win
        if stop <= X_bands[0].shape[1]:
            return [band[:, start:stop].astype(np.float32) for band in X_bands]
        return None
    
    with parallel_backend("threading", n_jobs=n_cores):
        segments = Parallel()(
            delayed(extract_segment)(t) for t in spike_timestamps
        )
    if mkl:
        print(f" >> Resetting MKL threads to {n_cores}")
        mkl.set_num_threads(n_cores)

    # --- Optional flat dataset creation (must happen before del X_bands) ---
    if create_flat_dataset_files:
        print(f">>> Flat dataset creation: {'enabled' if create_flat_dataset_files else 'disabled'}")
        if detection_csv_path is not None:
            band_idx_flat = find_band_index(freq_bands, flat_band)
            if band_idx_flat < 0:
                print(
                    f"\033[93mWarning: band {flat_band} not found in freq_bands {freq_bands}. "
                    "Skipping flat export.\033[0m"
                )
            else:
                _flat_dir = flat_output_dir if flat_output_dir is not None else Path(output_dir) / "flat_dataset"
                create_flat_dataset_for_patient(
                    X_band=X_bands[band_idx_flat],
                    sfreq=sfreq,
                    patient_id=patient_id,
                    detection_csv_path=detection_csv_path,
                    output_dir=_flat_dir,
                    segment_length=segment_length,
                    spike_detection_threshold=spike_detection_threshold,
                    n_non_spike_segments=n_non_spike_segments,
                    non_spike_exclusion_seconds=non_spike_exclusion_seconds,
                    random_seed=random_seed,
                )
        else:
            print(
                f"\033[93mWarning: flat dataset creation enabled but no detection CSV "
                f"provided for {patient_id}. Skipping flat export.\033[0m"
            )

    del X_bands  # free memory
    gc.collect()

    if noise_cfg is None:
        noise_cfg = {
            "ptp_z_thresh": 6.0,
            "flat_ptp_thresh": 0.05,
            "flat_channel_frac_thresh": 0.3,
            "abs_z_thresh": 10.0,
            "abs_z_frac_thresh": 0.01,
            "hf_ratio_z_thresh": 6.0,
        }

    # Stack segments and remove Nones
    segments_bands = [[] for _ in range(len(freq_bands))]
    kept_timestamps = []
    in_bounds_event_indices = []
    for i_evt, (t, res) in enumerate(zip(spike_timestamps, segments)):
        if res is None:
            continue
        in_bounds_event_indices.append(i_evt)
        kept_timestamps.append(t)
        for i, seg_i in enumerate(res):
            segments_bands[i].append(seg_i)
    segments_bands = [
        np.array(sb, dtype=np.float32) for sb in segments_bands
    ]
    kept_timestamps = np.array(kept_timestamps)

    # Keep a pre-rejection copy for dropped-segment QC plotting.
    if qc_dropped_path is not None:
        segments_bands_pre_reject = [sb.copy() for sb in segments_bands]
        timestamps_pre_reject = kept_timestamps.copy()
    else:
        segments_bands_pre_reject = None
        timestamps_pre_reject = None

    # Drop noisy segments after filtering and normalization.
    if drop_noisy:
        print(">>> Dropping noisy segments using robust epoch-level criteria...")
        segments_bands, kept_timestamps, noise_qc = drop_noisy_segments(
            segments_bands=segments_bands,
            timestamps=kept_timestamps,
            valid_mask=valid_mask,
            noise_cfg=noise_cfg,
        )
        print(
            f"  > Dropped {noise_qc['dropped']} / "
            f"{noise_qc['dropped'] + noise_qc['kept']} segments "
            f"({100.0 * noise_qc['drop_fraction']:.2f}%)"
        )
        print(f"  > Drop reasons: {noise_qc['reason_counts']}")
    else:
        noise_qc = {
            "kept": int(len(kept_timestamps)),
            "dropped": 0,
            "drop_fraction": 0.0,
            "reason_counts": {
                "ptp": 0,
                "flat": 0,
                "extreme_samples": 0,
                "hf_ratio": 0,
            },
            "detail": {
                "keep_mask": [True] * int(len(kept_timestamps)),
                "drop_mask": [False] * int(len(kept_timestamps)),
                "bad_ptp": [False] * int(len(kept_timestamps)),
                "bad_flat": [False] * int(len(kept_timestamps)),
                "bad_extreme": [False] * int(len(kept_timestamps)),
                "bad_hf": [False] * int(len(kept_timestamps)),
                "ptp_z": [0.0] * int(len(kept_timestamps)),
                "flat_frac": [0.0] * int(len(kept_timestamps)),
                "extreme_frac": [0.0] * int(len(kept_timestamps)),
                "hf_ratio_z": [0.0] * int(len(kept_timestamps)),
            },
        }
        print(">>> Skipping noisy segment rejection (--no-drop_noisy_segments)")

    # Write detailed per-patient rejection report.
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    kept_mask = np.array(noise_qc["detail"]["keep_mask"], dtype=bool)
    dropped_mask = np.array(noise_qc["detail"]["drop_mask"], dtype=bool)
    bad_ptp_mask = np.array(noise_qc["detail"]["bad_ptp"], dtype=bool)
    bad_flat_mask = np.array(noise_qc["detail"]["bad_flat"], dtype=bool)
    bad_extreme_mask = np.array(noise_qc["detail"]["bad_extreme"], dtype=bool)
    bad_hf_mask = np.array(noise_qc["detail"]["bad_hf"], dtype=bool)

    in_bounds_event_indices = np.array(in_bounds_event_indices, dtype=int)
    in_bounds_timestamps = np.array([spike_timestamps[i] for i in in_bounds_event_indices], dtype=float)
    in_bounds_event_index_set = set(in_bounds_event_indices.tolist())

    boundary_dropped_indices = [
        int(i) for i in range(len(spike_timestamps)) if i not in in_bounds_event_index_set
    ]

    noisy_dropped_indices = in_bounds_event_indices[dropped_mask].astype(int).tolist()
    noisy_kept_indices = in_bounds_event_indices[kept_mask].astype(int).tolist()

    report = {
        "patient_id": patient_id,
        "input_spike_count": int(len(spike_timestamps)),
        "in_bounds_segment_count": int(len(in_bounds_event_indices)),
        "boundary_dropped_count": int(len(boundary_dropped_indices)),
        "noisy_dropped_count": int(np.sum(dropped_mask)),
        "final_kept_count": int(np.sum(kept_mask)),
        "noise_rejection_enabled": bool(drop_noisy),
        "noise_cfg": noise_cfg,
        "boundary_dropped": [
            {
                "event_index": int(i),
                "timestamp_sec": float(spike_timestamps[i]),
                "reason": "out_of_bounds",
            }
            for i in boundary_dropped_indices
        ],
        "noisy_dropped": [
            {
                "event_index": int(evt_idx),
                "timestamp_sec": float(ts),
                "reasons": [
                    reason
                    for reason, cond in [
                        ("ptp", bool(bad_ptp_mask[i_local])),
                        ("flat", bool(bad_flat_mask[i_local])),
                        ("extreme_samples", bool(bad_extreme_mask[i_local])),
                        ("hf_ratio", bool(bad_hf_mask[i_local])),
                    ]
                    if cond
                ],
                "metrics": {
                    "ptp_z": float(noise_qc["detail"]["ptp_z"][i_local]),
                    "flat_frac": float(noise_qc["detail"]["flat_frac"][i_local]),
                    "extreme_frac": float(noise_qc["detail"]["extreme_frac"][i_local]),
                    "hf_ratio_z": float(noise_qc["detail"]["hf_ratio_z"][i_local]),
                },
            }
            for i_local, (evt_idx, ts) in enumerate(zip(in_bounds_event_indices, in_bounds_timestamps))
            if dropped_mask[i_local]
        ],
        "kept_segments": [
            {
                "event_index": int(evt_idx),
                "timestamp_sec": float(ts),
            }
            for i_local, (evt_idx, ts) in enumerate(zip(in_bounds_event_indices, in_bounds_timestamps))
            if kept_mask[i_local]
        ],
        "summary": {
            "noise_drop_fraction_within_in_bounds": float(np.mean(dropped_mask) if len(dropped_mask) else 0.0),
            "reason_counts": noise_qc["reason_counts"],
            "noisy_dropped_indices": noisy_dropped_indices,
            "noisy_kept_indices": noisy_kept_indices,
        },
    }

    report_path = logs_dir / f"{patient_id}_segment_rejection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f">>> Saved segment rejection report: {report_path}")

    if len(kept_timestamps) == 0:
        print(f"\033[93mWarning: No segments left after rejection for {patient_id}.\033[0m")

    # Save to .npy
    print(f">>> Saving segments to .npy file...")
    for i, band_segments in enumerate(segments_bands):
        l, h = freq_bands[i]
        out_path = os.path.join(output_dir, f"{patient_id}_spikes_{int(l)}-{int(h)}Hz.npy")
        np.save(out_path, band_segments)
        print(f"Saved band {l}-{h} Hz: {band_segments.shape} → {out_path}")

    # Generate QC plot if applicable
    if qc_path:
        print(f">>> Generating quality control plot: {qc_path}")
        quality_control_plot(
            segments_bands=segments_bands, timestamps=kept_timestamps, patient_id=patient_id,
            channels_to_keep=channels_to_keep, freq_bands=freq_bands,
            output_path=qc_path
        )

    # Generate QC plot for dropped (noisy) segments if requested.
    if qc_dropped_path:
        if segments_bands_pre_reject is None:
            print(">>> Dropped-segment QC requested, but no pre-rejection segments were cached.")
        elif np.sum(dropped_mask) == 0:
            print(">>> No noisy-dropped segments to plot for dropped QC.")
        else:
            dropped_segments_bands = [sb[dropped_mask] for sb in segments_bands_pre_reject]
            dropped_timestamps = timestamps_pre_reject[dropped_mask]
            print(f">>> Generating dropped-segment QC plot: {qc_dropped_path}")
            quality_control_plot(
                segments_bands=dropped_segments_bands,
                timestamps=dropped_timestamps,
                patient_id=f"{patient_id} (dropped)",
                channels_to_keep=channels_to_keep,
                freq_bands=freq_bands,
                output_path=qc_dropped_path,
            )

    return segments_bands, kept_timestamps

# Main entry point
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process EDF files into EEG spike segment dataset.")
    parser.add_argument("--edf_dir", type=str, required=True,
                        help="Directory containing EDF files (RESP folders allowed).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed spike .npy files.")
    parser.add_argument("--events_dir", type=str, required=True,
                        help="Directory containing JSON spike timestamp files.")
    parser.add_argument("--qc_dir", type=str, required=False, default=None,
                        help="Directory for saving QC plots (optional).")
    parser.add_argument("--log_dir", type=str, required=False, default=None,
                        help="Optional directory for log files. If omitted, logs go to output_dir.")
    parser.add_argument("--n_cores", type=int, required=False, default=6 if mkl else 1,
                        help="Number of CPU cores to use for parallel processing.")
    parser.add_argument(
        "--drop_noisy_segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop noisy segments after filtering+normalization (default: True).",
    )
    parser.add_argument(
        "--noise_ptp_z_thresh",
        type=float,
        default=6.0,
        help="Robust z-score threshold on epoch median peak-to-peak amplitude.",
    )
    parser.add_argument(
        "--noise_flat_ptp_thresh",
        type=float,
        default=0.05,
        help="Segment/channel considered flat if peak-to-peak is below this (normalized units).",
    )
    parser.add_argument(
        "--noise_flat_channel_frac_thresh",
        type=float,
        default=0.3,
        help="Drop epoch if this fraction of channels are flat.",
    )
    parser.add_argument(
        "--noise_abs_z_thresh",
        type=float,
        default=10.0,
        help="Absolute robust-z threshold for extreme samples.",
    )
    parser.add_argument(
        "--noise_abs_z_frac_thresh",
        type=float,
        default=0.01,
        help="Drop epoch if fraction of extreme samples exceeds this threshold.",
    )
    parser.add_argument(
        "--noise_hf_ratio_z_thresh",
        type=float,
        default=6.0,
        help="Robust z-score threshold on high-frequency contamination ratio.",
    )
    # --- Flat dataset arguments ---
    parser.add_argument(
        "--create_flat_dataset_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create additional flat 1-70Hz segment dataset files for spike-encoder pretraining.",
    )
    parser.add_argument(
        "--detection_csv_dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing Persyst detection CSV files with Time, Channel, and Perception columns.",
    )
    parser.add_argument(
        "--flat_output_dir",
        type=str,
        required=False,
        default=None,
        help="Directory for flat dataset outputs. Defaults to output_dir/flat_dataset.",
    )
    parser.add_argument(
        "--n_non_spike_segments",
        type=int,
        default=1000,
        help="Number of non-spike segments to sample per patient for flat dataset creation.",
    )
    parser.add_argument(
        "--spike_detection_threshold",
        type=float,
        default=0.9,
        help="Perception threshold used to label detections as thresholded spikes in flat metadata.",
    )
    parser.add_argument(
        "--non_spike_exclusion_seconds",
        type=float,
        default=1.0,
        help="Total exclusion window around each Persyst detection for non-spike sampling.",
    )
    parser.add_argument(
        "--flat_random_seed",
        type=int,
        default=42,
        help="Random seed for non-spike segment sampling.",
    )
    args = parser.parse_args()

    # Display arguments
    print(">>> [RUN] create_eeg_ds.py with the following settings:")
    print("Arguments:")
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")
    
    if mkl:
        print(f">>> Using MKL, max threads: {mkl.get_max_threads()}")
        print(f">>> Number of cores set: {args.n_cores}")
        if args.n_cores > mkl.get_max_threads():
            print(
                f"\033[93mWarning: n_cores ({args.n_cores}) exceeds MKL max threads "
                f"({mkl.get_max_threads()}). This may lead to suboptimal performance.\033[0m"
            )
    else:
        print(">>> Using MKL: False")

    # Determine log location
    log_dir = args.log_dir if args.log_dir is not None else args.output_dir
    setup_logger(log_dir)

    # Create QC directory if needed
    if args.qc_dir:
        os.makedirs(args.qc_dir, exist_ok=True)

    # Gather EDF files
    edf_list = glob(os.path.join(args.edf_dir, "RESP*", "*.edf"))
    print(f">>> Found {len(edf_list)} EDF files.")

    # Dataset-level aggregation for drop statistics.
    dataset_drop_summary = []
    warning_drop_threshold = 0.25

    print(f">>> Flat dataset creation: {'enabled' if args.create_flat_dataset_files else 'disabled'}")

    for edf_file in edf_list:
        patient_id = Path(edf_file).stem.split("_")[-1]
        events_json_file = os.path.join(args.events_dir, f"{patient_id}_events.json")

        if os.path.exists(events_json_file):
            qc_path = None
            qc_dropped_path = None
            if args.qc_dir:
                qc_path = os.path.join(args.qc_dir, f"quality_control_{patient_id}.png")
                dropped_qc_dir = os.path.join(args.qc_dir, "dropped")
                os.makedirs(dropped_qc_dir, exist_ok=True)
                qc_dropped_path = os.path.join(dropped_qc_dir, f"quality_control_dropped_{patient_id}.png")

            # Resolve detection CSV for flat dataset creation
            detection_csv_file = None
            if args.create_flat_dataset_files and args.detection_csv_dir is not None:
                candidate_csvs = [
                    Path(args.detection_csv_dir) / f"{patient_id}.csv",
                    Path(args.detection_csv_dir) / f"{patient_id}_detections.csv",
                    Path(args.detection_csv_dir) / f"{patient_id}_spikes.csv",
                ]
                for c in candidate_csvs:
                    if c.exists():
                        detection_csv_file = c
                        break
                if detection_csv_file is None:
                    print(
                        f"\033[93mWarning: no detection CSV found for {patient_id} in "
                        f"{args.detection_csv_dir}. Skipping flat export.\033[0m"
                    )
                else:
                    print(f">>> Using detection CSV: {detection_csv_file}")

            segments, spike_timestamps = process_patient_edf(
                edf_path=edf_file,
                events_json_path=events_json_file,
                output_dir=args.output_dir,
                patient_id=patient_id,
                qc_path=qc_path,
                qc_dropped_path=qc_dropped_path,
                n_cores=args.n_cores,
                drop_noisy=args.drop_noisy_segments,
                noise_cfg={
                    "ptp_z_thresh": args.noise_ptp_z_thresh,
                    "flat_ptp_thresh": args.noise_flat_ptp_thresh,
                    "flat_channel_frac_thresh": args.noise_flat_channel_frac_thresh,
                    "abs_z_thresh": args.noise_abs_z_thresh,
                    "abs_z_frac_thresh": args.noise_abs_z_frac_thresh,
                    "hf_ratio_z_thresh": args.noise_hf_ratio_z_thresh,
                },
                create_flat_dataset_files=args.create_flat_dataset_files,
                detection_csv_path=detection_csv_file,
                flat_output_dir=args.flat_output_dir,
                n_non_spike_segments=args.n_non_spike_segments,
                spike_detection_threshold=args.spike_detection_threshold,
                non_spike_exclusion_seconds=args.non_spike_exclusion_seconds,
                random_seed=args.flat_random_seed,
            )
            del segments, spike_timestamps  # clear memory
            gc.collect()

            # Collect per-subject drop fraction from saved subject report.
            subject_report_path = Path(args.output_dir) / "logs" / f"{patient_id}_segment_rejection_report.json"
            if subject_report_path.exists():
                try:
                    with open(subject_report_path, "r") as f:
                        subject_report = json.load(f)
                    drop_fraction = float(
                        subject_report.get("summary", {}).get("noise_drop_fraction_within_in_bounds", 0.0)
                    )
                    dataset_drop_summary.append(
                        {
                            "patient_id": patient_id,
                            "drop_fraction": drop_fraction,
                            "in_bounds_segment_count": int(subject_report.get("in_bounds_segment_count", 0)),
                            "noisy_dropped_count": int(subject_report.get("noisy_dropped_count", 0)),
                            "final_kept_count": int(subject_report.get("final_kept_count", 0)),
                        }
                    )
                except Exception as e:
                    print(
                        f"\033[93mWarning: Could not parse subject drop report for {patient_id}: {e}\033[0m"
                    )
            else:
                print(
                    f"\033[93mWarning: Subject drop report not found for {patient_id} at {subject_report_path}\033[0m"
                )

        else:
            print(f">>> Events JSON not found for patient {patient_id}, skipping.")

    # Write dataset-level drop summary and warning file.
    logs_dir = Path(args.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = logs_dir / "dataset_drop_summary.json"
    dataset_summary_payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "warning_drop_threshold": warning_drop_threshold,
        "n_subjects": len(dataset_drop_summary),
        "subjects": sorted(dataset_drop_summary, key=lambda x: x["patient_id"]),
    }
    with open(dataset_summary_path, "w") as f:
        json.dump(dataset_summary_payload, f, indent=2)
    print(f">>> Saved dataset-level drop summary: {dataset_summary_path}")

    warning_subjects = [
        s for s in dataset_drop_summary if s.get("drop_fraction", 0.0) > warning_drop_threshold
    ]
    warning_path = logs_dir / "warning_drop_ratio.txt"
    with open(warning_path, "w") as f:
        f.write(f"Subjects with drop_fraction > {warning_drop_threshold:.2f}\n")
        if not warning_subjects:
            f.write("None\n")
        else:
            for s in sorted(warning_subjects, key=lambda x: x["patient_id"]):
                f.write(f"{s['patient_id']}\t{s['drop_fraction']:.4f}\n")
    print(f">>> Saved warning drop-ratio file: {warning_path}")
