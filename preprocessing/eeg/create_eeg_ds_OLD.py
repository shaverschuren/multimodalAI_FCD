"""
create_eeg_ds.py
Create EEG dataset from EDF files with channel selection, filtering, and segmentation of spike segments.
Assumes that bad/missing channels have been interpolated and data has been re-referenced earlier.
Requires extracted detection timestamps (integer seconds from start of recording), see extract_event_timestamps.py.
Saves processed segments as NumPy arrays.

For the filtering and segment extraction, we're doing some parallelisation on Snellius. 
Also, for OOM purposes, we're doing chunked FIR filtering to avoid large FFT buffers.

Default bandpass: 1.0-70Hz
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
import mkl
import mne
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from mne.filter import create_filter, _overlap_add_filter

matplotlib.use("Agg")

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
        freq_bands = [f"Band {i}" for i in range(N_bands)]

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
    segment_length=1.0, qc_path=None, n_cores=6
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
    print(f" >> Resetting MKL threads to {n_cores}")
    mkl.set_num_threads(n_cores)
    del X_bands  # free memory
    gc.collect()

    # Stack segments and remove Nones
    segments_bands = [[] for _ in range(len(freq_bands))]
    for res in segments:
        if res is None: continue
        for i, seg_i in enumerate(res):
            segments_bands[i].append(seg_i)
    segments_bands = [
    np.array(sb, dtype=np.float32) for sb in segments_bands
    ]

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
            segments_bands=segments_bands, timestamps=spike_timestamps, patient_id=patient_id,
            channels_to_keep=channels_to_keep, freq_bands=freq_bands,
            output_path=qc_path
        )

    return segments_bands, spike_timestamps

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
    parser.add_argument("--n_cores", type=int, required=False, default=6,
                        help="Number of CPU cores to use for parallel processing.")
    args = parser.parse_args()

    # Display arguments
    print(">>> [RUN] create_eeg_ds.py with the following settings:")
    print("Arguments:")
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")
    
    print(f">>> Using MKL, max threads: {mkl.get_max_threads()}")

    # Determine log location
    log_dir = args.log_dir if args.log_dir is not None else args.output_dir
    setup_logger(log_dir)

    # Create QC directory if needed
    if args.qc_dir:
        os.makedirs(args.qc_dir, exist_ok=True)

    # Gather EDF files
    edf_list = glob(os.path.join(args.edf_dir, "RESP*", "*.edf"))
    print(f">>> Found {len(edf_list)} EDF files.")

    for edf_file in edf_list:
        patient_id = Path(edf_file).stem.split("_")[-1]
        events_json_file = os.path.join(args.events_dir, f"{patient_id}_events.json")

        if os.path.exists(events_json_file):
            qc_path = None
            if args.qc_dir:
                qc_path = os.path.join(args.qc_dir, f"quality_control_{patient_id}.png")

            segments, spike_timestamps = process_patient_edf(
                edf_path=edf_file,
                events_json_path=events_json_file,
                output_dir=args.output_dir,
                patient_id=patient_id,
                qc_path=qc_path,
                n_cores=args.n_cores
            )
            del segments, spike_timestamps  # clear memory
            gc.collect()

        else:
            print(f">>> Events JSON not found for patient {patient_id}, skipping.")
