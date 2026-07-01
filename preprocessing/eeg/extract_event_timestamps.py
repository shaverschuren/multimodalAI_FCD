"""
extract_event_timestamps.py

Extract spike timestamps from Persyst spike-list CSV files.
Each CSV row represents a single spike detection event.
We also include Perception-based filtering.

Author: Sjors Verschuren
Updated: December 2025
"""

from glob import glob
import json
import pandas as pd
import os
import numpy as np

from util.config import get_data_root

# ---------------------------------------------------------
# Only include spikes with Perception >= this value:
PERCEPTION_THRESHOLD = 0.90
# ---------------------------------------------------------
# Paths — derived from config.json data_root when available.
_data_root = get_data_root()
_persyst_root = _data_root / "preprocessing" / "eeg" / "persyst" if _data_root else None
SPIKE_EXPORT_DIR: str = str(_persyst_root / "spike_exports") if _persyst_root else "spike_exports"
TREND_EXPORT_DIR: str = str(_persyst_root / "trend_exports") if _persyst_root else "trend_exports"
OUTPUT_DIR: str = (
    str(_data_root / "preprocessing" / "eeg" / "spikes" / "timestamps")
    if _data_root else "timestamps"
)
# ---------------------------------------------------------

def time_to_seconds(t):
    """
    Convert strings like "d1 00:03:04.8699" or "d2 12:15:00.1000"
    into absolute seconds since recording start.
    """
    day_prefix, clock_str = t.split(" ", 1)

    # Extract numeric day index (e.g., "d2" -> 2)
    day_idx = int(day_prefix[1:])

    # Convert HH:MM:SS.xxx to seconds
    clock_seconds = pd.to_timedelta(clock_str).total_seconds()

    # Apply 24h offset for multi-day recordings
    day_offset = (day_idx - 1) * 24 * 3600

    return day_offset + clock_seconds


def extract_spikes_list_format(df, perception_threshold=0.0):
    """
    Extract spike timestamps from Persyst list-style CSV.

    Expected columns:
        Time, Channel, Perception, Sign, Duration, Height, Angle, Group

    Returns:
        {"spikes": [timestamps_in_seconds]}
    """

    # Filter by perception threshold
    df_filtered = df[df["Perception"] >= perception_threshold]

    # Convert "Time" to seconds
    spike_times = df_filtered["Time"].apply(time_to_seconds)

    return {"spikes": spike_times.to_list()}

def load_persyst_trend_csv(path):
    # Read lines to find header
    with open(path, "r") as f:
        lines = f.readlines()

    data_start_line = None
    for i, line in enumerate(lines):
        if line.startswith("ClockDateTime"):
            data_start_line = i
            break

    if data_start_line is None:
        raise ValueError("Couldn't find data header ('ClockDateTime') in file.")

    # Load the data part into a DataFrame
    df = pd.read_csv(path, skiprows=data_start_line)

    return df

def extract_trend_events(df, spike_col_name, seizure_col_name):
    """
    spike_col_name: column containing spike counts (e.g. "I3_1")
    seizure_col_name: column containing seizure notifications (e.g. "I6_1")
    """

    time_seconds = df["Time"]

    # --- Extract spike timestamps ---
    spike_timestamps = time_seconds[df[spike_col_name] > 0].to_list()

    # --- Extract seizure timestamps ---
    seizure_flag = df[seizure_col_name].fillna(0)

    # Start: where flag goes 0 → 1
    seizure_start_idx = (seizure_flag.shift(1, fill_value=0) == 0) & (seizure_flag == 1)

    # Stop: where flag goes 1 → 0
    seizure_stop_idx  = (seizure_flag.shift(1, fill_value=0) == 1) & (seizure_flag == 0)

    seizure_start_times = time_seconds[seizure_start_idx].to_list()
    seizure_stop_times  = time_seconds[seizure_stop_idx].to_list()

    return {
        # "spikes": spike_timestamps,
        "seizure_starts": seizure_start_times,
        "seizure_stops": seizure_stop_times
    }

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_spikes = []
    n_seizures = []

    print(f"Using perception threshold: {PERCEPTION_THRESHOLD}")

    for csv in glob(os.path.join(SPIKE_EXPORT_DIR, "*.csv")):

        # Build subject ID from filename
        subjid = os.path.basename(csv).replace(".csv", "")

        # Load list-style spike CSV
        df_spikes = pd.read_csv(csv)
        # Load trend csv
        df_trend = load_persyst_trend_csv(
            os.path.join(
                TREND_EXPORT_DIR,
                f"event_export_{subjid}.csv"
            )
        )

        # Extract spike timestamps
        spikes = extract_spikes_list_format(
            df_spikes,
            perception_threshold=PERCEPTION_THRESHOLD
        )
        # Extract seizure timestamps from trend data
        trend_events = extract_trend_events(
            df_trend,
            spike_col_name="I3_1",
            seizure_col_name="I2_1"
        )

        # Combine events
        events = {
            **spikes,
            **trend_events
        }
        events["subject_id"] = subjid

        # Output JSON
        out_json = os.path.join(OUTPUT_DIR, f"{subjid}_events.json")
        with open(out_json, "w") as f:
            json.dump(events, f, indent=4)

        print(f"{subjid}: {len(events['spikes'])} spikes kept")
        print(f"{subjid}: {len(events['seizure_starts'])} seizures detected")
        n_spikes.append(len(events["spikes"]))
        n_seizures.append(len(events["seizure_starts"]))

    # Summary
    if len(n_spikes) > 0:
        print("----- Summary -----")
        print(f"Spikes - Mean: {np.mean(n_spikes):.1f}, "
              f"Median: {np.median(n_spikes):.1f}, "
              f"Std: {np.std(n_spikes):.1f}, "
              f"Min: {np.min(n_spikes)}, "
              f"Max: {np.max(n_spikes)}")
    else:
        print("No CSV files found.")
