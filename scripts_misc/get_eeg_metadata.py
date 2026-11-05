import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from statistics import mean, median


BIDS_DIR = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\BIDS_dataset"
)
BASELINE_PATH = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual_multimodal.tsv"
)
TEMP_PATIENT_DIR = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\_tmp\raw_EEG_KNF\EDFdata_plus"
)
CONVERSION_LOG_DIR = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata\conversion_logs"
)
SUBMONTAGE_CHANNELS = {
    channel.casefold() for channel in (
        "F9", "F10", "Fp2", "Fp1", "F8", "F4", "Fz", "F3", "F7", "T8", "C4",
        "Cz", "C3", "T7", "P8", "P4", "Pz", "P3", "P7", "O2", "O1",
    )
}


def read_interpolated_channels(patient_ids):
    patient_recordings = defaultdict(list)
    for patient_id in sorted(patient_ids):
        subject_id = patient_id.removeprefix("sub-")
        for conversion_type in ("edf2edf", "trc2edf"):
            log_path = CONVERSION_LOG_DIR / f"{subject_id}_{conversion_type}_conversion.tsv"
            if not log_path.is_file():
                continue
            with log_path.open(newline="", encoding="utf-8-sig") as log_file:
                reader = csv.DictReader(log_file, delimiter="\t")
                if "interpolated_channels" not in (reader.fieldnames or []):
                    raise ValueError(f"Missing interpolated_channels column: {log_path}")
                for row in reader:
                    channels = {
                        channel.strip().casefold()
                        for channel in (row["interpolated_channels"] or "").split(",")
                        if channel.strip()
                    }
                    patient_recordings[patient_id].append(channels)
    return patient_recordings


def print_count_statistics(label, counts):
    print(
        f"  {label}: total {sum(counts)}; mean {mean(counts):.2f}; "
        f"median {median(counts):.1f}; range {min(counts)}-{max(counts)}"
    )


def print_interpolation_statistics(patient_ids):
    patient_recordings = read_interpolated_channels(patient_ids)
    print(f"Patients with conversion recordings: {len(patient_recordings)}/{len(patient_ids)}")
    print(f"Baseline patients without conversion recordings: {len(patient_ids - patient_recordings.keys())}")
    print("Conversion-log recording = one input-file/block row; missing logs are excluded, not counted as zero.")
    if not patient_recordings:
        return

    for label, montage in (("All electrodes", None), ("21-channel 10-20 + F9/F10", SUBMONTAGE_CHANNELS)):
        recordings = []
        patient_totals = []
        patient_unique_counts = []
        recording_channels = Counter()
        patient_channels = Counter()
        for channel_sets in patient_recordings.values():
            selected = [channels if montage is None else channels & montage for channels in channel_sets]
            unique_channels = set().union(*selected)
            counts = [len(channels) for channels in selected]
            recordings.extend(counts)
            patient_totals.append(sum(counts))
            patient_unique_counts.append(len(unique_channels))
            for channels in selected:
                recording_channels.update(channels)
            patient_channels.update(unique_channels)

        affected_recordings = sum(count > 0 for count in recordings)
        affected_patients = sum(count > 0 for count in patient_totals)
        print(f"Interpolation ({label}):")
        print(
            f"  Recordings with interpolation: {affected_recordings}/{len(recordings)} "
            f"({affected_recordings / len(recordings):.1%})"
        )
        print(
            f"  Patients with interpolation: {affected_patients}/{len(patient_totals)} "
            f"({affected_patients / len(patient_totals):.1%})"
        )
        print_count_statistics("Interpolated channels per recording (including zeros)", recordings)
        affected_counts = [count for count in recordings if count > 0]
        if affected_counts:
            print(
                f"  Interpolated channels per affected recording: median {median(affected_counts):.1f}; "
                f"range {min(affected_counts)}-{max(affected_counts)}"
            )
        else:
            print("  Interpolated channels per affected recording: median N/A; range N/A (none affected)")
        print_count_statistics("Per patient, summed across recordings (including zeros)", patient_totals)
        print_count_statistics("Per patient, unique electrodes (including zeros)", patient_unique_counts)
        print("  Per electrode: recordings / patients with interpolation")
        for channel, count in recording_channels.most_common():
            print(
                f"    {channel}: {count}/{len(recordings)} ({count / len(recordings):.1%}) / "
                f"{patient_channels[channel]}/{len(patient_totals)} "
                f"({patient_channels[channel] / len(patient_totals):.1%})"
            )


def main():
    with BASELINE_PATH.open(newline="", encoding="utf-8-sig") as baseline_file:
        patient_ids = {row["participant_id"] for row in csv.DictReader(baseline_file, delimiter="\t")}
    # patient_ids &= {f"sub-{folder.name}" for folder in TEMP_PATIENT_DIR.glob("RESP*") if folder.is_dir()}

    patient_durations = defaultdict(float)
    patient_frequencies = defaultdict(set)

    print_interpolation_statistics(patient_ids)

    for patient_id in tqdm(sorted(patient_ids)):
        for sidecar_path in sorted((BIDS_DIR / patient_id).rglob("*_eeg.json")):
            with sidecar_path.open(encoding="utf-8-sig") as sidecar_file:
                metadata = json.load(sidecar_file)
            frequency = float(metadata["SamplingFrequency"])
            duration = float(metadata["RecordingDuration"]) / 60
            patient_durations[patient_id] += duration
            patient_frequencies[patient_id].add(frequency)

    durations = list(patient_durations.values())
    print(f"Patients with EEG sidecars: {len(durations)}/{len(patient_ids)}")
    print(f"Filtered baseline patients without EEG sidecars: {len(patient_ids - patient_durations.keys())}")
    if not durations:
        return

    print(f"Total recording duration: {sum(durations) / 60:.2f} h")
    print(
        f"Per patient (summed), minutes: mean {mean(durations):.1f}; median {median(durations):.1f}; "
        f"range {min(durations):.1f}-{max(durations):.1f}"
    )
    frequencies = Counter(tuple(sorted(values)) for values in patient_frequencies.values())
    print("Sampling frequencies (per patient):")
    for combination, count in frequencies.most_common():
        label = ", ".join(f"{frequency:g} Hz" for frequency in combination)
        if len(combination) > 1:
            label = f"Mixed ({label})"
        print(f"  {label}: {count} ({count / len(durations):.1%})")


if __name__ == "__main__":
    main()