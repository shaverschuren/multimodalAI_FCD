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


def main():
    with BASELINE_PATH.open(newline="", encoding="utf-8-sig") as baseline_file:
        patient_ids = {row["participant_id"] for row in csv.DictReader(baseline_file, delimiter="\t")}
    patient_ids &= {f"sub-{folder.name}" for folder in TEMP_PATIENT_DIR.glob("RESP*") if folder.is_dir()}

    patient_durations = defaultdict(float)
    patient_frequencies = defaultdict(set)

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