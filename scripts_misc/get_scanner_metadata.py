import csv
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import pydicom


CSV_PATH = Path(
	r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\mri\ria_pull\pre_op_scans_manual_select_final.csv"
)
BASELINE_PATH = Path(
	r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual_multimodal.tsv"
)


def main():
	with BASELINE_PATH.open(newline="", encoding="utf-8-sig") as baseline_file:
		patient_ids = {row["participant_id"] for row in csv.DictReader(baseline_file, delimiter="\t")}

	scanners = Counter()
	strengths = Counter()
	skipped = 0

	with CSV_PATH.open(newline="", encoding="utf-8-sig") as csv_file:
		for row in tqdm(csv.DictReader(csv_file, delimiter=";"), total=sum(1 for _ in open(CSV_PATH, newline="", encoding="utf-8-sig"))):
			if f"sub-{row['patient_id']}" not in patient_ids:
				continue
			folder = CSV_PATH.parent / "raw" / row["Type"] / row["session"] / row["scan"] / "DICOM"
			dicom_file = next(folder.glob("*.dcm"), None)
			if dicom_file is None:
				skipped += 1
				continue
			try:
				header = pydicom.dcmread(dicom_file, stop_before_pixels=True)
			except (OSError, pydicom.errors.InvalidDicomError):
				skipped += 1
				continue

			manufacturer = str(header.get("Manufacturer", "")).strip()
			model = str(header.get("ManufacturerModelName", "")).strip()
			scanners[f"{manufacturer} {model}".strip() or "Unknown"] += 1
			strength = header.get("MagneticFieldStrength")
			strengths[f"{float(strength):g} T" if strength is not None else "Unknown"] += 1
			if strength is not None and float(strength) < 1.5:
				tqdm.write(f"Warning: Low field strength ({strength} T) for file {dicom_file}")

	total = sum(scanners.values())
	print(f"Scans read: {total}; skipped (missing/unreadable DICOM): {skipped}")
	for title, counts in (("Scanner types", scanners), ("Field strengths", strengths)):
		print(f"\n{title} (per scan):")
		for label, count in counts.most_common():
			print(f"  {label}: {count} ({count / total:.1%})")


if __name__ == "__main__":
	main()
