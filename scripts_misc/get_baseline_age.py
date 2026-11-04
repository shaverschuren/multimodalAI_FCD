import csv
from pathlib import Path
from statistics import mean, median, stdev


BASELINE_PATH = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual_multimodal.tsv"
)


def main():
    with BASELINE_PATH.open(newline="", encoding="utf-8-sig") as baseline_file:
        rows = list(csv.DictReader(baseline_file, delimiter="\t"))
    ages = [float(row["age"]) for row in rows if row["age"].strip()]
    print(f"Patients: {len(rows)}; available ages: {len(ages)}")
    if ages:
        deviation = f"{stdev(ages):.1f}" if len(ages) > 1 else "n/a"
        num_pediatric = sum(1 for age in ages if age < 18)
        print(f"Pediatric patients (age < 18): {num_pediatric}")
        print(f"Age (years): mean {mean(ages):.1f}; SD {deviation}; median {median(ages):.1f}; range {min(ages):g}-{max(ages):g}")


if __name__ == "__main__":
    main()