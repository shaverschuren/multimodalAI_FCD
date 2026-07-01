"""Short script for baseline table filtering and summary stats."""

import csv
from collections import Counter
from statistics import mean, stdev

# Paths
TXT_FILE = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\full_test_subjects.txt"
TSV_FILE = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual_with_surgery_type.tsv"
OUTPUT_FILE = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual_full.tsv"


def _pct(count: int, total: int) -> float:
    return (100.0 * count / total) if total else 0.0


def _print_distribution(title: str, counter: Counter, total: int) -> None:
    print(f"\n{title}:")
    for key, count in sorted(counter.items(), key=lambda x: (-x[1], str(x[0]))):
        print(f"  {key}: {count} ({_pct(count, total):.1f}%)")

def _fcd_major_group(fcd_type: str) -> str:
    value = fcd_type.upper()
    if "FCD II" in value:
        return "II"
    if "FCD I" in value:
        return "I"
    return "Other/Unspecified"

# Read subject IDs from txt file
with open(TXT_FILE, 'r') as f:
    subject_ids = set(line.strip() for line in f if line.strip())

# Read TSV, filter by subject IDs, and write to new file
filtered_lines = []
found_ids = set()
with open(TSV_FILE, 'r') as f:
    for i, line in enumerate(f):
        # Keep header row
        if i == 0:
            filtered_lines.append(line)
            continue
        
        # Extract subject ID from TSV (format: sub-RESPxxxx)
        parts = line.split('\t')
        if parts:
            tsv_id = parts[0].strip()
            # Convert sub-RESPxxxx to RESPxxxx for matching
            if tsv_id.startswith('sub-'):
                subject_id = tsv_id[4:]  # Remove 'sub-' prefix
            else:
                subject_id = tsv_id
            
            if subject_id in subject_ids:
                filtered_lines.append(line)
                found_ids.add(subject_id)

# Check if all subject IDs from txt were found in TSV
missing_ids = subject_ids - found_ids
if missing_ids:
    raise ValueError(f"Subject IDs not found in TSV: {sorted(missing_ids)}")

# Write filtered lines to output file
with open(OUTPUT_FILE, 'w') as f:
    f.writelines(filtered_lines)

print(f"Filtered {len(filtered_lines) - 1} subjects from baseline_manual.tsv")

# Read filtered TSV to compute requested summary stats.
rows = []
with open(OUTPUT_FILE, "r", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

total_n = len(rows)

ages = []
for row in rows:
    value = row.get("age", "").strip()
    if value and value.lower() != "n/a":
        try:
            ages.append(float(value))
        except ValueError:
            continue

if ages:
    age_mean = mean(ages)
    age_std = stdev(ages) if len(ages) > 1 else 0.0
    print(f"\nAge: mean={age_mean:.2f}, std={age_std:.2f} (n={len(ages)})")
else:
    print("\nAge: no valid age values found")

sex_counts = Counter(row.get("sex", "").strip().upper() or "Missing" for row in rows)
male_count = sex_counts.get("M", 0)
female_count = sex_counts.get("F", 0)
print("\nSex:")
print(f"  Male: {male_count} ({_pct(male_count, total_n):.1f}%)")
print(f"  Female: {female_count} ({_pct(female_count, total_n):.1f}%)")
other_sex = total_n - male_count - female_count
if other_sex > 0:
    print(f"  Other/Missing: {other_sex} ({_pct(other_sex, total_n):.1f}%)")

fcd_subtypes = Counter(row.get("fcd_type", "").strip() or "Missing" for row in rows)
fcd_major = Counter(_fcd_major_group(k) for k in (row.get("fcd_type", "").strip() for row in rows))
print("\nFCD type (major):")
for key in ["I", "II", "Other/Unspecified"]:
    count = fcd_major.get(key, 0)
    if count > 0:
        print(f"  {key}: {count} ({_pct(count, total_n):.1f}%)")
_print_distribution("FCD subtype", fcd_subtypes, total_n)

lobe_counts = Counter(row.get("resection_lobe", "").strip() or "Missing" for row in rows)
_print_distribution("Resection lobe", lobe_counts, total_n)

side_counts = Counter(row.get("resection_side", "").strip() or "Missing" for row in rows)
_print_distribution("Resection side", side_counts, total_n)

mri_counts = Counter(row.get("mri_presurgical", "").strip() or "Missing" for row in rows)
_print_distribution("MRI presurgical", mri_counts, total_n)

ilae_counts = Counter(row.get("outcome_ilae", "").strip() or "Missing" for row in rows)
_print_distribution("ILAE outcome", ilae_counts, total_n)

surgery_type_counts = Counter(
    (row.get("surgery_type") or row.get("surgery type") or "").strip() or "Missing"
    for row in rows
)
_print_distribution("Surgery type", surgery_type_counts, total_n)
