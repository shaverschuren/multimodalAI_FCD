import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev

from scipy import stats


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocessing" / "gt"


def load_dice_by_patient(path):
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    dice_by_patient = {}
    for record in records:
        patient_id = record["patient_id"]
        if patient_id in dice_by_patient:
            raise ValueError(f"Duplicate patient_id {patient_id!r} in {path}")
        dice_by_patient[patient_id] = float(record["dice"])
    return dice_by_patient


def main():
    parser = argparse.ArgumentParser(
        description="Compare paired Dice scores before and after harmonisation."
    )
    parser.add_argument(
        "unharmonised",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR / "comparison_summary_unharmonised.json",
    )
    parser.add_argument(
        "harmonised",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR / "comparison_summary_harmonised.json",
    )
    args = parser.parse_args()

    unharmonised = load_dice_by_patient(args.unharmonised)
    harmonised = load_dice_by_patient(args.harmonised)
    patient_ids = sorted(unharmonised.keys() & harmonised.keys())
    if not patient_ids:
        raise ValueError("The two files have no patient IDs in common.")

    differences = [harmonised[patient_id] - unharmonised[patient_id] for patient_id in patient_ids]

    print("Patient ID       Unharmonised  Harmonised  Difference")
    for patient_id, difference in zip(patient_ids, differences):
        print(
            f"{patient_id:<15} {unharmonised[patient_id]:>12.4f}  "
            f"{harmonised[patient_id]:>10.4f}  {difference:>+10.4f}"
        )

    count = len(differences)
    average_difference = mean(differences)
    print(f"\nPaired patients: {count}")
    print(f"Mean difference: {average_difference:+.4f}")
    print(f"Median difference: {median(differences):+.4f}")
    print(f"SD of differences: {stdev(differences):.4f}" if count > 1 else "SD of differences: n/a")
    print(
        "Improved / unchanged / worsened: "
        f"{sum(value > 0 for value in differences)} / "
        f"{sum(value == 0 for value in differences)} / "
        f"{sum(value < 0 for value in differences)}"
    )

    if count > 1:
        margin = stats.t.ppf(0.975, df=count - 1) * stats.sem(differences)
        print(
            f"Mean difference 95% CI: "
            f"[{average_difference - margin:+.4f}, {average_difference + margin:+.4f}]"
        )
        t_result = stats.ttest_1samp(differences, popmean=0)
        print(f"Paired t-test: t={t_result.statistic:.4f}, p={t_result.pvalue:.6g}")

    nonzero_differences = [value for value in differences if value != 0]
    if nonzero_differences:
        wilcoxon_result = stats.wilcoxon(nonzero_differences, alternative="two-sided")
        print(f"Wilcoxon signed-rank: W={wilcoxon_result.statistic:.4f}, p={wilcoxon_result.pvalue:.6g}")
    else:
        print("Wilcoxon signed-rank: all differences are zero (p=1)")

    only_unharmonised = sorted(unharmonised.keys() - harmonised.keys())
    only_harmonised = sorted(harmonised.keys() - unharmonised.keys())
    if only_unharmonised or only_harmonised:
        print(
            f"Unpaired records excluded: {len(only_unharmonised)} unharmonised, "
            f"{len(only_harmonised)} harmonised"
        )


if __name__ == "__main__":
    main()