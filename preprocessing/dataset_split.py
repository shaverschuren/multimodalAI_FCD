import random
from pathlib import Path
import pandas as pd
import json

from util.config import get_data_root

_data_root = get_data_root()
SUBJECT_LIST_CSV: Path = (
    _data_root / "selection" / "selected_summary.csv"
    if _data_root
    else Path("selected_summary.csv")
)
K_FOLD_PATH: Path = (
    _data_root / "preprocessing" / "k_fold_splits.json"
    if _data_root
    else Path("k_fold_splits.json")
)

def _closest_integer_occurrences(k_folds: int, target_ratio: float) -> int:
    """Return integer fold occurrences per subject closest to target ratio."""
    return min(k_folds, max(0, int(round(k_folds * target_ratio))))


def _choose_occurrences(k_folds: int, ratios: dict[str, float]) -> tuple[int, int, int]:
    """
    Decide how many folds each subject should appear in train/val/test.

    The goal is to stay close to requested ratios and, if feasible, include every
    subject in validation and test at least once.
    """
    train_target = ratios["train"]
    val_target = ratios["val"]
    test_target = ratios["test"]

    val_occ = _closest_integer_occurrences(k_folds, val_target)
    test_occ = _closest_integer_occurrences(k_folds, test_target)

    # If possible, ensure each subject appears in val and test at least once.
    if k_folds >= 3:
        val_occ = max(1, val_occ)
        test_occ = max(1, test_occ)

    # Keep enough room for at least one training fold.
    if val_occ + test_occ >= k_folds:
        reduction = val_occ + test_occ - (k_folds - 1)
        while reduction > 0 and (val_occ > 1 or test_occ > 1):
            # Reduce the split currently farthest from target once removed.
            val_gap = abs((val_occ / k_folds) - val_target)
            test_gap = abs((test_occ / k_folds) - test_target)
            if val_occ > 1 and (val_gap >= test_gap or test_occ == 1):
                val_occ -= 1
            elif test_occ > 1:
                test_occ -= 1
            else:
                break
            reduction -= 1

    train_occ = k_folds - val_occ - test_occ
    return train_occ, val_occ, test_occ


def _validate_no_leakage_and_coverage(
    folds: dict[str, dict[str, list[str]]],
    all_subjects: set[str],
    expected_train_occ: int,
    expected_val_occ: int,
    expected_test_occ: int,
) -> None:
    """Validate fold splits are leakage-free and preserve expected occurrence counts."""
    train_counts = {sid: 0 for sid in all_subjects}
    val_counts = {sid: 0 for sid in all_subjects}
    test_counts = {sid: 0 for sid in all_subjects}

    for fold_name, fold_data in folds.items():
        train_set = set(fold_data["train_ids"])
        val_set = set(fold_data["val_ids"])
        test_set = set(fold_data["test_ids"])

        # No leakage: a subject may only be in one split per fold.
        if train_set.intersection(val_set):
            overlap = sorted(train_set.intersection(val_set))[:10]
            raise ValueError(f"Leakage in {fold_name}: train/val overlap detected: {overlap}")
        if train_set.intersection(test_set):
            overlap = sorted(train_set.intersection(test_set))[:10]
            raise ValueError(f"Leakage in {fold_name}: train/test overlap detected: {overlap}")
        if val_set.intersection(test_set):
            overlap = sorted(val_set.intersection(test_set))[:10]
            raise ValueError(f"Leakage in {fold_name}: val/test overlap detected: {overlap}")

        # Every subject should appear exactly once in this fold.
        union_set = train_set.union(val_set).union(test_set)
        if union_set != all_subjects:
            missing = sorted(all_subjects - union_set)[:10]
            unknown = sorted(union_set - all_subjects)[:10]
            raise ValueError(
                f"Fold {fold_name} does not partition all subjects. "
                f"Missing={missing}, Unknown={unknown}"
            )

        for sid in train_set:
            train_counts[sid] += 1
        for sid in val_set:
            val_counts[sid] += 1
        for sid in test_set:
            test_counts[sid] += 1

    # Across folds, each subject should appear expected number of times per split.
    for sid in sorted(all_subjects):
        if train_counts[sid] != expected_train_occ:
            raise ValueError(
                f"Unexpected train occurrence count for {sid}: "
                f"got {train_counts[sid]}, expected {expected_train_occ}"
            )
        if val_counts[sid] != expected_val_occ:
            raise ValueError(
                f"Unexpected val occurrence count for {sid}: "
                f"got {val_counts[sid]}, expected {expected_val_occ}"
            )
        if test_counts[sid] != expected_test_occ:
            raise ValueError(
                f"Unexpected test occurrence count for {sid}: "
                f"got {test_counts[sid]}, expected {expected_test_occ}"
            )


def split_dataset(
    subject_ids: list[str],
    k_fold_output_path: Path,
    ratios: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    k_folds: int = 10,
    random_seed: int = 42,
) -> None:
    """
    Create k-fold train/val/test splits over all subjects.

    Each fold contains train_ids, val_ids, and test_ids. When possible, each
    subject appears in val and in test at least once across all folds.
    
    Args:
        subject_ids: List of subject ID strings
        k_fold_output_path: Path to save k-fold split JSON
        ratios: Target split ratios for train/val/test
        k_folds: Number of folds for k-fold cross-validation (default: 10)
        random_seed: Random seed for reproducibility (default: 42)
    """
    if k_folds < 1:
        raise ValueError("k_folds must be >= 1")

    required_keys = {"train", "val", "test"}
    if set(ratios.keys()) != required_keys:
        raise ValueError(f"ratios must contain exactly keys {required_keys}")

    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"ratios must sum to 1.0, got {ratio_sum}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle subject IDs
    shuffled_ids = subject_ids.copy()
    random.shuffle(shuffled_ids)

    train_occ, val_occ, test_occ = _choose_occurrences(k_folds, ratios)
    achieved_ratios = {
        "train": train_occ / k_folds,
        "val": val_occ / k_folds,
        "test": test_occ / k_folds,
    }

    folds: dict[str, dict[str, list[str]]] = {}
    all_subjects = set(shuffled_ids)

    for fold in range(k_folds):
        val_ids: list[str] = []
        test_ids: list[str] = []

        for idx, sid in enumerate(shuffled_ids):
            val_fold_ids = {(idx + j) % k_folds for j in range(val_occ)}
            test_fold_ids = {(idx + val_occ + j) % k_folds for j in range(test_occ)}

            if fold in val_fold_ids:
                val_ids.append(sid)
            elif fold in test_fold_ids:
                test_ids.append(sid)

        train_ids = sorted(all_subjects - set(val_ids) - set(test_ids))
        folds[f"fold_{fold}"] = {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }
        print(
            f"Fold {fold}: "
            f"Train={len(train_ids)} "
            f"Val={len(val_ids)} "
            f"Test={len(test_ids)}"
        )

    _validate_no_leakage_and_coverage(
        folds=folds,
        all_subjects=all_subjects,
        expected_train_occ=train_occ,
        expected_val_occ=val_occ,
        expected_test_occ=test_occ,
    )

    payload = {
        "meta": {
            "k_folds": k_folds,
            "n_subjects": len(shuffled_ids),
            "random_seed": random_seed,
            "requested_ratios": ratios,
            "achieved_ratios": achieved_ratios,
            "subject_occurrences_per_split": {
                "train": train_occ,
                "val": val_occ,
                "test": test_occ,
            },
        },
        "folds": folds,
    }

    with open(k_fold_output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("Split complete")
    print(
        "Target ratios="
        f"{ratios} | Achieved ratios per fold={achieved_ratios}"
    )


if __name__ == "__main__":

    # Read subject ID's from selection csv
    subject_ids = pd.read_csv(SUBJECT_LIST_CSV)['Participant Id'].unique().tolist()

    # Split dataset
    split_dataset(subject_ids, K_FOLD_PATH)