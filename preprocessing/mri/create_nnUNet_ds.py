import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

K_FOLD_SPLITS_PATH = "/projects/prjs1713/MultimodalAI_Sjors/data/preprocessing/k_fold_splits.json"
SUCCESS_SUBJECTS_PATH="/projects/prjs1713/MultimodalAI_Sjors/data/preprocessing/mri/success_subjects.txt"
PREPROCESSING_OUTPUT_DIR="/projects/prjs1713/MultimodalAI_Sjors/data/preprocessing/mri/OUTPUT"
NNUNET_RAW_DIR="/projects/prjs1713/MultimodalAI_Sjors/nnUNet/nnUNet_raw"
DATASET_ID_START = 500
DATASET_SUFFIX = "FCD_combined"

def _load_kfold_splits(k_fold_splits_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load fold dict from k_fold_splits.json, supporting old/new JSON layouts."""
    with open(k_fold_splits_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "folds" in payload and isinstance(payload["folds"], dict):
        fold_payload = payload["folds"]
    elif isinstance(payload, dict):
        fold_payload = payload
    else:
        raise ValueError("Invalid k_fold_splits.json format: expected dict payload")

    fold_splits: Dict[str, Dict[str, List[str]]] = {}
    for fold_name, fold_data in fold_payload.items():
        if not isinstance(fold_data, dict) or not str(fold_name).startswith("fold_"):
            continue
        fold_splits[str(fold_name)] = {
            "train_ids": [str(s).strip() for s in fold_data.get("train_ids", []) if str(s).strip()],
            "val_ids": [str(s).strip() for s in fold_data.get("val_ids", []) if str(s).strip()],
            "test_ids": [str(s).strip() for s in fold_data.get("test_ids", []) if str(s).strip()],
        }

    if not fold_splits:
        raise ValueError("No fold_<idx> entries found in k_fold_splits.json")

    return dict(sorted(fold_splits.items(), key=lambda kv: int(kv[0].split("_")[-1])))


def _subject_file_triplet(preprocessing_dir: Path, subj_id: str) -> Tuple[Path, Path, Path]:
    subj_dir = preprocessing_dir / subj_id
    return (
        subj_dir / f"{subj_id}_T1w_norm.nii.gz",
        subj_dir / f"{subj_id}_FLAIR_norm.nii.gz",
        subj_dir / f"{subj_id}_gt_norm.nii.gz",
    )


def _copy_subject_to_split(preprocessing_dir: Path, subj_id: str, images_dir: Path, labels_dir: Path) -> bool:
    t1w_source, flair_source, label_source = _subject_file_triplet(preprocessing_dir, subj_id)
    if not t1w_source.exists() or not flair_source.exists() or not label_source.exists():
        print(f"Warning: Missing files for {subj_id}")
        return False

    t1w_dest = images_dir / f"{subj_id}_0000.nii.gz"
    flair_dest = images_dir / f"{subj_id}_0001.nii.gz"
    label_dest = labels_dir / f"{subj_id}.nii.gz"

    shutil.copy2(t1w_source, t1w_dest)
    shutil.copy2(flair_source, flair_dest)
    shutil.copy2(label_source, label_dest)
    return True


def _dataset_name_for_fold(dataset_id_start: int, fold_idx: int, dataset_suffix: str) -> str:
    return f"Dataset{dataset_id_start + fold_idx:03d}_{dataset_suffix}"


def _write_dataset_json(dataset_dir: Path, dataset_name: str, num_training: int) -> None:
    """Write nnUNet v2 dataset.json metadata for a 2-channel MRI segmentation task."""
    payload = {
        "name": dataset_name,
        "description": "FCD lesion segmentation (T1w + FLAIR)",
        "channel_names": {
            "0": "T1w",
            "1": "FLAIR",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _assert_no_leakage(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str], fold_name: str) -> None:
    """Fail fast if any subject leaks across train/val/test within a fold."""
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        details = []
        if train_val_overlap:
            details.append(f"train∩val={sorted(train_val_overlap)[:10]}")
        if train_test_overlap:
            details.append(f"train∩test={sorted(train_test_overlap)[:10]}")
        if val_test_overlap:
            details.append(f"val∩test={sorted(val_test_overlap)[:10]}")
        raise ValueError(f"Leakage detected in {fold_name}: " + " | ".join(details))


def _assert_split_matches_copied(
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
    copied_train_val_ids: Set[str],
    copied_test_ids: Set[str],
    fold_name: str,
) -> None:
    """Validate split IDs are fully represented by copied files and remain disjoint."""
    _assert_no_leakage(train_ids, val_ids, test_ids, fold_name)

    if not train_ids.issubset(copied_train_val_ids):
        missing = sorted(train_ids - copied_train_val_ids)[:10]
        raise ValueError(f"{fold_name}: train IDs missing from copied Tr files: {missing}")
    if not val_ids.issubset(copied_train_val_ids):
        missing = sorted(val_ids - copied_train_val_ids)[:10]
        raise ValueError(f"{fold_name}: val IDs missing from copied Tr files: {missing}")
    if not test_ids.issubset(copied_test_ids):
        missing = sorted(test_ids - copied_test_ids)[:10]
        raise ValueError(f"{fold_name}: test IDs missing from copied Ts files: {missing}")


def create_nnunet_dataset_per_fold(
    k_fold_splits_path: str,
    success_subjects_path: str,
    preprocessing_output_dir: str,
    nnunet_raw_dir: str,
    dataset_id_start: int = 500,
    dataset_suffix: str = "FCD",
    include_bonn_in_train: bool = True,
):
    """
    Create one nnUNet raw dataset per fold from k_fold_splits.json.

    For each fold:
    - test_ids -> imagesTs/labelsTs
    - train_ids + val_ids -> imagesTr/labelsTr
    - train/val split definition -> splits_final.json
    
    Args:
        k_fold_splits_path: Path to k_fold_splits.json
        success_subjects_path: Path to success_subjects.txt
        preprocessing_output_dir: Path to preprocessing/OUTPUT directory
        nnunet_raw_dir: Path to nnUNet_raw directory
        dataset_id_start: Starting dataset ID, e.g. 500 for fold_0->Dataset500_*
        dataset_suffix: Dataset suffix after ID
        include_bonn_in_train: Add successful Bonn* subjects to training set only
    """

    fold_splits = _load_kfold_splits(k_fold_splits_path)

    with open(success_subjects_path, "r", encoding="utf-8") as f:
        success_subjects = set(line.strip() for line in f if line.strip())

    bonn_subjects = {subject for subject in success_subjects if subject.startswith("Bonn")}

    preprocessing_dir = Path(preprocessing_output_dir)
    total_copied = 0

    for fold_name, fold_data in fold_splits.items():
        fold_idx = int(fold_name.split("_")[-1])
        dataset_name = _dataset_name_for_fold(dataset_id_start, fold_idx, dataset_suffix)

        train_ids = set(fold_data["train_ids"])
        val_ids = set(fold_data["val_ids"])
        test_ids = set(fold_data["test_ids"])

        valid_train_ids = train_ids.intersection(success_subjects)
        valid_val_ids = val_ids.intersection(success_subjects)
        valid_test_ids = test_ids.intersection(success_subjects)

        if include_bonn_in_train:
            valid_train_ids.update(bonn_subjects)
            valid_train_ids -= valid_test_ids

        # Ensure strict split boundaries.
        valid_train_ids -= valid_val_ids
        valid_train_ids -= valid_test_ids
        valid_val_ids -= valid_test_ids
        _assert_no_leakage(valid_train_ids, valid_val_ids, valid_test_ids, fold_name)

        dataset_dir = Path(nnunet_raw_dir) / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        images_tr_dir = dataset_dir / "imagesTr"
        labels_tr_dir = dataset_dir / "labelsTr"
        images_ts_dir = dataset_dir / "imagesTs"
        labels_ts_dir = dataset_dir / "labelsTs"

        images_tr_dir.mkdir(parents=True, exist_ok=True)
        labels_tr_dir.mkdir(parents=True, exist_ok=True)
        images_ts_dir.mkdir(parents=True, exist_ok=True)
        labels_ts_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{fold_name}] Building {dataset_name}: "
            f"train={len(valid_train_ids)}, val={len(valid_val_ids)}, test={len(valid_test_ids)}"
        )

        copied_train_val_ids: Set[str] = set()
        copied_test_ids: Set[str] = set()

        for subj_id in tqdm(sorted(valid_train_ids | valid_val_ids), desc=f"{fold_name} train/val"):
            if _copy_subject_to_split(preprocessing_dir, subj_id, images_tr_dir, labels_tr_dir):
                total_copied += 1
                copied_train_val_ids.add(subj_id)

        for subj_id in tqdm(sorted(valid_test_ids), desc=f"{fold_name} test"):
            if _copy_subject_to_split(preprocessing_dir, subj_id, images_ts_dir, labels_ts_dir):
                total_copied += 1
                copied_test_ids.add(subj_id)

        copied_train_ids = valid_train_ids.intersection(copied_train_val_ids)
        copied_val_ids = valid_val_ids.intersection(copied_train_val_ids)

        _assert_split_matches_copied(
            train_ids=copied_train_ids,
            val_ids=copied_val_ids,
            test_ids=copied_test_ids,
            copied_train_val_ids=copied_train_val_ids,
            copied_test_ids=copied_test_ids,
            fold_name=fold_name,
        )

        splits_path = dataset_dir / "splits_final.json"
        splits_payload = [
            {
                "train": sorted(copied_train_ids),
                "val": sorted(copied_val_ids),
            }
        ]
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits_payload, f, indent=2)

        _write_dataset_json(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            num_training=len(copied_train_val_ids),
        )

        print(f"[{fold_name}] Wrote splits: {splits_path}")
        print(f"[{fold_name}] Wrote dataset.json with numTraining={len(copied_train_val_ids)}")

    print("\nFold dataset creation complete")
    print(f"Total copied subject entries: {total_copied}")
    print(f"Output root: {nnunet_raw_dir}")


if __name__ == "__main__":

    create_nnunet_dataset_per_fold(
        k_fold_splits_path=K_FOLD_SPLITS_PATH,
        success_subjects_path=SUCCESS_SUBJECTS_PATH,
        preprocessing_output_dir=PREPROCESSING_OUTPUT_DIR,
        nnunet_raw_dir=NNUNET_RAW_DIR,
        dataset_id_start=DATASET_ID_START,
        dataset_suffix=DATASET_SUFFIX,
    )