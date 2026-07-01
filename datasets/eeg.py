"""
datasets/eeg.py
Dataset classes and utilities for EEG spike MIL models.

Provides dataset wrappers, data-loading helpers, and label constants shared by all
EEG MIL training scripts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets.augmentation import EEGSpikeAugment  # type: ignore


# -----------------------------------------------
# Task-level label constants
# -----------------------------------------------

LOBE_LABEL_TO_INT: Dict[str, int] = {
    "left_temporal":   0,
    "right_temporal":  0,
    "left_frontal":    1,
    "right_frontal":   1,
    "left_parietal":   2,
    "right_parietal":  2,
    "left_occipital":  3,
    "right_occipital": 3,
    "left_insula":     0,
    "right_insula":    0,
    "left_cingulate":  1,
    "right_cingulate": 1,
}

HEMI_LABEL_TO_INT: Dict[str, int] = {"left": 0, "right": 1}

LOBE_CLASSES: List[str] = ["temporal", "frontal", "parietal", "occipital"]

CHANNEL_LABEL_TO_INT: Dict[str, int] = {
    "Fp1": 0,  "Fp2": 1,  "F9": 2,   "F10": 3,  "F7": 4,
    "F3": 5,   "Fz": 6,   "F4": 7,   "F8": 8,
    "T7": 9,   "C3": 10,  "Cz": 11,  "C4": 12,  "T8": 13,
    "P7": 14,  "P3": 15,  "Pz": 16,  "P4": 17,  "P8": 18,
    "O1": 19,  "O2": 20,
}


# -----------------------------------------------
# Data-loading utilities (shared across scripts)
# -----------------------------------------------

def load_split(json_path: str, fold: int) -> Tuple[List[str], List[str]]:
    """Load train/val subject IDs for a given fold from k_fold_splits.json."""
    with open(json_path, "r") as f:
        payload = json.load(f)

    fold_payload = payload["folds"]
    fold_key = f"fold_{fold}"
    
    if fold_key not in fold_payload:
        raise ValueError(f"{fold_key} not found in {json_path}.")

    return fold_payload[fold_key]["train_ids"], fold_payload[fold_key]["val_ids"]


def load_labels(
    json_path: str,
    label_to_int: Dict[str, int],
    num_classes: int,
    multi_label: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load a label dictionary from a JSON file.

    Supports two formats:
      - dict  e.g. {"left_frontal": 0.9, "left_temporal": 0.1}  -> soft distribution
      - str   e.g. "left_frontal"                                 -> hard label

    Parameters
    ----------
    json_path    : path to JSON  {patient_id: label_or_dict}
    label_to_int : mapping from label string -> class index
    num_classes  : total number of classes
    multi_label  : if True, return float probability vectors; if False return int indices

    Returns
    -------
    dict  patient_id -> Tensor  (float32 vector of length num_classes, or int64 scalar)
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    label_dict: Dict[str, torch.Tensor] = {}

    for pid, entry in raw.items():
        if multi_label:
            vec = torch.zeros(num_classes, dtype=torch.float32)
            if isinstance(entry, dict):
                total = 0.0
                for name, prob in entry.items():
                    if name in label_to_int:
                        p = float(prob)
                        vec[label_to_int[name]] += p
                        total += p
                if total > 0:
                    vec /= total
            else:
                if entry in label_to_int:
                    vec[label_to_int[entry]] = 1.0
            label_dict[pid] = vec
        else:
            if isinstance(entry, dict):
                best = max(entry.items(), key=lambda kv: kv[1])[0]
                if best in label_to_int:
                    label_dict[pid] = int(label_to_int[best])
            else:
                if entry in label_to_int:
                    label_dict[pid] = int(label_to_int[entry])

    return label_dict


def load_regression_targets(json_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load normalized MNI coordinate regression targets from a JSON file.

    Expected format::

        {
            "RESP0001": {"normalized_mu": {"x": 0.12, "y": -0.34, "z": 0.56}},
            ...
        }

    Returns
    -------
    dict  patient_id -> {"mu": FloatTensor shape (3,)}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    targets: Dict[str, Dict[str, torch.Tensor]] = {}
    for pid, entry in data.items():
        mu = torch.tensor(
            [entry["normalized_mu"]["x"],
             entry["normalized_mu"]["y"],
             entry["normalized_mu"]["z"]],
            dtype=torch.float32,
        )
        targets[pid] = {"mu": mu}

    return targets


def load_multitask_targets(
    json_targets_path: str,
    lobe_json_path: Optional[str] = None,
    hemi_json_path: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load multi-task targets: normalized MNI coordinates + hemisphere + lobe labels.

    ``lobe_json_path`` and ``hemi_json_path`` are optional.  When omitted all
    patients receive a dummy target with ``*_mask = 0`` so the masked loss for
    that head contributes nothing.

    Returns
    -------
    dict  patient_id -> {
        "mu":          FloatTensor (3,)   normalized MNI xyz
        "hemi_target": LongTensor  ()     hemisphere class index
        "lobe_target": LongTensor  ()     lobe class index
        "hemi_mask":   FloatTensor ()     1.0 if hemi label exists, else 0.0
        "lobe_mask":   FloatTensor ()     1.0 if lobe label exists, else 0.0
    }
    """
    with open(json_targets_path, "r") as f:
        coord_data = json.load(f)

    raw_lobe_data = None
    if lobe_json_path is not None:
        with open(lobe_json_path, "r") as f:
            raw_lobe_data = json.load(f)

    lobe_dict = (
        load_labels(lobe_json_path, LOBE_LABEL_TO_INT, len(LOBE_CLASSES), multi_label=False)
        if lobe_json_path is not None else {}
    )
    hemi_dict = (
        load_labels(hemi_json_path, HEMI_LABEL_TO_INT, len(HEMI_LABEL_TO_INT), multi_label=False)
        if hemi_json_path is not None else {}
    )

    targets: Dict[str, Dict[str, torch.Tensor]] = {}

    for pid, entry in coord_data.items():
        mu = torch.tensor(
            [entry["normalized_mu"]["x"],
             entry["normalized_mu"]["y"],
             entry["normalized_mu"]["z"]],
            dtype=torch.float32,
        )

        hemi_has = pid in hemi_dict
        lobe_has = pid in lobe_dict

        target_entry: Dict[str, Any] = {
            "mu":          mu,
            "hemi_target": torch.tensor(hemi_dict[pid] if hemi_has else 0, dtype=torch.long),
            "lobe_target": torch.tensor(lobe_dict[pid] if lobe_has else 0, dtype=torch.long),
            "hemi_mask":   torch.tensor(float(hemi_has), dtype=torch.float32),
            "lobe_mask":   torch.tensor(float(lobe_has), dtype=torch.float32),
        }

        if raw_lobe_data is not None and pid in raw_lobe_data:
            raw_lobe_entry = raw_lobe_data[pid]
            lobe_distribution = torch.zeros(len(LOBE_CLASSES), dtype=torch.float32)
            if isinstance(raw_lobe_entry, dict):
                total = 0.0
                for name, prob in raw_lobe_entry.items():
                    if name in LOBE_LABEL_TO_INT:
                        value = float(prob)
                        lobe_distribution[LOBE_LABEL_TO_INT[name]] += value
                        total += value
                if total > 0:
                    lobe_distribution /= total
            else:
                if raw_lobe_entry in LOBE_LABEL_TO_INT:
                    lobe_distribution[LOBE_LABEL_TO_INT[raw_lobe_entry]] = 1.0
            if float(lobe_distribution.sum()) > 0.0:
                target_entry["lobe_distribution"] = lobe_distribution

        # Optional spatial supervision metadata for deconv head.
        mask_path = (
            entry.get("target_mask_path")
            or entry.get("mask_path")
            or entry.get("mni_mask_path")
            or entry.get("lesion_mask_path")
            or entry.get("mask_npy_path")
        )
        if mask_path is not None:
            target_entry["mask_path"] = str(mask_path)

        if "target_mask" in entry and entry["target_mask"] is not None:
            target_entry["target_mask"] = torch.tensor(entry["target_mask"], dtype=torch.float32)

        mask_space = entry.get("target_mask_space") or entry.get("mask_space") or entry.get("space")
        if mask_space is not None:
            target_entry["mask_space"] = str(mask_space)

        if "mask_affine" in entry and entry["mask_affine"] is not None:
            target_entry["mask_affine"] = entry["mask_affine"]

        targets[pid] = target_entry

    return targets


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    radius = int(max(1, round(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def gaussian_blur_3d(mask_5d: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur to a mask tensor of shape [1, 1, D, H, W]."""
    if sigma <= 0:
        return mask_5d
    if mask_5d.ndim != 5 or mask_5d.shape[0] != 1 or mask_5d.shape[1] != 1:
        raise ValueError(f"gaussian_blur_3d expects shape [1,1,D,H,W], got {tuple(mask_5d.shape)}")

    kernel_1d = _gaussian_kernel1d(float(sigma)).to(device=mask_5d.device, dtype=mask_5d.dtype)
    k = kernel_1d.numel()

    out = F.conv3d(
        mask_5d,
        kernel_1d.view(1, 1, k, 1, 1),
        padding=(k // 2, 0, 0),
    )
    out = F.conv3d(
        out,
        kernel_1d.view(1, 1, 1, k, 1),
        padding=(0, k // 2, 0),
    )
    out = F.conv3d(
        out,
        kernel_1d.view(1, 1, 1, 1, k),
        padding=(0, 0, k // 2),
    )
    return out


def _load_mask_from_path(mask_path: str) -> np.ndarray:
    suffix = Path(mask_path).suffix.lower()
    if suffix == ".npy":
        return np.load(mask_path).astype(np.float32)
    if suffix == ".npz":
        npz = np.load(mask_path, allow_pickle=True)
        try:
            if "mask" in npz:
                arr = npz["mask"]
            elif "gt" in npz:
                arr = npz["gt"]
            else:
                keys = list(npz.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ mask file {mask_path!r} has multiple arrays {keys} and no 'mask'/'gt' key."
                    )
                arr = npz[keys[0]]
        finally:
            npz.close()
        return np.asarray(arr, dtype=np.float32)

    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Loading NIfTI mask targets requires nibabel. Install nibabel or provide .npy/.npz masks."
        ) from exc

    nii = nib.load(mask_path)
    return np.asarray(nii.get_fdata(dtype=np.float32), dtype=np.float32)


def find_patient_files(
    data_dir: str,
    patient_ids: List[str],
    target_dict: Dict,
    test_mode: bool = False,
    skip_zero_labels: bool = False,
    file_suffix: str = "_spikes_1-70Hz.npy",
) -> Tuple[List[str], List[str], List]:
    """
    Build (ids, file_paths, targets) lists from a patient ID list.

    Patients are skipped if their spike file is missing or they have no entry in
    ``target_dict``. Optionally skip patients whose label tensor is all zeros.

    Parameters
    ----------
    data_dir          : directory containing ``<pid><file_suffix>`` files
    patient_ids       : ordered list of patient IDs to look up
    target_dict       : mapping patient_id -> target (any type)
    test_mode         : if True, keep only the first 4 valid patients
    skip_zero_labels  : if True, also skip patients whose target is a tensor of all zeros
    file_suffix       : suffix appended to patient_id to form the filename

    Returns
    -------
    (valid_ids, file_paths, targets)
    """
    valid_ids, files, targets = [], [], []
    skipped = []

    for pid in patient_ids:
        path = os.path.join(data_dir, f"{pid}{file_suffix}")
        target = target_dict.get(pid, None)

        if not os.path.exists(path) or target is None:
            skipped.append(pid)
            continue

        if skip_zero_labels and isinstance(target, torch.Tensor) and torch.all(target == 0):
            skipped.append(pid)
            continue

        valid_ids.append(pid)
        files.append(path)
        targets.append(target)

    if skipped:
        print(
            f"\033[38;5;208mWarning: Skipped {len(skipped)} patients "
            f"(missing file / label): {sorted(skipped)}\033[0m"
        )

    if test_mode:
        valid_ids, files, targets = valid_ids[:4], files[:4], targets[:4]
        print("\033[38;5;208mWarning: test_mode – limited to first 4 patients.\033[0m")

    print(f"Found {len(valid_ids)} valid patients out of {len(patient_ids)} requested.")
    return valid_ids, files, targets

class SpikeDataset(Dataset):
    """
    Flat per-spike dataset for single-spike classification.
    Each item returns ONE spike (C, window_size) and its patient label(s).
    
    Parameters:
        patient_ids:            list[str]
        patient_files:          list[str]  paths to npy arrays [n_spikes, C, L]
        patient_labels:         list[int] or list[list[int]] for multi-label
        max_spikes_per_patient: cap per patient (to balance distribution)
        segment_length:         full spike length in file (L)
        window_size:            crop size returned by dataset
        max_offset:             offset jitter around center
        training:               whether to apply augmentation
        num_classes:            total number of classes for multi-label encoding
    """
    def __init__(self, patient_ids, patient_files, patient_labels,
                 segment_length=256, window_size=128, max_offset=0,
                 max_spikes_per_patient=200, training=False, num_classes=None):

        self.patient_ids = patient_ids
        self.patient_files = patient_files
        self.patient_labels = patient_labels
        self.segment_length = segment_length
        self.window_size = window_size
        self.max_offset = max_offset
        self.training = training
        self.num_classes = num_classes

        # Determine if multi-label based on first label
        self.is_multilabel = isinstance(patient_labels[0], (list, np.ndarray))
        if self.is_multilabel and num_classes is None:
            raise ValueError("num_classes must be specified for multi-label classification")

        if self.training:
            self.augmenter = EEGSpikeAugment()

        self.spikes = []           # list of np arrays [C, L]
        self.labels = []           # list of ints or binary vectors

        # Load all spikes into memory (balanced via subsampling)
        for file, label in zip(patient_files, patient_labels):
            arr = np.load(file)       # shape [N, C, L]
            N = arr.shape[0]

            # cap number of spikes from this patient
            if N > max_spikes_per_patient:
                idx = np.random.choice(N, max_spikes_per_patient, replace=False)
                arr = arr[idx]

            # Convert label to binary vector if multi-label
            if self.is_multilabel:
                binary_label = np.zeros(self.num_classes, dtype=np.float32)
                binary_label[label] = 1.0
                processed_label = binary_label
            else:
                processed_label = label

            # extend lists
            for spike in arr:
                self.spikes.append(spike)   # (C, L)
                self.labels.append(processed_label)

        self.spikes = np.stack(self.spikes)   # (N_total, C, L)
        if self.is_multilabel:
            self.labels = np.stack(self.labels)  # (N_total, num_classes)
        else:
            self.labels = np.array(self.labels)   # (N_total,)

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, idx):
        spike = self.spikes[idx]      # (C, L)
        label = self.labels[idx]

        C, L = spike.shape
        center = L // 2

        # Crop with random offset if training
        if self.training:
            start = np.random.randint(
                center - self.window_size//2 - self.max_offset,
                center - self.window_size//2 + self.max_offset + 1,
            )
        else:
            start = center - self.window_size//2

        end = start + self.window_size
        spike = spike[:, start:end]   # shape (C, window_size)

        # Augmentation
        if self.training:
            spike = self.augmenter(spike)   # (C, window_size)

        label_dtype = torch.float32 if self.is_multilabel else torch.long
        return (
            torch.tensor(spike, dtype=torch.float32),  # (C, window)
            torch.tensor(label, dtype=label_dtype)
        )


class FlatSpikeEncoderPretrainDataset(Dataset):
    """
    Flat per-segment dataset for pretraining the temporal spike encoder on the
    1-70 Hz flat dataset files produced by create_eeg_ds.py.

    Each item returns ONE segment (C, window_size) and a target dict with
    channel, spike-binary, perception, and Persyst-flag targets.

    Parameters
    ----------
    flat_data_dir
        Directory containing per-patient ``*_flat_1-70Hz_segments.npy`` and
        ``*_flat_1-70Hz_metadata.csv`` files.
    patient_ids
        Optional list of patient IDs.  If None, all patients whose segment
        files exist in ``flat_data_dir`` are used.
    segment_length
        Expected full segment length (used for sanity checking only).
    window_size
        Temporal crop size returned by ``__getitem__``.
    max_offset
        Maximum random jitter (samples) around center crop during training.
    training
        If True apply ``EEGSpikeAugment`` and random offset.
    max_segments_per_patient
        Cap the number of segments loaded per patient.  Applied *after*
        harmonisation.  Default is 3000.
    harmonise_spike_ratio
        Desired spike-to-non-spike ratio.  Default ``2.0`` yields roughly
        2/3 spike and 1/3 non-spike segments per patient.  The majority
        class is downsampled without replacement to achieve the ratio;
        the minority class is left untouched.  Set to ``None`` to disable.
    include_detection_segments
        Include rows with ``segment_type == "detection"``.
    include_non_spike_segments
        Include rows with ``segment_type == "non_spike"``.
    require_valid_channel_target
        If True, drop rows where ``channel_target == -1``.
    use_memmap
        If True load ``.npy`` arrays with ``mmap_mode="r"``.
    """

    def __init__(
        self,
        flat_data_dir: str,
        patient_ids: Optional[List[str]] = None,
        segment_length: int = 256,
        window_size: int = 128,
        max_offset: int = 0,
        training: bool = False,
        max_segments_per_patient: Optional[int] = 3000,
        harmonise_spike_ratio: Optional[float] = 2.0,
        include_detection_segments: bool = True,
        include_non_spike_segments: bool = True,
        require_valid_channel_target: bool = False,
        use_memmap: bool = False,
    ):
        self.flat_data_dir = Path(flat_data_dir)
        self.segment_length = segment_length
        self.window_size = window_size
        self.max_offset = max_offset
        self.training = training
        self.harmonise_spike_ratio = harmonise_spike_ratio

        if training:
            self.augmenter = EEGSpikeAugment()

        # Discover patient IDs if not provided
        if patient_ids is None:
            patient_ids = find_flat_pretrain_patient_ids(flat_data_dir)
            print(f"Discovered {len(patient_ids)} patients in {flat_data_dir}.")

        self.segment_arrays: List[np.ndarray] = []
        self.index_table: List[dict] = []

        n_skipped = 0
        for pid in patient_ids:
            seg_path  = self.flat_data_dir / f"{pid}_flat_1-70Hz_segments.npy"
            meta_path = self.flat_data_dir / f"{pid}_flat_1-70Hz_metadata.csv"

            if not seg_path.exists() or not meta_path.exists():
                print(
                    f"\033[38;5;208mWarning: missing flat files for {pid}, skipping.\033[0m"
                )
                n_skipped += 1
                continue

            mmap = "r" if use_memmap else None
            arr  = np.load(seg_path, mmap_mode=mmap)    # (N, C, L)
            meta = pd.read_csv(meta_path)

            if len(meta) != arr.shape[0]:
                print(
                    f"\033[38;5;208mWarning: {pid} metadata rows ({len(meta)}) != "
                    f"segment array rows ({arr.shape[0]}). Skipping.\033[0m"
                )
                n_skipped += 1
                continue

            if window_size > arr.shape[2]:
                raise ValueError(
                    f"window_size={window_size} > segment length={arr.shape[2]} "
                    f"for patient {pid}."
                )

            array_id = len(self.segment_arrays)
            self.segment_arrays.append(arr)

            row_indices = []
            for i, (_, row) in enumerate(meta.iterrows()):
                seg_type = str(row.get("segment_type", ""))

                if seg_type == "detection" and not include_detection_segments:
                    continue
                if seg_type == "non_spike" and not include_non_spike_segments:
                    continue

                ch = str(row.get("detected_channel", "")).strip()
                channel_target = CHANNEL_LABEL_TO_INT.get(ch, -1)

                spike_target = 1.0 if seg_type == "detection" else 0.0

                raw_perc = row.get("perception", float("nan"))
                try:
                    perception = float(raw_perc)
                    if not np.isfinite(perception):
                        perception = 0.0
                except (ValueError, TypeError):
                    perception = 0.0

                is_persyst = float(bool(row.get("is_persyst_detection", False)))
                is_thresh  = float(bool(row.get("is_thresholded_spike", False)))

                if require_valid_channel_target and channel_target == -1:
                    continue

                row_indices.append({
                    "patient_id":           pid,
                    "array_index":          i,
                    "segments_array_id":    array_id,
                    "channel_target":       channel_target,
                    "spike_target":         spike_target,
                    "perception":           perception,
                    "is_persyst_detection": is_persyst,
                    "is_thresholded_spike": is_thresh,
                })

            # --- Harmonisation: target spike:non-spike ratio ---
            if harmonise_spike_ratio is not None and include_detection_segments and include_non_spike_segments:
                spike_rows    = [r for r in row_indices if r["spike_target"] == 1.0]
                nonspike_rows = [r for r in row_indices if r["spike_target"] == 0.0]
                n_s  = len(spike_rows)
                n_ns = len(nonspike_rows)
                if n_s > 0 and n_ns > 0:
                    target_ns = int(round(n_s / harmonise_spike_ratio))
                    target_s  = int(round(n_ns * harmonise_spike_ratio))
                    if n_ns > target_ns:
                        # Too many non-spikes — downsample non-spikes
                        chosen = np.random.choice(n_ns, target_ns, replace=False)
                        nonspike_rows = [nonspike_rows[i] for i in chosen]
                    elif n_s > target_s:
                        # Too many spikes — downsample spikes
                        chosen = np.random.choice(n_s, target_s, replace=False)
                        spike_rows = [spike_rows[i] for i in chosen]
                    row_indices = spike_rows + nonspike_rows

            # --- Cap total segments per patient ---
            if max_segments_per_patient is not None and len(row_indices) > max_segments_per_patient:
                chosen = np.random.choice(len(row_indices), max_segments_per_patient, replace=False)
                row_indices = [row_indices[i] for i in chosen]

            self.index_table.extend(row_indices)

        print(
            f"FlatSpikeEncoderPretrainDataset: {len(self.index_table)} segments from "
            f"{len(patient_ids) - n_skipped} patients "
            f"({n_skipped} skipped)."
        )

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, idx: int):
        entry  = self.index_table[idx]
        arr    = self.segment_arrays[entry["segments_array_id"]]
        seg    = arr[entry["array_index"]]   # (C, L)
        C, L   = seg.shape
        center = L // 2

        if self.training and self.max_offset > 0:
            start = int(np.random.randint(
                center - self.window_size // 2 - self.max_offset,
                center - self.window_size // 2 + self.max_offset + 1,
            ))
        else:
            start = center - self.window_size // 2

        start = max(0, min(start, L - self.window_size))
        seg = seg[:, start : start + self.window_size]   # (C, window_size)

        if self.training:
            seg = self.augmenter(seg)

        return (
            torch.tensor(seg, dtype=torch.float32),
            {
                "channel_target":       torch.tensor(entry["channel_target"],       dtype=torch.long),
                "spike_target":         torch.tensor(entry["spike_target"],         dtype=torch.float32),
                "perception":           torch.tensor(entry["perception"],           dtype=torch.float32),
                "is_persyst_detection": torch.tensor(entry["is_persyst_detection"], dtype=torch.float32),
                "is_thresholded_spike": torch.tensor(entry["is_thresholded_spike"], dtype=torch.float32),
                "patient_id":           entry["patient_id"],
            },
        )


def flat_spike_pretrain_collate(batch):
    """Collate function for :class:`FlatSpikeEncoderPretrainDataset`."""
    segments, targets = zip(*batch)
    out_targets = {
        "channel_target":       torch.stack([t["channel_target"]       for t in targets]),
        "spike_target":         torch.stack([t["spike_target"]         for t in targets]),
        "perception":           torch.stack([t["perception"]           for t in targets]),
        "is_persyst_detection": torch.stack([t["is_persyst_detection"] for t in targets]),
        "is_thresholded_spike": torch.stack([t["is_thresholded_spike"] for t in targets]),
        "patient_id":           [t["patient_id"] for t in targets],
    }
    return torch.stack(segments, dim=0), out_targets


def find_flat_pretrain_patient_ids(flat_data_dir: str) -> List[str]:
    """Return sorted patient IDs inferred from flat segment files in *flat_data_dir*."""
    files = sorted(Path(flat_data_dir).glob("*_flat_1-70Hz_segments.npy"))
    return [f.name.replace("_flat_1-70Hz_segments.npy", "") for f in files]


def build_flat_pretrain_datasets(
    flat_data_dir: str,
    splits_json_path: str,
    fold: int,
    segment_length: int = 256,
    window_size: int = 128,
    max_offset: int = 0,
    max_segments_per_patient: Optional[int] = 3000,
    harmonise_spike_ratio: Optional[float] = 2.0,
    require_valid_channel_target: bool = False,
    use_memmap: bool = False,
):
    """
    Build train/val :class:`FlatSpikeEncoderPretrainDataset` pairs from a fold
    split JSON (same format used by the MIL training scripts).

    Returns
    -------
    train_dataset, val_dataset
    """
    train_ids, val_ids = load_split(splits_json_path, fold)
    train_dataset = FlatSpikeEncoderPretrainDataset(
        flat_data_dir=flat_data_dir,
        patient_ids=train_ids,
        segment_length=segment_length,
        window_size=window_size,
        max_offset=max_offset,
        training=True,
        max_segments_per_patient=max_segments_per_patient,
        harmonise_spike_ratio=harmonise_spike_ratio,
        require_valid_channel_target=require_valid_channel_target,
        use_memmap=use_memmap,
    )
    val_dataset = FlatSpikeEncoderPretrainDataset(
        flat_data_dir=flat_data_dir,
        patient_ids=val_ids,
        segment_length=segment_length,
        window_size=window_size,
        max_offset=0,
        training=False,
        max_segments_per_patient=max_segments_per_patient,
        harmonise_spike_ratio=harmonise_spike_ratio,
        require_valid_channel_target=require_valid_channel_target,
        use_memmap=use_memmap,
    )
    return train_dataset, val_dataset


class PatientMILSpikeDataset(Dataset):
    """
    Returns one patient per dataset index.
    Each item contains all spikes belonging to the patient.

    Supports:
      - Classification targets (scalar or vector)
      - Regression targets (dict with keys: 'mu', 'sigma')
    """

    def __init__(
        self,
        patient_ids,
        patient_files,
        patient_labels,
        segment_length=256,
        window_size=128,
        max_offset=0,
        max_spikes_per_bag=32,
        min_spikes_per_patient=64,
        training=False,
        training_drop_ratio=0.1,
        multi_label=True,
    ):

        self.patient_ids = patient_ids
        self.patient_files = patient_files
        self.patient_labels = patient_labels

        self.segment_length = segment_length
        self.window_size = window_size
        self.max_offset = max_offset
        self.max_spikes = max_spikes_per_bag
        self.min_spikes = min_spikes_per_patient

        self.training = training
        self.training_drop_ratio = training_drop_ratio
        self.is_multilabel = multi_label

        # Determine task type from first label
        first_label = patient_labels[0]
        self.is_regression = isinstance(first_label, dict)

        # Set up augmenter if training
        if self.training:
            self.augmenter = EEGSpikeAugment()

        # Load EEG data into memory
        self.patients_data = [np.load(f) for f in patient_files]  # [n_spikes, C, L]

        # Filter out patients with too few spikes
        filtered_data = []
        filtered_ids = []
        filtered_labels = []
        filtered_files = []
        n_spikes_list = []
        for data, pid, label, file in zip(self.patients_data, self.patient_ids, self.patient_labels, self.patient_files):
            if data.shape[0] >= self.min_spikes:
                filtered_data.append(data)
                filtered_ids.append(pid)
                filtered_labels.append(label)
                filtered_files.append(file)
                n_spikes_list.append(data.shape[0])
            else:
                print(f">>> {Path(file).stem} has only {data.shape[0]} spikes (< {self.min_spikes}), skipping.")

        self.patients_data = filtered_data
        self.patient_ids = filtered_ids
        self.patient_labels = filtered_labels
        self.patient_files = filtered_files

        # Track remaining (not-yet-sampled) spike indices per patient.
        # For each patient we sample from this pool without replacement and only
        # refill once all spikes have been seen.
        self.available_spike_indices = [
            np.arange(data.shape[0], dtype=np.int64) for data in self.patients_data
        ]

        print(f"Loaded {len(self.patients_data)} valid patients after filtering for minimum spikes.")
        print(f"Number of spikes per patient: {np.median(n_spikes_list)} (median), {np.min(n_spikes_list)} - {np.max(n_spikes_list)} (min - max)")

    def __len__(self):
        return len(self.patients_data)

    def __getitem__(self, idx):
        spikes = self.patients_data[idx]  # [n_spikes, C, segment_length]
        label = self.patient_labels[idx]

        n_spikes_total, C, L = spikes.shape

        # ---------------------------------------------------------
        # Spike subsampling (MIL)
        # ---------------------------------------------------------
        if n_spikes_total > self.max_spikes:
            available = self.available_spike_indices[idx]
            selected_chunks = []
            remaining_to_pick = self.max_spikes

            # Consume from the remaining pool first; only refill after a full cycle.
            while remaining_to_pick > 0:
                if available.size == 0:
                    available = np.arange(n_spikes_total, dtype=np.int64)

                if available.size <= remaining_to_pick:
                    pick = available
                    available = np.empty(0, dtype=np.int64)
                else:
                    pick_pos = np.random.choice(available.size, remaining_to_pick, replace=False)
                    pick = available[pick_pos]
                    keep_mask = np.ones(available.size, dtype=bool)
                    keep_mask[pick_pos] = False
                    available = available[keep_mask]

                selected_chunks.append(pick)
                remaining_to_pick -= pick.size

            indices = np.concatenate(selected_chunks)
            self.available_spike_indices[idx] = available
            spikes = spikes[indices]

        n_spikes = spikes.shape[0]

        # ---------------------------------------------------------
        # Temporal cropping  # TODO: Probably remove, already in augmenter.
        # ---------------------------------------------------------
        if self.training:
            actual_start = L // 2 - self.window_size // 2
            starts = np.random.randint(
                actual_start - self.max_offset,
                actual_start + self.max_offset + 1,
                size=n_spikes,
            )
        else:
            starts = np.full(n_spikes, L // 2 - self.window_size // 2)

        ends = starts + self.window_size
        cropped = np.stack(
            [spikes[i, :, starts[i] : ends[i]] for i in range(n_spikes)]
        )

        # ---------------------------------------------------------
        # Training-time spike dropout + augmentation
        # ---------------------------------------------------------
        if self.training:
            keep_ratio = 1.0 - self.training_drop_ratio
            keep_count = max(1, int(n_spikes * keep_ratio))
            keep_idx = np.random.choice(n_spikes, keep_count, replace=False)

            cropped = cropped[keep_idx]
            n_spikes = keep_count

            augmented = []
            for i in range(n_spikes):
                augmented.append(self.augmenter(cropped[i]))
            cropped = np.stack(augmented)

        # ---------------------------------------------------------
        # Label handling
        # ---------------------------------------------------------
        if self.is_regression:
            # Expect dict with 'mu'
            # Should already be torch tensor
            target = label["mu"].clone()
        else:
            # Classification case (unchanged)
            label_dtype = torch.float32 if self.is_multilabel else torch.long
            target = torch.tensor(label, dtype=label_dtype)

        return (
            torch.tensor(cropped, dtype=torch.float32),  # [n_spikes, C, window_size]
            target,
        )


def mil_collate(batch):
    spikes_list, labels_list = zip(*batch)   # list of N_i spike tensors

    # Pad spikes to max spikes in batch
    max_spikes = max(s.shape[0] for s in spikes_list)

    padded = []
    masks = []

    for spikes in spikes_list:
        n, C, L = spikes.shape
        pad_n = max_spikes - n
        padded.append(torch.cat([
            spikes,
            torch.zeros(pad_n, C, L)
        ], dim=0))

        mask = torch.zeros(max_spikes)
        mask[:n] = 1
        masks.append(mask)

    return (
        torch.stack(padded, dim=0),          # (B, max_spikes, C, L)
        torch.stack(masks, dim=0),           # (B, max_spikes)
        torch.stack(labels_list, dim=0),     # (B, num_classes) or (B,)
    )


# -----------------------------------------------
# Multi-task MIL dataset and collate
# -----------------------------------------------

class MultiHeadTargetDataset(Dataset):
    """
    Wraps a :class:`PatientMILSpikeDataset` to return a multi-task target dict
    instead of the single-value target in the base dataset.

    The dict contains the keys produced by :func:`load_multitask_targets`:
    ``coord_target``, ``hemi_target``, ``lobe_target``, ``hemi_mask``, ``lobe_mask``.
    """

    def __init__(
        self,
        base_dataset: PatientMILSpikeDataset,
        target_by_pid: Dict,
        deconv_enabled: bool = False,
        deconv_output_shape: Optional[Tuple[int, int, int]] = None,
        deconv_target_blur_sigma: float = 0.0,
        deconv_require_mni_alignment: bool = False,
        deconv_mask_npz_dir: Optional[str] = None,
        harmonize_lobe_locations: bool = False,
        lobe_harmonization_factor: float = 1.0,
    ):
        self.base_dataset = base_dataset
        self.target_by_pid = target_by_pid
        self.patient_ids = base_dataset.patient_ids
        self.deconv_enabled = deconv_enabled
        self.deconv_output_shape = tuple(deconv_output_shape) if deconv_output_shape is not None else None
        self.deconv_target_blur_sigma = float(deconv_target_blur_sigma)
        self.deconv_require_mni_alignment = deconv_require_mni_alignment
        self.deconv_mask_npz_dir = deconv_mask_npz_dir
        self.harmonize_lobe_locations = harmonize_lobe_locations
        self.lobe_harmonization_factor = float(lobe_harmonization_factor)
        self._deconv_mask_cache: Dict[str, torch.Tensor] = {}
        self.sample_weights = self._build_sample_weights() if harmonize_lobe_locations else None

    def _get_lobe_distribution(self, pid: str) -> Optional[torch.Tensor]:
        target = self.target_by_pid.get(pid)
        if target is None:
            return None

        distribution = target.get("lobe_distribution", None)
        if distribution is None:
            if float(target.get("lobe_mask", torch.tensor(0.0)).item()) <= 0.5:
                return None
            lobe_target = target.get("lobe_target", None)
            if lobe_target is None:
                return None
            distribution = torch.zeros(len(LOBE_CLASSES), dtype=torch.float32)
            distribution[int(lobe_target.item())] = 1.0
        else:
            distribution = torch.as_tensor(distribution, dtype=torch.float32)

        distribution = distribution.view(-1)
        if distribution.numel() != len(LOBE_CLASSES):
            raise ValueError(
                f"Expected lobe distribution with {len(LOBE_CLASSES)} classes, got shape {tuple(distribution.shape)} for patient {pid!r}"
            )
        total = float(distribution.sum().item())
        if total <= 0.0 or not torch.isfinite(distribution).all():
            return None
        return distribution / total

    def _build_sample_weights(self) -> Optional[torch.Tensor]:
        if self.lobe_harmonization_factor <= 0.0:
            print("Lobe harmonization factor is <= 0; using uniform sampling.")
            return None

        lobe_distributions = []
        for pid in self.base_dataset.patient_ids:
            distribution = self._get_lobe_distribution(pid)
            if distribution is not None:
                lobe_distributions.append(distribution)

        if not lobe_distributions:
            print(
                "Lobe harmonization requested, but no usable lobe distributions were found; "
                "falling back to uniform sampling."
            )
            return None

        stacked = torch.stack(lobe_distributions, dim=0)
        class_mass = stacked.sum(dim=0)
        class_freq = class_mass / class_mass.sum().clamp_min(1e-8)
        class_rarity = class_freq.clamp_min(1e-8).reciprocal()
        class_rarity = class_rarity / class_rarity.mean().clamp_min(1e-8)

        raw_weights = []
        for pid in self.base_dataset.patient_ids:
            distribution = self._get_lobe_distribution(pid)
            if distribution is None:
                raw_weights.append(1.0)
                continue
            patient_rarity = float((distribution * class_rarity).sum().item())
            blended = 1.0 + self.lobe_harmonization_factor * (patient_rarity - 1.0)
            raw_weights.append(max(blended, 1e-3))

        weights = torch.tensor(raw_weights, dtype=torch.double)
        mean_weight = float(weights.mean().item())
        if mean_weight > 0.0:
            weights = weights / mean_weight

        print(
            "Enabled lobe harmonization for MultiHeadTargetDataset: "
            f"factor={self.lobe_harmonization_factor:.3f}, "
            f"weight range=[{float(weights.min().item()):.3f}, {float(weights.max().item()):.3f}]"
        )
        return weights

    def _load_deconv_mask_from_preproc_npz(self, pid: str) -> Optional[torch.Tensor]:
        if self.deconv_mask_npz_dir is None:
            return None

        npz_path = os.path.join(self.deconv_mask_npz_dir, f"{pid}_preproc.npz")
        if not os.path.exists(npz_path):
            return None

        npz = np.load(npz_path, allow_pickle=True)
        try:
            if "gt" not in npz:
                raise ValueError(
                    f"Found MRI preproc file for {pid!r} at {npz_path}, but it has no 'gt' field. "
                    "Deconv spatial supervision requires npz['gt']."
                )
            mask_np = np.asarray(npz["gt"], dtype=np.float32)
        finally:
            npz.close()

        return torch.from_numpy(mask_np).float()

    def _prepare_deconv_target_mask(self, pid: str) -> torch.Tensor:
        if pid in self._deconv_mask_cache:
            return self._deconv_mask_cache[pid]

        if self.deconv_output_shape is None:
            raise ValueError("deconv_output_shape must be provided when deconv_enabled=True")

        t = self.target_by_pid[pid]
        mask_space = str(t.get("mask_space", "")).lower().strip()
        if self.deconv_mask_npz_dir is not None:
            # preprocess_mri.py generates gt in normalized MNI target grid.
            mask_space = "mni"

        if self.deconv_require_mni_alignment and mask_space and mask_space not in {"mni", "normalized", "mni152"}:
            raise ValueError(
                "Deconv spatial head requires target masks aligned to the decoder output space. "
                "Native-space masks without transforms are not supported. "
                f"Patient {pid!r} has mask_space={mask_space!r}."
            )

        mask = self._load_deconv_mask_from_preproc_npz(pid)
        if mask is not None:
            pass
        elif t.get("target_mask") is not None:
            mask = torch.as_tensor(t["target_mask"], dtype=torch.float32)
        elif t.get("mask_path") is not None:
            mask_np = _load_mask_from_path(str(t["mask_path"]))
            mask = torch.from_numpy(mask_np).float()
        else:
            raise ValueError(
                f"Missing spatial mask target for patient {pid!r}. "
                "Provide MRI preproc npz with 'gt' at <pid>_preproc.npz, or provide 'target_mask'/'*mask_path' in targets JSON."
            )

        while mask.ndim > 3:
            mask = mask.squeeze(0)
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask for patient {pid!r}, got shape {tuple(mask.shape)}")
        if not torch.isfinite(mask).all():
            raise ValueError(f"Non-finite values found in target mask for patient {pid!r}")

        # Convert to binary mask and add channel dimension: [1,1,D,H,W]
        mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        # Resample to decoder output shape if needed
        if tuple(mask.shape[-3:]) != tuple(self.deconv_output_shape):
            mask = F.interpolate(mask, size=self.deconv_output_shape, mode="nearest")
        # Blur to create softer targets for deconv training.
        if self.deconv_target_blur_sigma > 0:
            mask = gaussian_blur_3d(mask, sigma=self.deconv_target_blur_sigma)
        # Renormalise to [0,1]
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_max > mask_min:
            mask = (mask - mask_min) / (mask_max - mask_min)
        else:
            mask = mask.clamp(0.0, 1.0)
        if float(mask.max()) <= 0.0:
            raise ValueError(f"Empty deconv mask target for patient {pid!r} after preprocessing")

        out = mask.squeeze(0)  # [1,D,H,W]
        self._deconv_mask_cache[pid] = out
        return out

    def get_deconv_target_mask(self, pid: str) -> torch.Tensor:
        return self._prepare_deconv_target_mask(pid).clone()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        spikes, _ = self.base_dataset[idx]
        pid = self.patient_ids[idx]
        t = self.target_by_pid[pid]
        out_target = {
            "coord_target": t["mu"].clone(),
            "hemi_target":  t["hemi_target"].clone(),
            "lobe_target":  t["lobe_target"].clone(),
            "hemi_mask":    t["hemi_mask"].clone(),
            "lobe_mask":    t["lobe_mask"].clone(),
        }

        if self.deconv_enabled:
            out_target["target_mask"] = self._prepare_deconv_target_mask(pid)

        return (
            spikes,
            out_target,
        )


def mil_multitask_collate(batch):
    """
    Collate function for :class:`MultiHeadTargetDataset`.

    Returns ``(spikes, mask, target_dict)`` where ``target_dict`` maps key
    names to batched tensors.
    """
    spikes_list, targets_list = zip(*batch)

    max_spikes = max(s.shape[0] for s in spikes_list)
    padded, masks = [], []

    for spikes in spikes_list:
        n, c, l = spikes.shape
        pad_n = max_spikes - n
        padded.append(torch.cat([spikes, torch.zeros(pad_n, c, l)], dim=0))
        m = torch.zeros(max_spikes)
        m[:n] = 1
        masks.append(m)

    out_targets = {
        "coord_target": torch.stack([t["coord_target"] for t in targets_list], dim=0),
        "hemi_target":  torch.stack([t["hemi_target"]  for t in targets_list], dim=0),
        "lobe_target":  torch.stack([t["lobe_target"]  for t in targets_list], dim=0),
        "hemi_mask":    torch.stack([t["hemi_mask"]    for t in targets_list], dim=0),
        "lobe_mask":    torch.stack([t["lobe_mask"]    for t in targets_list], dim=0),
    }

    if "target_mask" in targets_list[0]:
        out_targets["target_mask"] = torch.stack([t["target_mask"] for t in targets_list], dim=0)

    return (
        torch.stack(padded, dim=0),   # (B, max_spikes, C, L)
        torch.stack(masks, dim=0),    # (B, max_spikes)
        out_targets,
    )


if __name__ == "__main__":
    data_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\spikes\\"
    patient_ids = ["RESP0134", "RESP0306"]
    patient_files = [data_dir + "RESP0134_spikes_1-70Hz.npy", data_dir + "RESP0306_spikes_1-70Hz.npy"]
    
    # Example for multi-label: each patient can have multiple conditions
    patient_labels = [[0.0, 0.0, 0.1, 0.9], [0.0, 0.9, 0.1, 0.0]]

    dataset = PatientMILSpikeDataset(
        patient_ids, patient_files, patient_labels, multi_label=True, training=True
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=mil_collate, shuffle=True)

    plot_first_batch = True
    for batch_windows, batch_masks, batch_labels in loader:
        print(f"Windows shape: {batch_windows.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Labels: {batch_labels}")

        if plot_first_batch:
            from preprocessing.eeg.create_eeg_ds import quality_control_plot  # type: ignore

            spikes = batch_windows[0, batch_masks[0].bool()].numpy()
            spike_window_sec = (0.25 * spikes.shape[-1] / 256.0, 0.75 * spikes.shape[-1] / 256.0)
            quality_control_plot(
                [spikes], np.arange(len(spikes)),
                [f"Ch {i + 1}" for i in range(spikes.shape[1])], "demo", spike_window_sec=spike_window_sec,
                n_samples=20, plot_spike_window=False,
                output_path="L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\eeg_bag_plot.png"
                )
            plot_first_batch = False

    # -----------------------------------------------
    # FlatSpikeEncoderPretrainDataset smoke test
    # -----------------------------------------------
    flat_data_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\spikes\\flat_dataset"
    flat_ds = FlatSpikeEncoderPretrainDataset(
        flat_data_dir=flat_data_dir,
        patient_ids=["RESP0134", "RESP0306"],
        training=True,
        window_size=128,
        max_offset=16,
        max_segments_per_patient=1000,
    )

    if len(flat_ds) > 0:
        flat_loader = DataLoader(
            flat_ds,
            batch_size=32,
            shuffle=True,
            collate_fn=flat_spike_pretrain_collate,
        )
        segments, targets = next(iter(flat_loader))
        print(f"Flat segments shape:       {segments.shape}")
        print(f"channel_target shape:      {targets['channel_target'].shape}")
        print(f"spike_target shape:        {targets['spike_target'].shape}")
        print(f"perception shape:          {targets['perception'].shape}")
        print(f"is_thresholded_spike shape:{targets['is_thresholded_spike'].shape}")

        from preprocessing.eeg.create_eeg_ds import quality_control_plot  # type: ignore

        quality_control_plot(
            [segments.numpy()],
            np.arange(len(segments)),
            [f"Ch {i + 1}" for i in range(segments.shape[1])],
            "flat_demo",
            spike_window_sec=(0.25, 0.75),
            n_samples=20,
            plot_spike_window=False,
            output_path="L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\eeg_flat_bag_plot.png",
        )
    else:
        print("No flat segments found (flat_data_dir may not exist yet).")
