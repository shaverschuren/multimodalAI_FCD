from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import numpy as np
import nibabel as nib
import pandas as pd
import pyedflib
from tqdm.auto import tqdm

if __package__:
    _dicom_meta = import_module(f"{__package__}.get_dicom_meta")
else:
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    _dicom_meta = import_module("get_dicom_meta")

read_dicom_metadata = _dicom_meta.read_dicom_metadata
resolve_raw_mri_dicom_folder = _dicom_meta.resolve_raw_mri_dicom_folder


# =============================================================================
# USER CONFIG
# =============================================================================

MANIFEST_CSV = Path(
    r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\data_availability\BIDS_conversion\source_manifest.csv"
)
BIDS_ROOT = Path(
    r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\BIDS_dataset"
)

DRY_RUN = False
OVERWRITE = False
# If True, skip copying data payload files/directories and only (re)write metadata sidecars/log tables.
SKIP_DATA_COPY = False

DEFAULT_TASK_LABEL = "rest"
DEFAULT_POWER_LINE_FREQUENCY = 50
DEFAULT_EEG_REFERENCE = "n/a"
DEFAULT_SOFTWARE_FILTERS: str | dict[str, Any] = "n/a"
DEFAULT_HARDWARE_FILTERS: str | dict[str, Any] = "n/a"

EEG_CONVERSION_SCRIPT_SOURCE = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\src\preprocessing\eeg\preprocessing_convertEDF.m"
)
EEG_CONVERSION_SCRIPT_BIDS_PATH = BIDS_ROOT / "code" / "eeg" / EEG_CONVERSION_SCRIPT_SOURCE.name
EEG_CONVERSION_LOG_SOURCE_DIR = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata\conversion_logs")
EEG_CONVERSION_LOG_BIDS_DIR = BIDS_ROOT / "code" / "eeg_conversion_logs"

DEFAULT_SESSION_BY_SOURCE_CATEGORY = {
    "raw_eeg_edf": "ses-preop",
    "preprocessed_eeg": "ses-preop",
    "eeg_spikes": "ses-preop",
    "preprocessed_mri": "ses-preop",
    "freesurfer_fastsurfer": "ses-preop",
}

DERIVATIVE_PIPELINES = {
    "ground_truth": "resection-masks",
    "preprocessed_mri": "mri-preproc",
    "freesurfer_fastsurfer": "freesurfer-fastsurfer",
    "preprocessed_eeg": "eeg-preproc",
    "eeg_spikes": "eeg-spikes",
}

GROUND_TRUTH_DATASET_SUMMARY_DIR = "summary"
GROUND_TRUTH_DATASET_FIGURE_DIR = "figures"
GROUND_TRUTH_SUBJECT_AUX_DIR = "auxiliary"

IGNORED_COPY_FILENAMES = {
    "thumbs.db",
    "ehthumbs.db",
    "desktop.ini",
    ".ds_store",
}


# =============================================================================
# Regexes for inferring channel types from EDF labels.
# =============================================================================

EEG_1005_RE = re.compile(
    r"^(?:"
    r"FPZ?|AF|F|FC|FT|T|C|CP|TP|P|PO|O"
    r")"
    r"(?:Z|[0-9]{1,2})$",
    re.IGNORECASE,
)

REF_LIKE_RE = re.compile(
    r"^(?:A1|A2|M1|M2|TP9|TP10)$",
    re.IGNORECASE,
)

NON_SIGNAL_RE = re.compile(
    r"^(?:GND|GROUND|REF|REFERENCE|CMS|DRL|N|NZ|NASION|LPA|RPA)$",
    re.IGNORECASE,
)


# =============================================================================
# DATA MODEL / LOGGING
# =============================================================================

@dataclass
class CopyRecord:
    subject_id: str
    source_category: str
    source_path: str
    destination_path: str
    action: str
    status: str
    message: str


COPY_LOG: list[CopyRecord] = []
WARNING_LOG: list[str] = []

ANSI_ORANGE = "\033[38;5;208m"
ANSI_RESET = "\033[0m"


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def log_action(
    row: pd.Series,
    destination: Path,
    action: str,
    status: str,
    message: str = "",
) -> None:
    COPY_LOG.append(
        CopyRecord(
            subject_id=str(row.get("subject_id", "")),
            source_category=str(row.get("source_category", "")),
            source_path=str(row.get("source_path", "")),
            destination_path=str(destination),
            action=action,
            status=status,
            message=message,
        )
    )


def warn(message: str) -> None:
    WARNING_LOG.append(message)
    log(f"{ANSI_ORANGE}WARNING: {message}{ANSI_RESET}")


def log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def ensure_dir(path: Path) -> None:
    if DRY_RUN:
        return
    path.mkdir(parents=True, exist_ok=True)


def should_ignore_copy_path(path: Path) -> bool:
    return path.name.lower() in IGNORED_COPY_FILENAMES


def copytree_ignore_names(_src: str, names: list[str]) -> set[str]:
    return {name for name in names if name.lower() in IGNORED_COPY_FILENAMES}


def write_text(path: Path, text: str) -> None:
    log(f"WRITE TEXT: {path}")
    if DRY_RUN:
        return
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    log(f"WRITE JSON: {path}")
    if DRY_RUN:
        return
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    log(f"WRITE TSV: {path}")
    if DRY_RUN:
        return
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    df.to_csv(path, sep="\t", index=False, na_rep="n/a")


def copy_file(src: Path, dst: Path, row: pd.Series, action: str = "copy_file") -> None:
    if SKIP_DATA_COPY:
        log_action(row, dst, action, "skipped_metadata_only", "Skipping data copy because SKIP_DATA_COPY=True")
        return

    if not src.exists():
        log_action(row, dst, action, "missing_source", "Source file does not exist")
        warn(f"Missing source file: {src}")
        return

    if should_ignore_copy_path(src):
        log_action(row, dst, action, "skipped_ignored", f"Ignored filename: {src.name}")
        return

    if dst.exists() and not OVERWRITE:
        log_action(row, dst, action, "skipped_exists", "Destination exists and OVERWRITE=False")
        return

    log(f"COPY: {src} -> {dst}")
    if DRY_RUN:
        log_action(row, dst, action, "dry_run", "Would copy file")
        return

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    log_action(row, dst, action, "copied", "")


def copy_directory(src: Path, dst: Path, row: pd.Series, action: str = "copy_directory") -> None:
    if SKIP_DATA_COPY:
        log_action(row, dst, action, "skipped_metadata_only", "Skipping directory copy because SKIP_DATA_COPY=True")
        return

    if not src.exists():
        log_action(row, dst, action, "missing_source", "Source directory does not exist")
        warn(f"Missing source directory: {src}")
        return

    if dst.exists():
        if OVERWRITE:
            log(f"REMOVE EXISTING DIRECTORY: {dst}")
            if not DRY_RUN:
                shutil.rmtree(dst)
        else:
            log_action(row, dst, action, "skipped_exists", "Destination exists and OVERWRITE=False")
            return

    log(f"COPYTREE: {src} -> {dst}")
    if DRY_RUN:
        log_action(row, dst, action, "dry_run", "Would copy directory")
        return

    ensure_dir(dst.parent)
    shutil.copytree(src, dst, ignore=copytree_ignore_names)
    log_action(row, dst, action, "copied", "")


def get_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def sidecar_json_path(image_path: Path) -> Path:
    if get_extension(image_path) == ".nii.gz":
        return image_path.with_suffix("").with_suffix(".json")
    return image_path.with_suffix(".json")


def _as_positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _sanitize_raw_mri_sidecar(sidecar: dict[str, Any], suffix: str) -> None:
    for key in [
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "DeviceSerialNumber",
        "SequenceName",
        "PulseSequenceType",
        "PulseSequenceDetails",
        "ReceiveCoilName",
        "ReceiveCoilActiveElements",
        "CoilCombinationMethod",
        "MatrixCoilMode",
    ]:
        value = sidecar.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            sidecar[key] = "_".join(cleaned) if cleaned else "n/a"
            continue
        text = str(value).strip()
        sidecar[key] = text if text else "n/a"

    for key in ["DwellTime", "EchoTime", "InversionTime", "RepetitionTime", "MagneticFieldStrength", "FlipAngle"]:
        value = _as_positive_float(sidecar.get(key))
        if value is None:
            sidecar.pop(key, None)
        else:
            sidecar[key] = value

    if "NonlinearGradientCorrection" in sidecar and not isinstance(sidecar["NonlinearGradientCorrection"], bool):
        sidecar.pop("NonlinearGradientCorrection", None)

    if "MRAcquisitionType" in sidecar:
        acq = str(sidecar["MRAcquisitionType"]).strip().upper()
        if acq in {"1D", "2D", "3D"}:
            sidecar["MRAcquisitionType"] = acq
        else:
            sidecar.pop("MRAcquisitionType", None)

    # Structural MRI sidecars should not carry SliceTiming.
    if suffix in {"T1w", "T2w", "FLAIR", "T2star", "PD", "PDT2"}:
        sidecar.pop("SliceTiming", None)


def _ordered_raw_mri_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    preferred_order = [
        "Modality", "SourceFile", "ConversionSoftware", "ConversionSoftwareVersion", "ConversionNotes",
        "DicomMetadataAvailable", "DicomMetadataReason", "SourceDicomDirectory",
        "Manufacturer", "ManufacturersModelName", "DeviceSerialNumber", "SoftwareVersions", "StationName",
        "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
        "BodyPart", "PatientPosition", "MRAcquisitionType",
        "PulseSequenceType", "SequenceName", "PulseSequenceDetails",
        "ScanningSequence", "SequenceVariant", "ScanOptions",
        "ReceiveCoilName", "ReceiveCoilActiveElements", "CoilCombinationMethod", "MatrixCoilMode",
        "MagneticFieldStrength", "FlipAngle", "RepetitionTime", "EchoTime", "InversionTime",
        "DwellTime", "PixelBandwidth", "EchoTrainLength", "SliceThickness", "SpacingBetweenSlices",
        "NonlinearGradientCorrection",
        "NiftiHeaderAvailable", "NiftiHeaderReason", "SpatialReference", "ImageOrientation",
        "NiftiSpatialShape", "NiftiVoxelSize", "NiftiSpatialUnits", "NiftiTimeUnits", "NumberOfVolumes",
        "AffineTransform",
        "StudyDate", "SeriesDate", "AcquisitionDate", "StudyTime", "SeriesTime", "AcquisitionTime",
    ]

    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in sidecar:
            ordered[key] = sidecar[key]

    for key, value in sidecar.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def bids_subject(subject_id: str) -> str:
    return f"sub-{subject_id}"


def clean_label(label: str) -> str:
    label = str(label)
    label = label.replace("ses-", "")
    label = label.replace("sub-", "")
    return "".join(ch for ch in label if ch.isalnum())


def norm_channel_label(label: str) -> str:
    if label is None:
        return ""

    s = str(label).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^(EEG|POL|POLY|CHAN|CH)[\s_\-:]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[\s_\-:]*(REF|AVG|AV|LE|M1|M2|A1|A2)$", "", s, flags=re.IGNORECASE)

    return s


def normalize_session(row: pd.Series) -> str:
    session = str(row.get("session", "ses-unknown"))
    if session and session != "nan" and session != "ses-unknown":
        return session

    source_category = str(row.get("source_category", ""))
    return DEFAULT_SESSION_BY_SOURCE_CATEGORY.get(source_category, "ses-unknown")


def bids_session_entity(session: str) -> str:
    return f"ses-{clean_label(session)}"


def run_entity(run_value: Any) -> str:
    if pd.isna(run_value) or str(run_value).strip() in {"", "None", "nan"}:
        return ""

    run = str(run_value).strip()
    if run.startswith("run-"):
        return run
    return f"run-{clean_label(run)}"


def make_bids_stem(row: pd.Series, suffix: str, extra_entities: Optional[list[str]] = None) -> str:
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    entities = [subject, session]

    if extra_entities:
        entities.extend([e for e in extra_entities if e])

    run = run_entity(row.get("run"))
    if run:
        entities.append(run)

    entities.append(suffix)
    return "_".join(entities)


def relative_to_bids(path: Path) -> str:
    try:
        return str(path.relative_to(BIDS_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# =============================================================================
# EDF HEADER HELPERS
# =============================================================================

def read_edf_metadata(edf_path: Path) -> dict[str, Any]:
    if pyedflib is None:
        return {
            "available": False,
            "reason": "pyedflib is not installed",
            "labels": [],
            "sample_frequencies": [],
            "startdate": "n/a",
            "starttime": "n/a",
            "n_channels": "n/a",
            "duration": "n/a",
        }

    try:
        reader = pyedflib.EdfReader(str(edf_path))
        n_channels = int(reader.signals_in_file)
        duration = float(reader.file_duration)
        labels = list(reader.getSignalLabels())
        sample_frequencies = [float(x) for x in reader.getSampleFrequencies()]
        transducer = [reader.getTransducer(chn) for chn in range(n_channels)]
        physical_dimension = [reader.getPhysicalDimension(chn) for chn in range(n_channels)]
        prefilter = [reader.getPrefilter(chn) for chn in range(n_channels)]

        start_date = "n/a"
        start_time = "n/a"
        try:
            if hasattr(reader, "getStartdatetime"):
                start_dt = reader.getStartdatetime()
            elif hasattr(reader, "getStartDatetime"):
                start_dt = reader.getStartDatetime()
            else:
                start_dt = None

            if start_dt is not None and hasattr(start_dt, "strftime"):
                start_date = start_dt.strftime("%Y-%m-%d")
                start_time = start_dt.strftime("%H:%M:%S")
        except Exception as exc:
            warn(f"Could not parse EDF start date/time from {edf_path}: {exc}")

        reader.close()

        return {
            "available": True,
            "labels": labels,
            "sample_frequencies": sample_frequencies,
            "transducer": transducer,
            "physical_dimension": physical_dimension,
            "prefilter": prefilter,
            "startdate": start_date,
            "starttime": start_time,
            "n_channels": n_channels,
            "duration": duration,
        }
    except Exception as exc:  # pragma: no cover
        warn(f"Could not read EDF metadata from {edf_path}: {exc}")
        return {
            "available": False,
            "reason": str(exc),
            "labels": [],
            "sample_frequencies": [],
            "startdate": "n/a",
            "starttime": "n/a",
            "n_channels": "n/a",
            "duration": "n/a",
        }


def infer_eeg_channel_type(label: str) -> str:
    raw = "" if label is None else str(label)
    lower = raw.lower()
    compact = norm_channel_label(raw)

    if re.search(
        r"(trigger|trig|status|marker|mark|event|mrk|mkr|stim|sync|ttl|annot|annotation|edf annotations?)",
        lower,
    ):
        return "TRIG"

    if re.search(r"(ecg|ekg|hart|cardio)", lower):
        return "ECG"

    if re.search(r"(eog|oog|eye|ocul)", lower):
        return "EOG"

    if re.search(r"(emg|myo|muscle|spier|chin|kin)", lower):
        return "EMG"

    if re.search(
        r"(resp|respiration|breath|breathing|airflow|flow|snore|"
        r"adem|ademhaling|thor|thorax|chest|borst|abd|abdomen|abdominaal|buik)",
        lower,
    ):
        return "MISC"

    if re.search(
        r"(spo2|sao2|sat|saturatie|oxygen|oximeter|oximetry|"
        r"pulse|puls|pols|pleth|ppg|beat|hartslag)",
        lower,
    ):
        return "MISC"

    if re.search(r"(temp|temperature|temperatuur)", lower):
        return "MISC"

    if re.search(r"(photic|flash|flits|lamp)", lower):
        return "MISC"

    if re.search(r"(misc|aux|dc|analog|analogue|sensor|unknown|onbekend)", lower):
        return "MISC"

    if NON_SIGNAL_RE.match(compact):
        return "MISC"

    if REF_LIKE_RE.match(compact):
        return "EEG"

    if EEG_1005_RE.match(compact):
        return "EEG"

    return "MISC"


def make_channels_tsv(
    edf_metadata: dict[str, Any],
    channel_status: Optional[dict[str, tuple[str, str]]] = None,
) -> pd.DataFrame:
    labels = edf_metadata.get("labels", []) or []
    sample_frequencies = edf_metadata.get("sample_frequencies", []) or []
    physical_dimension = edf_metadata.get("physical_dimension", []) or []
    prefilter = edf_metadata.get("prefilter", []) or []
    status_lookup = channel_status or {}

    rows = []
    for idx, label in enumerate(labels):
        normalized_label = norm_channel_label(label)
        channel_type = infer_eeg_channel_type(label)
        if channel_type == "EEG":
            status, status_description = status_lookup.get(normalized_label, ("good", "n/a"))
        else:
            status, status_description = "n/a", "n/a"
        rows.append(
            {
                "name": label,
                "type": channel_type,
                "units": physical_dimension[idx] if idx < len(physical_dimension) and physical_dimension[idx] else "n/a",
                "sampling_frequency": sample_frequencies[idx] if idx < len(sample_frequencies) else "n/a",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "notch": "n/a",
                "status": status,
                "status_description": status_description,
                "description": prefilter[idx] if idx < len(prefilter) and prefilter[idx] else "n/a",
            }
        )

    if not rows:
        rows.append(
            {
                "name": "n/a",
                "type": "EEG",
                "units": "n/a",
                "sampling_frequency": "n/a",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "notch": "n/a",
                "status": "n/a",
                "status_description": "EDF metadata could not be read; install pyedflib for channel table generation",
                "description": "n/a",
            }
        )

    return pd.DataFrame(rows)


def parse_channel_list(raw_value: Any) -> set[str]:
    if raw_value is None or pd.isna(raw_value):
        return set()

    text = str(raw_value).strip()
    if not text or text.lower() == "n/a":
        return set()

    return {
        normalized
        for part in re.split(r"[;,]", text)
        if (normalized := norm_channel_label(part))
    }


def normalize_input_block_id(raw_value: Any) -> str:
    if raw_value is None or pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    if not text:
        return ""

    return Path(text).stem.strip().lower()


def find_eeg_conversion_tsv(subject_id: str, input_file: str) -> Optional[Path]:
    subject_prefix = f"{subject_id}_"
    target_block_id = normalize_input_block_id(input_file)

    if not target_block_id:
        warn(f"Empty EEG input_file identifier for subject {subject_id}: {input_file}")
        return None

    if not EEG_CONVERSION_LOG_SOURCE_DIR.exists():
        warn(f"EEG conversion log directory does not exist: {EEG_CONVERSION_LOG_SOURCE_DIR}")
        return None

    found_candidate = False
    for tsv_path in sorted(EEG_CONVERSION_LOG_SOURCE_DIR.glob(f"{subject_prefix}*2edf_conversion.tsv")):
        found_candidate = True
        try:
            conversion_df = pd.read_csv(tsv_path, sep="\t")
        except Exception as exc:
            warn(f"Could not read EEG conversion TSV {tsv_path}: {exc}")
            continue

        if "input_file" not in conversion_df.columns:
            warn(f"EEG conversion TSV is missing the input_file column: {tsv_path}")
            continue

        input_block_ids = conversion_df["input_file"].apply(normalize_input_block_id)
        if (input_block_ids == target_block_id).any():
            return tsv_path

    if not found_candidate:
        warn(
            f"No EEG conversion TSV found for subject {subject_id} in {EEG_CONVERSION_LOG_SOURCE_DIR}"
        )
    else:
        warn(
            f"No matching EEG conversion TSV row found for subject {subject_id} and input block {target_block_id}"
        )

    return None


def infer_eeg_channel_status(subject_id: str, input_file: str) -> dict[str, tuple[str, str]]:
    tsv_path = find_eeg_conversion_tsv(subject_id, input_file)
    if tsv_path is None:
        return {}

    target_block_id = normalize_input_block_id(input_file)

    conversion_df = pd.read_csv(tsv_path, sep="\t")

    match = conversion_df[conversion_df["input_file"].apply(normalize_input_block_id) == target_block_id]
    if match.empty:
        warn(
            f"EEG conversion TSV did not contain a matching input_file row after lookup succeeded: {tsv_path}"
        )
        return {}

    row = match.iloc[0]
    channel_status: dict[str, tuple[str, str]] = {}

    for label in parse_channel_list(row.get("good_channels")):
        channel_status[label] = ("good", "n/a")

    for label in parse_channel_list(row.get("bad_channels")):
        channel_status[label] = ("bad", "Marked bad during preprocessing_convertEDF.m")

    return channel_status


def make_eeg_json(edf_metadata: dict[str, Any], source_path: Path, derivative: bool = False) -> dict[str, Any]:
    sample_frequencies = edf_metadata.get("sample_frequencies", []) or []
    unique_fs = sorted(set(float(x) for x in sample_frequencies))

    if len(unique_fs) == 1:
        sampling_frequency: Any = unique_fs[0]
    elif len(unique_fs) > 1:
        sampling_frequency = sorted(sample_frequencies)[len(sample_frequencies) // 2]
        warn(
            f"Multiple sample frequencies ({unique_fs}) in {source_path}. This is likely due to an annotation or other non-EEG channel. "
            f"Using median ({sampling_frequency}) for EEG JSON and writing per-channel values in channels.tsv"
        )
    else:
        sampling_frequency = "n/a"
        warn(
            f"SamplingFrequency could not be determined for {source_path}. "
            "Install pyedflib or manually add this value for strict BIDS validation."
        )

    payload = {
        "TaskName": DEFAULT_TASK_LABEL,
        "TaskDescription": "Resting-state scalp EEG recording.",
        "Manufacturer": "n/a",
        "ManufacturersModelName": "n/a",
        "InstitutionName": "n/a",
        "InstitutionAddress": "n/a",
        "InstitutionalDepartmentName": "n/a",
        "DeviceSerialNumber": "n/a",
        "SoftwareVersions": "n/a",
        "SamplingFrequency": sampling_frequency,
        "PowerLineFrequency": DEFAULT_POWER_LINE_FREQUENCY,
        "SoftwareFilters": DEFAULT_SOFTWARE_FILTERS,
        "HardwareFilters": DEFAULT_HARDWARE_FILTERS,
        "EEGReference": DEFAULT_EEG_REFERENCE,
        "EEGGround": "n/a",
        "EEGPlacementScheme": "n/a",
        "Instructions": "n/a",
        "SubjectArtefactDescription": "n/a",
        "CapManufacturer": "n/a",
        "CapManufacturersModelName": "n/a",
        "CogAtlasID": "n/a",
        "CogPOID": "n/a",
        "RecordingType": "continuous" if not derivative else "discontinuous",
        "RecordingDuration": edf_metadata.get("duration", "n/a"),
        "RecordingStartDate": edf_metadata.get("startdate", "n/a"),
        "RecordingStartTime": edf_metadata.get("starttime", "n/a"),
        "EEGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EEG"),
        "EOGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EOG"),
        "ECGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "ECG"),
        "EMGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EMG"),
        "MISCChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "MISC"),
        "MiscChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "MISC"),
        "TriggerChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "TRIG"),
        "SourceFile": str(source_path),
        "ConversionNotes": "Events may be missing due to loss in conversion from TRC/SIG-STS files.",
    }

    if derivative:
        payload["Description"] = "Preprocessed and/or concatenated EEG derivative copied from source pipeline."

    return payload


def can_write_sidecars_for_target(dst: Path) -> bool:
    if DRY_RUN:
        return True
    if not SKIP_DATA_COPY:
        return True
    if dst.exists():
        return True

    warn(f"Skipping sidecar rewrite because target data file does not exist in metadata-only mode: {dst}")
    return False


def copy_eeg_conversion_provenance(manifest: pd.DataFrame) -> None:
    eeg_rows = manifest[manifest["source_category"].isin({"raw_eeg_edf", "preprocessed_eeg"})]
    subject_ids = sorted({str(subject_id) for subject_id in eeg_rows["subject_id"].astype(str)})

    script_row = pd.Series(
        {
            "subject_id": "dataset",
            "source_category": "eeg_conversion_provenance",
            "source_path": str(EEG_CONVERSION_SCRIPT_SOURCE),
        }
    )
    copy_file(EEG_CONVERSION_SCRIPT_SOURCE, EEG_CONVERSION_SCRIPT_BIDS_PATH, script_row, action="copy_code")

    if not subject_ids:
        return

    seen_paths: set[Path] = set()
    for subject_id in subject_ids:
        for tsv_path in sorted(EEG_CONVERSION_LOG_SOURCE_DIR.glob(f"{subject_id}_*2edf_conversion.tsv")):
            if tsv_path in seen_paths:
                continue
            seen_paths.add(tsv_path)
            copy_file(
                tsv_path,
                EEG_CONVERSION_LOG_BIDS_DIR / tsv_path.name,
                script_row,
                action="copy_code",
            )


# =============================================================================
# NIFTI HEADER HELPERS
# =============================================================================

def read_nifti_metadata(nifti_path: Path) -> dict[str, Any]:
    if nib is None:
        return {
            "NiftiHeaderAvailable": False,
            "NiftiHeaderReason": "nibabel is not installed",
        }

    try:
        img = nib.load(str(nifti_path))
        header = img.header
        shape = list(header.get_data_shape())
        zooms = header.get_zooms()
        spatial_zooms = [float(z) for z in zooms[:3]]
        xyz_units, time_units = header.get_xyzt_units()
        axcodes = nib.aff2axcodes(img.affine)

        payload: dict[str, Any] = {
            "NiftiHeaderAvailable": True,
            "SpatialReference": "orig",
            "ImageOrientation": "".join(axcodes),
            "NiftiSpatialShape": shape[:3],
            "NiftiVoxelSize": spatial_zooms,
            "NiftiSpatialUnits": xyz_units or "unknown",
            "NiftiTimeUnits": time_units or "unknown",
            "AffineTransform": np.asarray(img.affine, dtype=float).tolist(),
            "NumberOfVolumes": int(shape[3]) if len(shape) > 3 else 1,
        }

        if len(zooms) > 3 and float(zooms[3]) > 0:
            tr_seconds = float(zooms[3])
            if time_units == "msec":
                tr_seconds = tr_seconds / 1000.0
            elif time_units == "usec":
                tr_seconds = tr_seconds / 1_000_000.0
            payload["RepetitionTime"] = tr_seconds

        if len(spatial_zooms) >= 3 and spatial_zooms[2] > 0:
            payload["SliceThickness"] = spatial_zooms[2]

        return payload
    except Exception as exc:  # pragma: no cover
        warn(f"Could not read NIfTI metadata from {nifti_path}: {exc}")
        return {
            "NiftiHeaderAvailable": False,
            "NiftiHeaderReason": str(exc),
        }


# =============================================================================
# CONVERTERS
# =============================================================================

def convert_raw_mri(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    suffix = str(row.get("bids_suffix", "UNKNOWN"))

    if suffix == "UNKNOWN" or not suffix:
        suffix = "T1w"
        warn(f"Unknown MRI suffix for {src}; defaulting to T1w. Check manually.")

    stem = make_bids_stem(row, suffix=suffix)
    dst = BIDS_ROOT / subject / session / "anat" / f"{stem}{get_extension(src)}"
    copy_file(src, dst, row)

    if not can_write_sidecars_for_target(dst):
        return

    sidecar = {
        "Modality": "MR",
        "SourceFile": str(src),
        "ConversionSoftware": "custom-source-to-bids-conversion",
        "ConversionSoftwareVersion": "dicom2nifti-2.6.2",
        "ConversionNotes": "DICOM to NIfTI conversion performed using `dicom2nifti-2.6.2` via `scripts_other/create_nifti_dataset.py`. Acquisition metadata was read from the NIfTI header and source DICOM folder where available.",
    }
    if get_extension(dst) in {".nii", ".nii.gz"}:
        nifti_for_metadata = dst if dst.exists() and not DRY_RUN else src
        sidecar.update(read_nifti_metadata(nifti_for_metadata))

        dicom_dir = resolve_raw_mri_dicom_folder(src)
        if dicom_dir is None:
            warn(f"Could not infer DICOM source folder from MRI NIfTI filename: {src.name}")
            sidecar.update(
                {
                    "DicomMetadataAvailable": False,
                    "DicomMetadataReason": "Could not infer source DICOM folder from NIfTI filename",
                }
            )
        else:
            sidecar.update(read_dicom_metadata(dicom_dir))

        sidecar.setdefault("InstitutionName", "n/a")
        sidecar.setdefault("InstitutionAddress", "n/a")
        sidecar.setdefault("InstitutionalDepartmentName", "n/a")
        sidecar.setdefault("DeviceSerialNumber", "n/a")
        sidecar.setdefault("SequenceName", "n/a")
        sidecar.setdefault("MRAcquisitionType", "n/a")
        sidecar.setdefault("PulseSequenceType", "n/a")
        sidecar.setdefault("PulseSequenceDetails", "n/a")
        sidecar.setdefault("ReceiveCoilName", "n/a")
        sidecar.setdefault("ReceiveCoilActiveElements", "n/a")
        sidecar.setdefault("CoilCombinationMethod", "n/a")
        sidecar.setdefault("MatrixCoilMode", "n/a")
        _sanitize_raw_mri_sidecar(sidecar, suffix)
        sidecar = _ordered_raw_mri_sidecar(sidecar)

    write_json(sidecar_json_path(dst), sidecar)


def convert_raw_eeg(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    task_entity = f"task-{clean_label(DEFAULT_TASK_LABEL)}"

    stem = make_bids_stem(row, suffix="eeg", extra_entities=[task_entity])
    dst = BIDS_ROOT / subject / session / "eeg" / f"{stem}.edf"
    copy_file(src, dst, row)

    if not can_write_sidecars_for_target(dst):
        return

    edf_metadata = read_edf_metadata(src)
    eeg_json = make_eeg_json(edf_metadata, source_path=src, derivative=False)
    channel_status = infer_eeg_channel_status(str(row["subject_id"]), src.name)
    channels_tsv = make_channels_tsv(edf_metadata, channel_status=channel_status)

    write_json(dst.with_suffix(".json"), eeg_json)
    write_tsv(dst.with_name(dst.name.replace("_eeg.edf", "_channels.tsv")), channels_tsv)


def update_scans_tsv(raw_outputs: list[Path]) -> None:
    grouped: dict[tuple[Path, str, str], list[Path]] = {}

    for path in raw_outputs:
        try:
            rel = path.relative_to(BIDS_ROOT)
        except ValueError:
            continue

        parts = rel.parts
        if len(parts) < 4:
            continue
        subject, session = parts[0], parts[1]
        session_dir = BIDS_ROOT / subject / session
        grouped.setdefault((session_dir, subject, session), []).append(path)

    for (session_dir, subject, session), paths in grouped.items():
        rows = []
        for path in sorted(paths):
            # Per BIDS, scans.tsv `filename` is relative to the session directory,
            # not dataset-root relative.
            try:
                filename_value = str(path.relative_to(session_dir)).replace("\\", "/")
            except ValueError:
                filename_value = relative_to_bids(path)
            rows.append(
                {
                    "filename": filename_value,
                    "acq_time": "n/a",
                }
            )
        scans_path = session_dir / f"{subject}_{session}_scans.tsv"
        write_tsv(scans_path, pd.DataFrame(rows))


def copy_cortex_picture(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = "ses-intraop"
    dst = BIDS_ROOT / "sourcedata" / subject / session / "photos" / src.name
    copy_file(src, dst, row, action="copy_sourcedata")


def copy_ground_truth(row: pd.Series) -> None:
    src = Path(row["source_path"])
    pipeline = DERIVATIVE_PIPELINES["ground_truth"]

    if str(row["subject_id"]) == "dataset":
        if src.suffix.lower() == ".png":
            dst = BIDS_ROOT / "derivatives" / pipeline / GROUND_TRUTH_DATASET_FIGURE_DIR / src.name
        else:
            dst = BIDS_ROOT / "derivatives" / pipeline / GROUND_TRUTH_DATASET_SUMMARY_DIR / src.name
        copy_file(src, dst, row, action="copy_derivative")
        return

    subject = bids_subject(str(row["subject_id"]))
    session = "ses-postop"

    if str(row.get("modality")) == "resection_mask":
        stem = f"{subject}_{session}_desc-resection_mask"
        dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "anat" / f"{stem}{get_extension(src)}"
        copy_file(src, dst, row, action="copy_derivative")
        if not can_write_sidecars_for_target(dst):
            return
        write_json(
            sidecar_json_path(dst),
            {
                "Description": "Final ground-truth volumetric resection mask",
                "SourceFile": str(src),
                "Type": "Manual or pipeline-derived segmentation mask",
                "SpatialReference": "n/a",
            },
        )
    else:
        if get_extension(src) in {".nii", ".nii.gz"}:
            dst_dir = BIDS_ROOT / "derivatives" / pipeline / subject / session / "anat"
        elif src.suffix.lower() == ".png":
            dst_dir = BIDS_ROOT / "derivatives" / pipeline / subject / session / "figures"
        else:
            dst_dir = BIDS_ROOT / "derivatives" / pipeline / subject / session / GROUND_TRUTH_SUBJECT_AUX_DIR

        dst = dst_dir / src.name
        copy_file(src, dst, row, action="copy_derivative_auxiliary")

        if get_extension(dst) in {".nii", ".nii.gz"}:
            if not can_write_sidecars_for_target(dst):
                return
            sidecar_payload: dict[str, Any] = {
                "Description": f"Ground-truth derivative artifact copied from {src.name}",
                "SourceFile": str(src),
                "SpatialReference": "n/a",
            }
            if src.name.endswith("_pic2mri_smooth.nii.gz"):
                sidecar_payload["Description"] = "Smoothed probability map used for ground-truth harmonisation"
            elif src.name.endswith("_pic2mri_harmonised.nii.gz"):
                sidecar_payload["Description"] = "Harmonised pic2mri mask"
            elif src.name.endswith("_gt_mask_harmonised.nii.gz"):
                sidecar_payload["Description"] = "Harmonised ground-truth mask"

            write_json(sidecar_json_path(dst), sidecar_payload)


def copy_preprocessed_mri(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    pipeline = DERIVATIVE_PIPELINES["preprocessed_mri"]

    suffix = str(row.get("bids_suffix", "UNKNOWN"))
    if suffix == "UNKNOWN" or not suffix:
        dst_name = src.name
    else:
        stem = f"{subject}_{session}_desc-preproc_{suffix}"
        dst_name = f"{stem}{get_extension(src)}"

    dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "anat" / dst_name
    copy_file(src, dst, row, action="copy_derivative")

    if not can_write_sidecars_for_target(dst):
        return

    if get_extension(dst) in {".nii", ".nii.gz"}:
        sidecar = {
            "Description": "Preprocessed MRI derivative copied from the upstream preprocessing pipeline",
            "SourceFile": str(src),
            "SpatialReference": "n/a",
        }
        nifti_for_metadata = dst if dst.exists() and not DRY_RUN else src
        sidecar.update(read_nifti_metadata(nifti_for_metadata))
        write_json(sidecar_json_path(dst), sidecar)


def copy_freesurfer(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    pipeline = DERIVATIVE_PIPELINES["freesurfer_fastsurfer"]
    dst = BIDS_ROOT / "derivatives" / pipeline / subject
    copy_directory(src, dst, row, action="copy_derivative_directory")


def copy_preprocessed_eeg(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    pipeline = DERIVATIVE_PIPELINES["preprocessed_eeg"]
    task_entity = f"task-{clean_label(DEFAULT_TASK_LABEL)}"
    desc_entity = "desc-concatpreproc"

    stem = make_bids_stem(row, suffix="eeg", extra_entities=[task_entity, desc_entity])
    dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "eeg" / f"{stem}.edf"
    copy_file(src, dst, row, action="copy_derivative")

    if not can_write_sidecars_for_target(dst):
        return

    edf_metadata = read_edf_metadata(src)
    write_json(dst.with_suffix(".json"), make_eeg_json(edf_metadata, source_path=src, derivative=True))
    write_tsv(dst.with_name(dst.name.replace("_eeg.edf", "_channels.tsv")), make_channels_tsv(edf_metadata))


def copy_eeg_spikes(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    pipeline = DERIVATIVE_PIPELINES["eeg_spikes"]

    band = src.stem.replace(str(row["subject_id"]), "").strip("_")
    band_label = clean_label(band.replace("-", "to")) or "spikes"

    stem = f"{subject}_{session}_desc-{band_label}_eeg"
    dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "eeg" / f"{stem}.npy"
    copy_file(src, dst, row, action="copy_derivative")

    if not can_write_sidecars_for_target(dst):
        return

    write_json(
        dst.with_suffix(".json"),
        {
            "Description": "Processed EEG spike recording derivative stored as NumPy array",
            "SourceFile": str(src),
            "ArrayFormat": "NumPy .npy",
            "FrequencyBandFromFilename": band,
            "Units": "n/a",
            "ConversionNotes": "This file contains EEG spike data (1s segments 256Hz, detected with Persyst 15 with Perception Score > 0.9) stored as a NumPy array [n_segments x n_channels x n_samples]. Units are per-subject z-score.",
        },
    )


def dispatch_row(row: pd.Series) -> None:
    category = str(row["source_category"])

    if category == "raw_mri_preop" or category == "raw_mri_postop":
        convert_raw_mri(row)
    elif category == "raw_eeg_edf":
        convert_raw_eeg(row)
    elif category == "intraoperative_cortex_picture":
        copy_cortex_picture(row)
    elif category == "ground_truth" or category == "ground_truth_dataset":
        copy_ground_truth(row)
    elif category == "preprocessed_mri":
        copy_preprocessed_mri(row)
    elif category == "freesurfer_fastsurfer":
        copy_freesurfer(row)
    elif category == "preprocessed_eeg":
        copy_preprocessed_eeg(row)
    elif category == "eeg_spikes":
        copy_eeg_spikes(row)
    else:
        warn(f"Unknown source_category={category}; skipping row with source_path={row.get('source_path')}")


# =============================================================================
# LOG WRITING
# =============================================================================

def write_logs() -> None:
    log_dir = BIDS_ROOT / "code" / "conversion_logs"
    ensure_dir(log_dir)

    if COPY_LOG:
        df = pd.DataFrame([record.__dict__ for record in COPY_LOG])
        if not DRY_RUN:
            df.to_csv(log_dir / "copy_log.tsv", sep="\t", index=False)
        log(f"WRITE LOG: {log_dir / 'copy_log.tsv'}")

    if WARNING_LOG:
        text = "\n".join(WARNING_LOG) + "\n"
        write_text(log_dir / "warnings.txt", text)


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def validate_manifest(manifest: pd.DataFrame) -> None:
    required_columns = {
        "subject_id",
        "source_category",
        "datatype",
        "session",
        "modality",
        "source_path",
        "source_filename",
        "file_extension",
        "bids_derivative",
        "bids_suffix",
        "run",
        "notes",
    }
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")


def run_data_conversion(manifest: pd.DataFrame) -> None:
    validate_manifest(manifest)

    log(f"Rows: {len(manifest)}")
    log(f"BIDS root: {BIDS_ROOT}")
    log(f"DRY_RUN: {DRY_RUN}")
    log(f"OVERWRITE: {OVERWRITE}")
    log(f"SKIP_DATA_COPY: {SKIP_DATA_COPY}")

    ensure_dir(BIDS_ROOT)

    row_iterator = manifest.iterrows()
    if tqdm is not None:
        row_iterator = tqdm(row_iterator, total=len(manifest), desc="Converting manifest rows", unit="row")

    for _, row in row_iterator:
        dispatch_row(row)

    raw_outputs = [
        Path(record.destination_path)
        for record in COPY_LOG
        if record.source_category in {"raw_mri_preop", "raw_mri_postop", "raw_eeg_edf"}
        and record.status in {"copied", "dry_run", "skipped_exists", "skipped_metadata_only"}
    ]
    update_scans_tsv(raw_outputs)

    copy_eeg_conversion_provenance(manifest)

    write_logs()

    log("Done.")
    log(f"Copy records: {len(COPY_LOG)}")
    log(f"Warnings: {len(WARNING_LOG)}")
    if DRY_RUN:
        log("This was a dry run. Set DRY_RUN = False to actually write/copy files.")


def main() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)
    log(f"Loaded manifest: {MANIFEST_CSV}")
    run_data_conversion(manifest)


if __name__ == "__main__":
    main()
