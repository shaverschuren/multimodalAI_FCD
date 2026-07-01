from __future__ import annotations

"""Resolve raw MRI DICOM provenance and extract metadata for MRI sidecars.

The converter uses the raw NIfTI filename to map back to the original RIA scan
selection CSVs and then reads one readable DICOM file from the source folder to
populate the MRI sidecar JSON with acquisition metadata.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import pydicom
except ModuleNotFoundError:  # pragma: no cover
    pydicom = None


# MRI DICOM provenance files used in scripts_other/create_nifti_dataset.py
RIA_PREOP_SCANS_CSV = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\mri\ria_pull\pre_op_scans_manual_select_final.csv"
)
RIA_POSTOP_SCANS_CSV = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\mri\ria_pull\post_op_scans_manual_select_final.csv"
)
RIA_DICOM_ROOT = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\mri\ria_pull\raw")


def _scan_value_to_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def _parse_nifti_origin_key(nifti_path: Path) -> Optional[tuple[str, str, str]]:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = nifti_path.stem

    parts = stem.rsplit("-", 2)
    if len(parts) != 3:
        return None

    patient_id, phase, scan_type = parts[0], parts[1].lower(), parts[2]
    if phase not in {"preop", "postop"}:
        return None

    return (patient_id, phase, scan_type)


def _scan_df_to_lookup(df: pd.DataFrame, phase: str) -> dict[tuple[str, str, str], Path]:
    lookup: dict[tuple[str, str, str], Path] = {}

    for _, row in df.iterrows():
        patient_id = _scan_value_to_str(row.get("patient_id"))
        scan_type = _scan_value_to_str(row.get("Type"))
        session = _scan_value_to_str(row.get("session"))
        scan_id = _scan_value_to_str(row.get("scan"))

        if not all([patient_id, scan_type, session, scan_id]):
            continue

        dicom_dir = RIA_DICOM_ROOT / scan_type / session / scan_id / "DICOM"
        lookup[(patient_id, phase, scan_type)] = dicom_dir

    return lookup


def _load_scan_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path, sep=";")
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def build_raw_mri_dicom_lookup() -> dict[tuple[str, str, str], Path]:
    lookup: dict[tuple[str, str, str], Path] = {}

    pre_df = _load_scan_csv(RIA_PREOP_SCANS_CSV)
    if not pre_df.empty:
        lookup.update(_scan_df_to_lookup(pre_df, phase="preop"))

    post_df = _load_scan_csv(RIA_POSTOP_SCANS_CSV)
    if not post_df.empty:
        lookup.update(_scan_df_to_lookup(post_df, phase="postop"))

    return lookup


def resolve_raw_mri_dicom_folder(nifti_path: Path) -> Optional[Path]:
    key = _parse_nifti_origin_key(nifti_path)
    if key is None:
        return None
    return build_raw_mri_dicom_lookup().get(key)


def _dicom_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, (list, tuple)):
        return [_dicom_scalar(v) for v in value]

    text = str(value)
    if "\\" in text:
        return [_dicom_scalar(v) for v in text.split("\\")]
    return text


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", ".")
            if not text:
                return None
            return float(text)
        return float(value)
    except Exception:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    text = "" if value is None else str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "f", "0", "no", "n", "off"}:
        return False
    return None


def _get_tag_value(ds: Any, group: int, element: int) -> Any:
    try:
        data_element = ds.get((group, element))
        if data_element is None:
            return None
        return data_element.value
    except Exception:
        return None


def _normalize_mr_acquisition_type(value: Any) -> Optional[str]:
    text = "" if value is None else str(value).strip().upper()
    if text in {"1D", "2D", "3D"}:
        return text
    return None


def _normalize_sequence_name(value: Any) -> Optional[str]:
    scalar = _dicom_scalar(value)
    if scalar in {None, ""}:
        return None

    if isinstance(scalar, list):
        scalar = "_".join(str(v) for v in scalar if str(v).strip())

    text = str(scalar).strip()
    if not text:
        return None
    return text


def _extract_dicom_dataset(dicom_dir: Path) -> Optional[Any]:
    if pydicom is None:
        return None

    if not dicom_dir.exists():
        return None

    candidates = sorted([p for p in dicom_dir.iterdir() if p.is_file()])
    for candidate in candidates:
        try:
            return pydicom.dcmread(str(candidate), stop_before_pixels=True, force=True)
        except Exception:
            continue

    return None


def read_dicom_metadata(dicom_dir: Path) -> dict[str, Any]:
    if pydicom is None:
        return {
            "DicomMetadataAvailable": False,
            "DicomMetadataReason": "pydicom is not installed",
            "SourceDicomDirectory": str(dicom_dir),
        }

    ds = _extract_dicom_dataset(dicom_dir)
    if ds is None:
        return {
            "DicomMetadataAvailable": False,
            "DicomMetadataReason": "No readable DICOM file found",
            "SourceDicomDirectory": str(dicom_dir),
        }

    payload: dict[str, Any] = {
        "DicomMetadataAvailable": True,
        "SourceDicomDirectory": str(dicom_dir),
    }

    direct_fields = {
        "Manufacturer": "Manufacturer",
        "ManufacturerModelName": "ManufacturersModelName",
        "InstitutionName": "InstitutionName",
        "StationName": "StationName",
        "DeviceSerialNumber": "DeviceSerialNumber",
        "SoftwareVersions": "SoftwareVersions",
        "SeriesDescription": "SeriesDescription",
        "ProtocolName": "ProtocolName",
        "SequenceName": "SequenceName",
        "ScanningSequence": "ScanningSequence",
        "SequenceVariant": "SequenceVariant",
        "ScanOptions": "ScanOptions",
        "MRAcquisitionType": "MRAcquisitionType",
        "BodyPartExamined": "BodyPart",
        "PatientPosition": "PatientPosition",
        "StudyDate": "StudyDate",
        "SeriesDate": "SeriesDate",
        "AcquisitionDate": "AcquisitionDate",
        "StudyTime": "StudyTime",
        "SeriesTime": "SeriesTime",
        "AcquisitionTime": "AcquisitionTime",
    }

    for dicom_key, bids_key in direct_fields.items():
        value = _dicom_scalar(getattr(ds, dicom_key, None))
        if value not in {None, ""}:
            payload[bids_key] = value

    # Use explicit, valid BIDS enums for MR acquisition type where available.
    mr_acq = _normalize_mr_acquisition_type(payload.get("MRAcquisitionType"))
    if mr_acq is not None:
        payload["MRAcquisitionType"] = mr_acq
    else:
        payload.pop("MRAcquisitionType", None)

    numeric_fields = {
        "MagneticFieldStrength": "MagneticFieldStrength",
        "FlipAngle": "FlipAngle",
        "PixelBandwidth": "PixelBandwidth",
        "SpacingBetweenSlices": "SpacingBetweenSlices",
        "SliceThickness": "SliceThickness",
        "EchoTrainLength": "EchoTrainLength",
    }

    for dicom_key, bids_key in numeric_fields.items():
        value = getattr(ds, dicom_key, None)
        if value is None:
            continue
        try:
            payload[bids_key] = float(value)
        except Exception:
            payload[bids_key] = _dicom_scalar(value)

    repetition_time_ms = getattr(ds, "RepetitionTime", None)
    if repetition_time_ms is not None:
        try:
            payload["RepetitionTime"] = float(repetition_time_ms) / 1000.0
        except Exception:
            payload["RepetitionTime"] = _dicom_scalar(repetition_time_ms)

    echo_time_ms = getattr(ds, "EchoTime", None)
    if echo_time_ms is not None:
        try:
            payload["EchoTime"] = float(echo_time_ms) / 1000.0
        except Exception:
            payload["EchoTime"] = _dicom_scalar(echo_time_ms)

    inversion_time_ms = getattr(ds, "InversionTime", None)
    if inversion_time_ms is not None:
        try:
            payload["InversionTime"] = float(inversion_time_ms) / 1000.0
        except Exception:
            payload["InversionTime"] = _dicom_scalar(inversion_time_ms)

    # SequenceName is recommended; recover from related scanner fields when absent.
    sequence_name = _normalize_sequence_name(payload.get("SequenceName"))
    if sequence_name is not None:
        payload["SequenceName"] = sequence_name
    else:
        payload.pop("SequenceName", None)

    if "SequenceName" not in payload:
        for key in ("PulseSequenceName", "ScanningSequence", "ProtocolName", "SeriesDescription"):
            fallback = _normalize_sequence_name(getattr(ds, key, None))
            if fallback is not None:
                payload["SequenceName"] = fallback
                break

    # NonlinearGradientCorrection is optional/recommended but strictly boolean when present.
    nonlinear_value = (
        getattr(ds, "NonlinearGradientCorrection", None)
        or getattr(ds, "GradientNonlinearityCorrection", None)
    )
    nonlinear_bool = _to_bool(nonlinear_value)
    if nonlinear_bool is not None:
        payload["NonlinearGradientCorrection"] = nonlinear_bool

    # DwellTime is optional/recommended. Prefer explicit DICOM fields and Siemens private tag.
    dwell_candidates = [
        _get_tag_value(ds, 0x0019, 0x1018),  # Siemens private: dwell in ns (common)
        getattr(ds, "DwellTime", None),
    ]
    dwell_seconds: Optional[float] = None
    for candidate in dwell_candidates:
        value = _to_float(candidate)
        if value is None:
            continue

        # Heuristic: values larger than microseconds likely represent nanoseconds.
        if value > 1e-4:
            value = value / 1_000_000_000.0

        if value > 0:
            dwell_seconds = value
            break

    if dwell_seconds is not None:
        payload["DwellTime"] = dwell_seconds

    return payload