# create_bids_dataset.py
# Author: Sjors
# Date: May 2026
# Description: Script to create a BIDS-compliant dataset structure
#              for our study.

from __future__ import annotations

import re
import unicodedata
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyedflib
import nibabel as nib
from tqdm.auto import tqdm


# =============================================================================
# USER CONFIG
# =============================================================================

DATASET_NAME = "RESPect - Resected FCD multimodal MRI-EEG dataset"
BIDS_VERSION = "1.11.1"

MANIFEST_CSV = Path(
    r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\data_availability\BIDS_conversion\source_manifest.csv"
)
BIDS_ROOT = Path(
    r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\BIDS_dataset"
)
# Folder that contains the selected_*.csv files exported from Castor EDC.
# Set to None to skip enriched participants files (only participant_id is written).
PARTICIPANTS_CSV_DIR: Optional[Path] = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection"
)

# Column name that contains the participant / subject identifier in ALL source CSVs.
PARTICIPANTS_ID_COLUMN = "Participant Id"

# Per-output-column mapping: { bids_column: (csv_filename, source_column_name) }
# Set source_column_name to None to skip that column for now.
# ---------------------------------------------------------------------------
# Source-file names (relative to PARTICIPANTS_CSV_DIR)
# ---------------------------------------------------------------------------
DEMOGR_FILE   = "selected_demographics.csv"
PATH_FILE     = "selected_pathology.csv"
SURG_FILE     = "selected_surgery.csv"
OUTCOME_FILE  = "selected_outcome.csv"
MRI_FILE      = "selected_mri.csv"

# ---------------------------------------------------------------------------
# Column names — update here if the Castor export changes
# ---------------------------------------------------------------------------
DEMOGR_DOB_COL          = "P0Demogr01"   # date of birth
DEMOGR_SEX_COL          = "P0Demogr02"   # sex: 0=F, 1=M

PATH_SURG_DATE_COL      = "P9Path01"     # date of pathology specimen ≈ surgery date
PATH_FCD_TYPE_COL       = "P9Path04"     # FCD subtype (see FCD_TYPE_MAP)

SURG_LOBE_COL           = "P4EpSG05"    # anatomical location (see LOBE_MAP)
SURG_SIDE_COL           = "P4EpSG07"    # laterality (see SIDE_MAP)
SURG_CONCLUSION_COL     = "P4_custom_all_conclusions"  # free-text

OUTCOME_ENGEL_COL       = "P10Out12"    # Engel class (see ENGEL_MAP)
OUTCOME_LAST_DATE_COL   = "P10Out03"    # date of last follow-up

MRI_TIMING_COL          = "P11MRI12"    # 1=presurgical, 2=after 1st surgery, ...
MRI_PATHOLOGY_COL       = "P11MRI10"    # abnormality group (non-null -> MRI positive)
MRI_SIDE_COL            = "P11MRI07"    # laterality of MRI abnormality (see SIDE_MAP)
MRI_LOBE_COL            = "P11MRI08"    # anatomical location of abnormality (see LOBE_MAP)

# ---------------------------------------------------------------------------
# Encoding / recoding maps
# ---------------------------------------------------------------------------
SEX_MAP: dict[int, str] = {0: "F", 1: "M"}

FCD_TYPE_MAP: dict[int, str] = {
    0: "FCD Ia",
    1: "FCD Ib",
    2: "FCD Ic",
    3: "FCD IIa",
    4: "FCD IIb",
    5: "FCD IIIa",
    6: "FCD IIIb",
    7: "FCD IIIc",
    8: "FCD IIId",
    9: "FCD I (not further specified)",
}

LOBE_MAP: dict[int, str] = {
    0: "temporal",
    1: "frontal",
    2: "parietal",
    3: "occipital",
    4: "insular",
    5: "fronto-temporal",
}

SIDE_MAP: dict[int, str] = {
    0: "left",
    1: "right",
    2: "bilateral",
    666: "n/a",
}

ENGEL_MAP: dict[int, str] = {
    0:  "Ia",  1:  "Ib",  2:  "Ic",  3:  "Id",
    4:  "IIa", 5:  "IIb", 6:  "IIc", 7:  "IId",
    8:  "IIIa", 9: "IIIb",
    10: "IVa", 11: "IVb", 16: "IVc",
    12: "I",   13: "II",  14: "III", 15: "IV",
}

# Dry run. This prints intended operations without copying/writing data.
DRY_RUN = False
# If False, existing files are left untouched.
OVERWRITE = False

# EEG defaults. The script will try to read EDF headers with pyedflib if installed.
DEFAULT_TASK_LABEL = "rest"
DEFAULT_POWER_LINE_FREQUENCY = 50
DEFAULT_EEG_REFERENCE = "n/a"
DEFAULT_SOFTWARE_FILTERS: str | dict[str, Any] = "n/a"
DEFAULT_HARDWARE_FILTERS: str | dict[str, Any] = "n/a"

# Rows with session == ses-unknown are mapped according to source category.
# You can change these once you know how to determine pre-/post-op for EEG derivatives.
DEFAULT_SESSION_BY_SOURCE_CATEGORY = {
    "raw_eeg_edf": "ses-preop",
    "preprocessed_eeg": "ses-preop",
    "eeg_spikes": "ses-preop",
    "preprocessed_mri": "ses-preop",
    "freesurfer_fastsurfer": "ses-preop",
}

# Derivative pipeline folder names.
DERIVATIVE_PIPELINES = {
    "ground_truth": "resection-masks",
    "preprocessed_mri": "mri-preproc",
    "freesurfer_fastsurfer": "freesurfer-fastsurfer",
    "preprocessed_eeg": "eeg-preproc",
    "eeg_spikes": "eeg-spikes",
}

# Non-BIDS auxiliary data go under sourcedata.
SOURCEDATA_CATEGORIES = {
    "intraoperative_cortex_picture",
}

# =============================================================================
# Regexes for inferring channel types from EDF labels.
# =============================================================================

# Fairly broad 10-20 / 10-10 / 10-5-ish scalp EEG label pattern.
# Examples: Fp1, Fpz, AF7, F3, Fz, FC1, FT9, T7, C3, CP5, Pz, PO8, O2
EEG_1005_RE = re.compile(
    r"^(?:"
    r"FPZ?|AF|F|FC|FT|T|C|CP|TP|P|PO|O"
    r")"
    r"(?:Z|[0-9]{1,2})$",
    re.IGNORECASE,
)

# Reference-like electrodes. 
REF_LIKE_RE = re.compile(
    r"^(?:A1|A2|M1|M2|TP9|TP10)$",
    re.IGNORECASE,
)

# Common ground/reference/system labels that should not be treated as EEG signal channels.
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
    if not src.exists():
        log_action(row, dst, action, "missing_source", "Source file does not exist")
        warn(f"Missing source file: {src}")
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
    shutil.copytree(src, dst)
    log_action(row, dst, action, "copied", "")


def get_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def sidecar_json_path(image_path: Path) -> Path:
    if get_extension(image_path) == ".nii.gz":
        return image_path.with_suffix("").with_suffix(".json")
    return image_path.with_suffix(".json")


def bids_subject(subject_id: str) -> str:
    """RESP1234 -> sub-RESP1234."""
    return f"sub-{subject_id}"


def clean_label(label: str) -> str:
    """BIDS labels should not contain underscores or special characters."""
    label = str(label)
    label = label.replace("ses-", "")
    label = label.replace("sub-", "")
    return "".join(ch for ch in label if ch.isalnum())


def norm_channel_label(label: str) -> str:
    """Normalize channel label for robust matching."""
    if label is None:
        return ""

    s = str(label).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Common cleanup
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)

    # Remove common EEG prefixes used by some systems, e.g. EEG Fp1, EEG-Fp1
    s = re.sub(r"^(EEG|POL|POLY|CHAN|CH)[\s_\-:]*", "", s, flags=re.IGNORECASE)

    # Remove common suffixes that indicate derivations/references,
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
# PARTICIPANTS FILES
# =============================================================================

# ---------------------------------------------------------------------------
# Helpers used only within the participants extraction
# ---------------------------------------------------------------------------

def _pt_load(csv_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    """Load *filename* from *csv_dir*. Returns None (with a warning) if missing."""
    p = csv_dir / filename
    if not p.exists():
        warn(f"Participants CSV not found: {p}")
        return None
    df = pd.read_csv(p)
    if PARTICIPANTS_ID_COLUMN not in df.columns:
        warn(f"Column '{PARTICIPANTS_ID_COLUMN}' not found in {p}; skipping.")
        return None
    df[PARTICIPANTS_ID_COLUMN] = df[PARTICIPANTS_ID_COLUMN].astype(str).str.strip()
    return df


def _pt_get(df: Optional[pd.DataFrame], raw_id: str, col: str) -> Any:
    """Return the value of *col* for *raw_id* in *df*, or 'n/a' if missing/NaN."""
    if df is None or col not in df.columns:
        return "n/a"
    match = df[df[PARTICIPANTS_ID_COLUMN] == raw_id]
    if match.empty:
        return "n/a"
    val = match.iloc[0][col]
    return "n/a" if pd.isna(val) else val


def _pt_recode(value: Any, mapping: dict) -> str:
    """Cast *value* to int and look it up in *mapping*; fall back gracefully."""
    if value == "n/a" or (not isinstance(value, str) and pd.isna(value)):
        return "n/a"
    try:
        return mapping.get(int(float(str(value))), str(value))
    except (ValueError, TypeError):
        return str(value)


def _pt_age(dob_raw: Any, surgery_date_raw: Any) -> Any:
    """Return integer age at surgery (years), or 'n/a' on any parse failure."""
    if dob_raw == "n/a" or surgery_date_raw == "n/a":
        return "n/a"
    try:
        dob  = pd.to_datetime(dob_raw, dayfirst=True, errors="raise")
        surg = pd.to_datetime(surgery_date_raw, dayfirst=True, errors="raise")
        return int((surg - dob).days / 365.25)
    except Exception:
        return "n/a"


def _pt_mri_status(mri_df: Optional[pd.DataFrame], raw_id: str) -> str:
    """Return 'positive', 'negative', or 'n/a' based on presurgical MRI rows."""
    if mri_df is None:
        return "n/a"
    presurg = mri_df[
        (mri_df[PARTICIPANTS_ID_COLUMN] == raw_id)
        & (mri_df[MRI_TIMING_COL] == 1)
    ]
    if presurg.empty:
        return "n/a"
    if MRI_PATHOLOGY_COL in presurg.columns and presurg[MRI_PATHOLOGY_COL].notna().any():
        return "positive"
    return "negative"


def extract_participants_from_csv(csv_dir: Path, subject_ids: list[str]) -> pd.DataFrame:
    """
    Load the selected_*.csv files from *csv_dir* and assemble one row per subject.

    File names and source column names are the module-level constants defined above
    (DEMOGR_FILE, PATH_FILE, etc.) — update only those if the Castor export changes.
    Missing files or columns silently produce 'n/a' values.
    """
    raw_ids = [sid.replace("sub-", "") for sid in subject_ids]

    # Load each source file once
    demogr  = _pt_load(csv_dir, DEMOGR_FILE)
    pathol  = _pt_load(csv_dir, PATH_FILE)
    surg    = _pt_load(csv_dir, SURG_FILE)
    outcome = _pt_load(csv_dir, OUTCOME_FILE)
    mri_df  = _pt_load(csv_dir, MRI_FILE)   # multiple rows per subject — keep all rows

    rows = []
    for raw_id, bids_id in zip(raw_ids, subject_ids):
        # sex
        sex = _pt_recode(_pt_get(demogr, raw_id, DEMOGR_SEX_COL), SEX_MAP)

        # age at surgery (DOB from demographics, surgery date from pathology report)
        dob       = _pt_get(demogr, raw_id, DEMOGR_DOB_COL)
        surg_date = _pt_get(pathol,  raw_id, PATH_SURG_DATE_COL)
        age       = _pt_age(dob, surg_date)

        # FCD subtype
        fcd_type = _pt_recode(_pt_get(pathol, raw_id, PATH_FCD_TYPE_COL), FCD_TYPE_MAP)

        # resection lobe and side
        resection_lobe = _pt_recode(_pt_get(surg, raw_id, SURG_LOBE_COL), LOBE_MAP)
        resection_side = _pt_recode(_pt_get(surg, raw_id, SURG_SIDE_COL), SIDE_MAP)

        # free-text surgery conclusion
        surgery_conclusion = _pt_get(surg, raw_id, SURG_CONCLUSION_COL)

        # Engel outcome and date of last follow-up
        outcome_engel = _pt_recode(_pt_get(outcome, raw_id, OUTCOME_ENGEL_COL), ENGEL_MAP)
        outcome_date  = _pt_get(outcome, raw_id, OUTCOME_LAST_DATE_COL)

        # presurgical MRI status (positive / negative / n/a)
        mri_status = _pt_mri_status(mri_df, raw_id)

        # MRI lesion lobe and side (first presurgical row only)
        mri_lobe = mri_side = "n/a"
        if mri_df is not None and mri_status == "positive":
            presurg = mri_df[
                (mri_df[PARTICIPANTS_ID_COLUMN] == raw_id)
                & (mri_df[MRI_TIMING_COL] == 1)
            ]
            if not presurg.empty:
                mri_lobe = _pt_recode(presurg.iloc[0].get(MRI_LOBE_COL, "n/a"), LOBE_MAP)
                mri_side = _pt_recode(presurg.iloc[0].get(MRI_SIDE_COL, "n/a"), SIDE_MAP)

        rows.append({
            "participant_id":     bids_id,
            "age":                age,
            "sex":                sex,
            "fcd_type":           fcd_type,
            "resection_lobe":     resection_lobe,
            "resection_side":     resection_side,
            "mri_presurgical":    mri_status,
            "mri_lesion_lobe":    mri_lobe,
            "mri_lesion_side":    mri_side,
            "outcome_engel":      outcome_engel,
            "outcome_date":       outcome_date,
            "surgery_conclusion": surgery_conclusion,
        })

    return pd.DataFrame(rows)


# Column-level descriptions written to participants.json.
PARTICIPANTS_JSON_FIELDS: dict[str, dict[str, Any]] = {
    "participant_id": {
        "Description": "Unique participant identifier",
    },
    "age": {
        "Description": "Age at surgery, computed from date of birth (demographics) and date of pathology specimen",
        "Units": "years",
    },
    "sex": {
        "Description": "Biological sex of the participant",
        "Levels": {"M": "Male", "F": "Female"},
    },
    "fcd_type": {
        "Description": "FCD subtype according to the ILAE 2022 classification (source: pathology report)",
        "Levels": {v: v for v in FCD_TYPE_MAP.values()},
    },
    "resection_lobe": {
        "Description": "Lobe of resection as recorded in the surgical report",
        "Levels": {v: v.capitalize() + " lobe" for v in LOBE_MAP.values()},
    },
    "resection_side": {
        "Description": "Hemisphere of resection",
        "Levels": {"left": "Left hemisphere", "right": "Right hemisphere", "bilateral": "Bilateral"},
    },
    "mri_presurgical": {
        "Description": "Whether a structural abnormality was identified on presurgical MRI",
        "Levels": {"positive": "Abnormality present", "negative": "No abnormality detected"},
    },
    "mri_lesion_lobe": {
        "Description": "Lobe of the MRI lesion on the first presurgical scan",
        "Levels": {v: v.capitalize() + " lobe" for v in LOBE_MAP.values()},
    },
    "mri_lesion_side": {
        "Description": "Laterality of the MRI lesion on the first presurgical scan",
        "Levels": {"left": "Left hemisphere", "right": "Right hemisphere", "bilateral": "Bilateral"},
    },
    "outcome_engel": {
        "Description": "Seizure outcome at last available follow-up (Engel classification)",
        "Levels": {
            "Ia": "Completely seizure-free since surgery",
            "Ib": "Only auras since surgery",
            "Ic": "Some seizures after surgery but seizure-free for >=2 years",
            "Id": "Seizure-free since surgery but on AEDs",
            "IIa": "Rare disabling seizures",
            "IIb": "More than rare disabling seizures but rare overall",
            "IIc": "Worthwhile improvement",
            "IId": "Prolonged seizure-free intervals but not <2 years",
            "IIIa": "Worthwhile improvement",
            "IIIb": "No worthwhile improvement",
            "IVa": "Significant worsening",
            "IVb": "No change",
            "IVc": "Significant worsening",
        },
    },
    "outcome_date": {
        "Description": "Date of last follow-up assessment used for outcome scoring",
    },
    "surgery_conclusion": {
        "Description": "Free-text conclusion from the surgical report",
    },
}


def write_participants_files(manifest: pd.DataFrame) -> None:
    """Write participants.tsv and participants.json to the BIDS root."""
    subject_ids = sorted(
        bids_subject(s) for s in manifest["subject_id"].astype(str).unique()
    )

    use_enriched = (
        PARTICIPANTS_CSV_DIR is not None
        and PARTICIPANTS_CSV_DIR.exists()
    )

    if PARTICIPANTS_CSV_DIR is not None and not PARTICIPANTS_CSV_DIR.exists():
        warn(
            f"PARTICIPANTS_CSV_DIR does not exist: {PARTICIPANTS_CSV_DIR}; "
            "falling back to minimal participants files."
        )

    if use_enriched:
        participants_df = extract_participants_from_csv(PARTICIPANTS_CSV_DIR, subject_ids)
        # Ensure participant_id is the first column.
        cols = ["participant_id"] + [c for c in participants_df.columns if c != "participant_id"]
        participants_df = participants_df[cols]
        # Only keep JSON field descriptions for columns that are actually present.
        participants_json: dict[str, Any] = {
            col: PARTICIPANTS_JSON_FIELDS[col]
            for col in participants_df.columns
            if col in PARTICIPANTS_JSON_FIELDS
        }
    else:
        participants_df = pd.DataFrame({"participant_id": subject_ids})
        participants_json = {
            k: v for k, v in PARTICIPANTS_JSON_FIELDS.items() if k == "participant_id"
        }

    write_tsv(BIDS_ROOT / "participants.tsv", participants_df)
    write_json(BIDS_ROOT / "participants.json", participants_json)


# =============================================================================
# TOP-LEVEL DATASET FILES
# =============================================================================

def write_top_level_files(manifest: pd.DataFrame) -> None:
    raw_description = {
        "Name": DATASET_NAME,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "Authors": [
            "Sjors Verschuren",
            "Robert Helling",
            "Galia Anguelova",
            "Nicole van Klink",
            "Fernando Gomez-Acebo Ruiz",
            "Maeike Zijlmans"
        ],
        "Acknowledgements": (
            "We thank all patients for their participation and the clinical and technical "
            "staff involved in data acquisition and annotation."
        ),
        "HowToAcknowledge": "n/a",
        "Funding": ["n/a"],
        "EthicsApprovals": ["n/a"],
        "ReferencesAndLinks": [],
    }
    write_json(BIDS_ROOT / "dataset_description.json", raw_description)

    readme = (
        f"# {DATASET_NAME}\n\n"
        "## Overview\n\n"
        "This dataset contains multimodal neuroimaging and electrophysiology data from patients "
        "with drug-resistant focal epilepsy due to Focal Cortical Dysplasia (FCD) who underwent "
        "resection surgery. It is structured according to the Brain Imaging Data Structure (BIDS) "
        f"specification v{BIDS_VERSION}.\n\n"
        "## Data modalities\n\n"
        "**Raw data** (`raw/`)\n\n"
        "- `anat/` — Pre-operative T1-weighted and FLAIR structural MRI; post-operative T1-weighted MRI\n"
        "- `eeg/` — Pre-operative resting-state scalp EEG in EDF format (originally recorded in TRC/SIG format, "
        "converted via Persyst 15; bandpass 1–70 Hz, 256 Hz)\n\n"
        "**Derivatives**\n\n"
        "- `resection-masks/` — Volumetric ground-truth resection masks derived from post-operative MRI\n"
        "- `mri-preproc/` — Preprocessed MRI volumes: N4 bias-field correction, skull stripping (HD-BET), "
        "affine registration to MNI152 space (ANTs/FSL), resampling to 1 mm isotropic, "
        "intensity normalisation (robust z-score); final shape 160 × 192 × 160 voxels\n"
        "- `freesurfer-fastsurfer/` — Cortical surface reconstructions produced by FreeSurfer / FastSurfer\n"
        "- `eeg-preproc/` — Preprocessed and concatenated EEG recordings in EDF format\n"
        "- `eeg-spikes/` — Interictal epileptiform discharge (spike) segments detected with Persyst 15 "
        "(Perception Score > 0.9), stored as NumPy arrays [n_segments × n_channels × n_samples] "
        "(1 s @ 256 Hz, per-subject z-score)\n\n"
        "**Source data** (`sourcedata/`)\n\n"
        "- Intraoperative cortex photographs (aECoG grid placement)\n\n"
        "## Sessions\n\n"
        "| Label | Description |\n"
        "|-------|-------------|\n"
        "| `ses-preop` | Pre-operative recordings |\n"
        "| `ses-postop` | Post-operative recordings |\n"
        "| `ses-intraop` | Intraoperative recordings / photographs |\n\n"
        "## Notes\n\n"
        "- EEG recording start times and event annotations were not available from the source files "
        "and are marked `n/a` throughout.\n"
        "- MRI-to-NIfTI conversion was performed with `dicom2nifti` v2.6.2.\n"
        "- Subject identifiers follow the `sub-RESP****` scheme.\n"
    )
    write_text(BIDS_ROOT / "README", readme)

    write_participants_files(manifest)

    # Optional, but useful because this dataset explicitly uses pre-/post-op/intra-op labels.
    sessions_rows = []
    for subject_id, sub_df in manifest.groupby("subject_id"):
        for session in sorted({normalize_session(row) for _, row in sub_df.iterrows()}):
            sessions_rows.append(
                {
                    "participant_id": bids_subject(str(subject_id)),
                    "session_id": bids_session_entity(session),
                    "session_label": session,
                }
            )
    if sessions_rows:
        write_tsv(BIDS_ROOT / "sessions.tsv", pd.DataFrame(sessions_rows))


def write_derivative_dataset_descriptions() -> None:
    for pipeline_name in sorted(set(DERIVATIVE_PIPELINES.values())):
        payload = {
            "Name": pipeline_name,
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "custom-source-to-bids-conversion",
                    "Description": "Files copied and organized from source manifest into a BIDS-compatible derivative structure.",
                }
            ],
            "SourceDatasets": [
                {
                    "URL": "n/a",
                    "Description": "Local source dataset organized by RESP**** identifiers",
                }
            ],
        }
        write_json(BIDS_ROOT / "derivatives" / pipeline_name / "dataset_description.json", payload)


# =============================================================================
# EDF HEADER HELPERS
# =============================================================================

def read_edf_metadata(edf_path: Path) -> dict[str, Any]:
    """
    Read basic EDF metadata if pyedflib is installed.

    Returns an empty-ish dictionary when unavailable, so conversion can continue.
    """
    if pyedflib is None:
        return {
            "available": False,
            "reason": "pyedflib is not installed",
            "labels": [],
            "sample_frequencies": [],
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
        reader.close()

        return {
            "available": True,
            "labels": labels,
            "sample_frequencies": sample_frequencies,
            "transducer": transducer,
            "physical_dimension": physical_dimension,
            "prefilter": prefilter,
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
            "n_channels": "n/a",
            "duration": "n/a",
        }


def infer_eeg_channel_type(label: str) -> str:
    raw = "" if label is None else str(label)
    lower = raw.lower()
    compact = norm_channel_label(raw)

    # Triggers / status / event channels
    if re.search(
        r"(trigger|trig|status|marker|mark|event|mrk|mkr|stim|sync|ttl|annot|annotation|edf annotations?)",
        lower,
    ):
        return "TRIG"

    # ECG
    if re.search(r"(ecg|ekg|hart|cardio)", lower):
        return "ECG"

    # EOG
    if re.search(r"(eog|oog|eye|ocul)", lower):
        return "EOG"

    # EMG
    if re.search(r"(emg|myo|muscle|spier|chin|kin)", lower):
        return "EMG"

    # Resp
    if re.search(
        r"(resp|respiration|breath|breathing|airflow|flow|snore|"
        r"adem|ademhaling|thor|thorax|chest|borst|abd|abdomen|abdominaal|buik)",
        lower,
    ):
        return "MISC"

    # Pulse/sat
    if re.search(
        r"(spo2|sao2|sat|saturatie|oxygen|oximeter|oximetry|"
        r"pulse|puls|pols|pleth|ppg|beat|hartslag)",
        lower,
    ):
        return "MISC"

    # Temp
    if re.search(r"(temp|temperature|temperatuur)", lower):
        return "MISC"

    # Photic
    if re.search(r"(photic|flash|flits|lamp)", lower):
        return "MISC"

    # Generic auxiliary / miscellaneous
    if re.search(r"(misc|aux|dc|analog|analogue|sensor|unknown|onbekend)", lower):
        return "MISC"

    # Explicit non-signal/reference/system labels
    if NON_SIGNAL_RE.match(compact):
        return "MISC"

    # Reference-like electrodes. For raw BIDS, I would usually type these as EEG
    # if they are actual recorded voltage channels. Exclude them later from CAR.
    if REF_LIKE_RE.match(compact):
        return "EEG"

    # Standard scalp EEG labels
    if EEG_1005_RE.match(compact):
        return "EEG"

    return "MISC"


def make_channels_tsv(edf_metadata: dict[str, Any]) -> pd.DataFrame:
    labels = edf_metadata.get("labels", []) or []
    sample_frequencies = edf_metadata.get("sample_frequencies", []) or []
    physical_dimension = edf_metadata.get("physical_dimension", []) or []
    prefilter = edf_metadata.get("prefilter", []) or []

    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "name": label,
                "type": infer_eeg_channel_type(label),
                "units": physical_dimension[idx] if idx < len(physical_dimension) and physical_dimension[idx] else "n/a",
                "sampling_frequency": sample_frequencies[idx] if idx < len(sample_frequencies) else "n/a",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "notch": "n/a",
                "status": "good",
                "status_description": "n/a",
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


def make_eeg_json(edf_metadata: dict[str, Any], source_path: Path, derivative: bool = False) -> dict[str, Any]:
    sample_frequencies = edf_metadata.get("sample_frequencies", []) or []
    unique_fs = sorted(set(float(x) for x in sample_frequencies))

    if len(unique_fs) == 1:
        sampling_frequency: Any = unique_fs[0]
    elif len(unique_fs) > 1:
        sampling_frequency = max(unique_fs)
        warn(
            f"Multiple sample frequencies in {source_path}. "
            f"Using max for EEG JSON and writing per-channel values in channels.tsv: {unique_fs}"
        )
    else:
        sampling_frequency = "n/a"
        warn(
            f"SamplingFrequency could not be determined for {source_path}. "
            "Install pyedflib or manually add this value for strict BIDS validation."
        )

    payload = {
        "TaskName": DEFAULT_TASK_LABEL,
        "Manufacturer": "n/a",
        "ManufacturersModelName": "n/a",
        "SamplingFrequency": sampling_frequency,
        "PowerLineFrequency": DEFAULT_POWER_LINE_FREQUENCY,
        "SoftwareFilters": DEFAULT_SOFTWARE_FILTERS,
        "HardwareFilters": DEFAULT_HARDWARE_FILTERS,
        "EEGReference": DEFAULT_EEG_REFERENCE,
        "RecordingType": "continuous" if not derivative else "discontinuous",
        "RecordingDuration": edf_metadata.get("duration", "n/a"),
        "RecordingStartTime": "n/a",
        "EEGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EEG"),
        "EOGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EOG"),
        "ECGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "ECG"),
        "EMGChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "EMG"),
        "MiscChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "MISC"),
        "TriggerChannelCount": sum(1 for label in edf_metadata.get("labels", []) if infer_eeg_channel_type(label) == "TRIG"),
        "SourceFile": str(source_path),
        "ConversionNotes": "Start time and events were intentionally not extracted from source TRC/SIG-STS files.",
    }

    if derivative:
        payload["Description"] = "Preprocessed or concatenated EEG derivative copied from source pipeline."

    return payload


# =============================================================================
# NIFTI HEADER HELPERS
# =============================================================================

def read_nifti_metadata(nifti_path: Path) -> dict[str, Any]:
    """Extract useful metadata from a NIfTI header for MRI sidecars."""
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

        payload: dict[str, Any] = {
            "NiftiHeaderAvailable": True,
            "NiftiSpatialShape": shape[:3],
            "NiftiVoxelSize": spatial_zooms,
            "NiftiSpatialUnits": xyz_units or "unknown",
            "NiftiTimeUnits": time_units or "unknown",
            "NumberOfVolumes": int(shape[3]) if len(shape) > 3 else 1,
        }

        # BIDS expects RepetitionTime in seconds.
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
# RAW BIDS CONVERTERS
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

    sidecar = {
        "Modality": "MR",
        "SourceFile": str(src),
        "ConversionSoftware": "custom-source-to-bids-conversion",
        "ConversionSoftwareVersion": "dicom2nifti-2.6.2",
        "ConversionNotes": "DICOM to NIfTI conversion performed using `dicom2nifti-2.6.2` via `scripts_other/create_nifti_dataset.py`. Acquisition metadata was read from the NIfTI header where available.",
    }
    if get_extension(dst) in {".nii", ".nii.gz"}:
        nifti_for_metadata = dst if dst.exists() and not DRY_RUN else src
        sidecar.update(read_nifti_metadata(nifti_for_metadata))

    write_json(sidecar_json_path(dst), sidecar)


def convert_raw_eeg(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    task_entity = f"task-{clean_label(DEFAULT_TASK_LABEL)}"

    stem = make_bids_stem(row, suffix="eeg", extra_entities=[task_entity])
    dst = BIDS_ROOT / subject / session / "eeg" / f"{stem}.edf"
    copy_file(src, dst, row)

    edf_metadata = read_edf_metadata(src)
    eeg_json = make_eeg_json(edf_metadata, source_path=src, derivative=False)
    channels_tsv = make_channels_tsv(edf_metadata)

    write_json(dst.with_suffix(".json"), eeg_json)
    write_tsv(dst.with_name(dst.name.replace("_eeg.edf", "_channels.tsv")), channels_tsv)


def update_scans_tsv(raw_outputs: list[Path]) -> None:
    """Create/update per-session scans.tsv files for raw BIDS files."""
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
            rows.append(
                {
                    "filename": relative_to_bids(path),
                    "acq_time": "n/a",
                }
            )
        scans_path = session_dir / f"{subject}_{session}_scans.tsv"
        write_tsv(scans_path, pd.DataFrame(rows))


# =============================================================================
# SOURCEDATA / DERIVATIVE CONVERTERS
# =============================================================================

def copy_cortex_picture(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = "ses-intraop"
    dst = BIDS_ROOT / "sourcedata" / subject / session / "photos" / src.name
    copy_file(src, dst, row, action="copy_sourcedata")


def copy_ground_truth(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = "ses-postop"
    pipeline = DERIVATIVE_PIPELINES["ground_truth"]

    if str(row.get("modality")) == "resection_mask":
        stem = f"{subject}_{session}_desc-resection_mask"
        dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "anat" / f"{stem}{get_extension(src)}"
        copy_file(src, dst, row, action="copy_derivative")
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
        dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "auxiliary" / src.name
        copy_file(src, dst, row, action="copy_derivative_auxiliary")


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

    if get_extension(dst) in {".nii", ".nii.gz"}:
        sidecar = {
            "Description": "Preprocessed MRI derivative copied from source pipeline",
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

    edf_metadata = read_edf_metadata(src)
    write_json(dst.with_suffix(".json"), make_eeg_json(edf_metadata, source_path=src, derivative=True))
    write_tsv(dst.with_name(dst.name.replace("_eeg.edf", "_channels.tsv")), make_channels_tsv(edf_metadata))


def copy_eeg_spikes(row: pd.Series) -> None:
    src = Path(row["source_path"])
    subject = bids_subject(str(row["subject_id"]))
    session = bids_session_entity(normalize_session(row))
    pipeline = DERIVATIVE_PIPELINES["eeg_spikes"]

    # Preserve useful frequency-band information from the source filename.
    # Example: RESP1234_spikes_1-70Hz.npy -> desc-spikes1to70Hz
    band = src.stem.replace(str(row["subject_id"]), "").strip("_")
    band_label = clean_label(band.replace("-", "to")) or "spikes"

    stem = f"{subject}_{session}_desc-{band_label}_eeg"
    dst = BIDS_ROOT / "derivatives" / pipeline / subject / session / "eeg" / f"{stem}.npy"
    copy_file(src, dst, row, action="copy_derivative")

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
    elif category == "ground_truth":
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
# MAIN
# =============================================================================

def main() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)

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

    log(f"Loaded manifest: {MANIFEST_CSV}")
    log(f"Rows: {len(manifest)}")
    log(f"BIDS root: {BIDS_ROOT}")
    log(f"DRY_RUN: {DRY_RUN}")
    log(f"OVERWRITE: {OVERWRITE}")

    ensure_dir(BIDS_ROOT)
    write_top_level_files(manifest)
    write_derivative_dataset_descriptions()

    row_iterator = manifest.iterrows()
    if tqdm is not None:
        row_iterator = tqdm(row_iterator, total=len(manifest), desc="Converting manifest rows", unit="row")

    for _, row in row_iterator:
        dispatch_row(row)

    # Build scans.tsv from copied raw MRI/EEG destination paths in copy log.
    raw_outputs = [
        Path(record.destination_path)
        for record in COPY_LOG
        if record.source_category in {"raw_mri_preop", "raw_mri_postop", "raw_eeg_edf"}
        and record.status in {"copied", "dry_run", "skipped_exists"}
    ]
    update_scans_tsv(raw_outputs)

    write_logs()

    log("Done.")
    log(f"Copy records: {len(COPY_LOG)}")
    log(f"Warnings: {len(WARNING_LOG)}")
    if DRY_RUN:
        log("This was a dry run. Set DRY_RUN = False to actually write/copy files.")


if __name__ == "__main__":
    main()
