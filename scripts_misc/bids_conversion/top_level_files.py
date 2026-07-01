from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


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
PARTICIPANTS_CSV_DIR: Optional[Path] = Path(
    r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection"
)
PARTICIPANTS_ID_COLUMN = "Participant Id"

SUMMARY_FILE = "selected_summary.csv"
DEMOGR_FILE = "selected_demographics.csv"
PATH_FILE = "selected_pathology.csv"
SURG_FILE = "selected_surgery.csv"
OUTCOME_FILE = "selected_outcome.csv"
MRI_FILE = "selected_mri.csv"

DEMOGR_DOB_COL = "P0Demogr01"
DEMOGR_SEX_COL = "P0Demogr02"

PATH_SURG_DATE_COL = "P9Path01"
PATH_FCD_TYPE_COL = "P9Path04"

SURG_LOBE_COL = "P4EpSG05"
SURG_SIDE_COL = "P4EpSG07"
SURG_CONCLUSION_COL = "P4_custom_all_conclusions"

OUTCOME_SEIZURE_FREE_COL = "P10Out02"
OUTCOME_ENGEL_COL = "P10Out12"
OUTCOME_ILAE_COL = "P10Out13"
OUTCOME_LAST_DATE_COL = "P10Out01"

MRI_TIMING_COL = "P11MRI12"
MRI_PATHOLOGY_COL = "P11MRI10"

MRI_NEGATIVE_KEYWORDS = (
    "no abnormalities",
    "no abnormality",
    "normal",
)

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
    9: "FCD I (not further classifiable)",
    10: "FCD II (not further classifiable)",
    11: "FCD III (not further classifiable)",
    12: "FCD (not further classifiable)",

}

LOBE_MAP: dict[int, str] = {
    0: "temporal",
    1: "frontal",
    2: "parietal",
    3: "occipital",
    4: "insular",
    5: "fronto-temporal",
    6: "fronto-parietal",
    7: "temporo-parietal",
    8: "temporo-occipital",
    9: "parieto-occipital",
    10: "fronto-parieto-occipital",
    17: "temporo-parieto-occipital",
    18: "hemispheric",
    15: "multifocal",
    999: "n/a"
}

SIDE_MAP: dict[int, str] = {
    0: "left",
    1: "right",
    2: "bilateral",
    666: "n/a",
}

SEIZURE_FREE_MAP: dict[int, str] = {
    0: False,
    1: True,
    888: None,
    666: None
}

ENGEL_MAP: dict[int, str] = {
    0: "Ia",
    1: "Ib",
    2: "Ic",
    3: "Id",
    4: "IIa",
    5: "IIb",
    6: "IIc",
    7: "IId",
    8: "IIIa",
    9: "IIIb",
    10: "IVa",
    11: "IVb",
    16: "IVc",
    12: "I",
    13: "II",
    14: "III",
    15: "IV",
    888: "n/a",
    666: "n/a",
    -1: "I"
}

ILAE_MAP: dict[int, str] = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    888: "n/a",
    666: "n/a",
    -1: "1/2"
}

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

OVERWRITE = True


# =============================================================================
# HELPERS
# =============================================================================

def warn(message: str) -> None:
    print(f"WARNING: {message}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE:
        return
    df.to_csv(path, sep="\t", index=False, na_rep="n/a")


def bids_subject(subject_id: str) -> str:
    return f"sub-{subject_id}"


def clean_label(label: str) -> str:
    label = str(label)
    label = label.replace("ses-", "")
    label = label.replace("sub-", "")
    return "".join(ch for ch in label if ch.isalnum())


def normalize_session(row: pd.Series) -> str:
    session = str(row.get("session", "ses-unknown"))
    if session and session != "nan" and session != "ses-unknown":
        return session

    source_category = str(row.get("source_category", ""))
    return DEFAULT_SESSION_BY_SOURCE_CATEGORY.get(source_category, "ses-unknown")


def bids_session_entity(session: str) -> str:
    return f"ses-{clean_label(session)}"


# =============================================================================
# PARTICIPANTS FILES
# =============================================================================

def _pt_load(csv_dir: Path, filename: str) -> Optional[pd.DataFrame]:
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
    if df is None or col not in df.columns:
        return "n/a"
    match = df[df[PARTICIPANTS_ID_COLUMN] == raw_id]
    if match.empty:
        return "n/a"
    val = match.iloc[0][col]
    return "n/a" if pd.isna(val) else val


def _pt_recode(value: Any, mapping: dict) -> str:
    if value == "n/a" or (not isinstance(value, str) and pd.isna(value)):
        return "n/a"
    try:
        return mapping.get(int(float(str(value))), str(value))
    except (ValueError, TypeError):
        return str(value)


def _pt_age(dob_raw: Any, surgery_date_raw: Any) -> Any:
    if dob_raw == "n/a" or surgery_date_raw == "n/a":
        return "n/a"
    try:
        dob = pd.to_datetime(dob_raw, dayfirst=True, errors="raise")
        surg = pd.to_datetime(surgery_date_raw, dayfirst=True, errors="raise")
        return int((surg - dob).days / 365.25)
    except Exception:
        return "n/a"


def _pt_is_binary_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "x"}:
        return True

    try:
        return float(text) == 1.0
    except (ValueError, TypeError):
        return False


def _pt_split_pathology_status(row: pd.Series) -> Optional[str]:
    pathology_cols = [
        c for c in row.index if str(c).startswith(MRI_PATHOLOGY_COL) and str(c) != MRI_PATHOLOGY_COL
    ]
    if not pathology_cols:
        return None

    active_cols = [c for c in pathology_cols if _pt_is_binary_true(row.get(c))]
    if not active_cols:
        return None

    has_negative = any(
        kw in str(col).lower()
        for col in active_cols
        for kw in MRI_NEGATIVE_KEYWORDS
    )
    has_positive = any(
        all(kw not in str(col).lower() for kw in MRI_NEGATIVE_KEYWORDS)
        for col in active_cols
    )

    if has_positive:
        return "positive"
    if has_negative:
        return "negative"
    return None


def _pt_mri_status(mri_df: Optional[pd.DataFrame], raw_id: str) -> str:
    if mri_df is None:
        return "n/a"
    presurg = mri_df[
        (mri_df[PARTICIPANTS_ID_COLUMN] == raw_id)
        & (mri_df[MRI_TIMING_COL] == 1)
    ]
    if presurg.empty:
        return "n/a"

    split_statuses = [
        s for s in (_pt_split_pathology_status(row) for _, row in presurg.iterrows()) if s is not None
    ]
    if "positive" in split_statuses:
        return "positive"
    if "negative" in split_statuses:
        return "negative"

    if MRI_PATHOLOGY_COL in presurg.columns and presurg[MRI_PATHOLOGY_COL].notna().any():
        return "positive"
    return "negative"


def extract_participants_from_csv(csv_dir: Path, subject_ids: list[str]) -> pd.DataFrame:
    raw_ids = [sid.replace("sub-", "") for sid in subject_ids]

    summary = _pt_load(csv_dir, SUMMARY_FILE)
    demogr = _pt_load(csv_dir, DEMOGR_FILE)
    pathol = _pt_load(csv_dir, PATH_FILE)
    surg = _pt_load(csv_dir, SURG_FILE)
    outcome = _pt_load(csv_dir, OUTCOME_FILE)
    mri_df = _pt_load(csv_dir, MRI_FILE)

    rows = []
    for raw_id, bids_id in zip(raw_ids, subject_ids):
        sex = _pt_recode(_pt_get(demogr, raw_id, DEMOGR_SEX_COL), SEX_MAP)

        dob = _pt_get(demogr, raw_id, DEMOGR_DOB_COL)
        surg_date = _pt_get(pathol, raw_id, PATH_SURG_DATE_COL)
        age = _pt_age(dob, surg_date)

        fcd_type = _pt_recode(_pt_get(pathol, raw_id, PATH_FCD_TYPE_COL), FCD_TYPE_MAP)

        resection_lobe = _pt_recode(_pt_get(surg, raw_id, SURG_LOBE_COL), LOBE_MAP)
        resection_side = _pt_recode(_pt_get(surg, raw_id, SURG_SIDE_COL), SIDE_MAP)
        surgery_conclusion = _pt_get(summary, raw_id, SURG_CONCLUSION_COL).replace("\n", " | ").strip()

        outcome_engel = _pt_recode(_pt_get(outcome, raw_id, OUTCOME_ENGEL_COL), ENGEL_MAP)
        outcome_ilae = _pt_recode(_pt_get(outcome, raw_id, OUTCOME_ILAE_COL), ILAE_MAP)
        outcome_seizure_free = _pt_recode(_pt_get(outcome, raw_id, OUTCOME_SEIZURE_FREE_COL), SEIZURE_FREE_MAP)
        outcome_date = _pt_get(outcome, raw_id, OUTCOME_LAST_DATE_COL)

        if outcome_seizure_free and outcome_engel == "n/a":
            outcome_engel = "I"
        if outcome_seizure_free and outcome_ilae == "n/a":
            outcome_ilae = "1/2"

        mri_status = _pt_mri_status(mri_df, raw_id)

        rows.append(
            {
                "participant_id": bids_id,
                "age": age,
                "sex": sex,
                "fcd_type": fcd_type,
                "resection_lobe": resection_lobe,
                "resection_side": resection_side,
                "mri_presurgical": mri_status,
                "outcome_engel": outcome_engel,
                "outcome_ilae": outcome_ilae,
                "outcome_date": outcome_date,
                "surgery_conclusion": surgery_conclusion,
            }
        )

    return pd.DataFrame(rows)


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
    "outcome_engel": {
        "Description": "Seizure outcome at last available follow-up (Engel classification)",
        "Levels": {
            "Ia": "Completely seizure-free since surgery.",
            "Ib": "Non-disabling simple partial seizures only since surgery.",
            "Ic": "Some disabling seizures after surgery, but free of disabling seizures for at least 2 years.",
            "Id": "Generalized convulsions with antiseizure medication withdrawal only.",
            "IIa": "Initially free of disabling seizures, but now has rare seizures.",
            "IIb": "Rare disabling seizures since surgery.",
            "IIc": "More than rare disabling seizures after surgery, but rare seizures for at least 2 years.",
            "IId": "Nocturnal seizures only.",
            "IIIa": "Worthwhile seizure reduction.",
            "IIIb": "Prolonged seizure-free intervals amounting to more than half of the follow-up period, but not less than 2 years.",
            "IVa": "Significant seizure reduction.",
            "IVb": "No appreciable change.",
            "IVc": "Seizures worse.",
            "I": "I, not further specified. Recorded as seizure-free",
            "n/a": "n/a",
        },
    },
    "outcome_ilae": {
        "Description": "Seizure outcome at last available follow-up (ILAE classification)",
        "Levels": {
            "1": "Completely seizure-free; no auras.",
            "2": "Only auras; no other seizures.",
            "3": "One to three seizure days per year, with or without auras.",
            "4": "Four seizure days per year to 50 percent reduction of baseline seizure days, with or without auras.",
            "5": "Less than 50 percent reduction of baseline seizure days to 100 percent increase of baseline seizure days, with or without auras.",
            "6": "More than 100 percent increase of baseline seizure days, with or without auras.",
            "1/2": "1 or 2, not further specified. Recorded as seizure-free",
            "n/a": "n/a",
        },
    },
    "outcome_date": {
        "Description": "Date of last follow-up assessment used for outcome scoring",
    },
    "surgery_conclusion": {
        "Description": "Free-text conclusion from the surgical report (newline characters replaced with ' | ')",
    },
}


def write_participants_files(manifest: pd.DataFrame) -> None:
    subject_ids = sorted(
        bids_subject(s) for s in manifest["subject_id"].astype(str).unique()
    )

    use_enriched = PARTICIPANTS_CSV_DIR is not None and PARTICIPANTS_CSV_DIR.exists()

    if PARTICIPANTS_CSV_DIR is not None and not PARTICIPANTS_CSV_DIR.exists():
        warn(
            f"PARTICIPANTS_CSV_DIR does not exist: {PARTICIPANTS_CSV_DIR}; "
            "falling back to minimal participants files."
        )

    if use_enriched:
        participants_df = extract_participants_from_csv(PARTICIPANTS_CSV_DIR, subject_ids)
        cols = ["participant_id"] + [c for c in participants_df.columns if c != "participant_id"]
        participants_df = participants_df[cols]
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
# TOP-LEVEL WRITING
# =============================================================================

def write_top_level_files(manifest: pd.DataFrame) -> None:
    raw_description = {
        "Name": DATASET_NAME,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "License": "n/a",
        "HEDVersion": [],
        "GeneratedBy": [
            {
                "Name": "scripts_other/bids_conversion/pipeline.py",
                "Description": "Top-level BIDS metadata generation and conversion orchestration.",
            }
        ],
        "SourceDatasets": [
            {
                "URL": "n/a",
            }
        ],
        "Authors": [
            "Sjors Verschuren",
            "Robert Helling",
            "Galia Anguelova",
            "Nicole van Klink",
            "Fernando Gomez-Acebo Ruiz",
            "Maeike Zijlmans"
        ],
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
        "- `anat/` - Pre-operative T1-weighted and FLAIR structural MRI; post-operative T1-weighted MRI\n"
        "- `eeg/` - Pre-operative resting-state scalp EEG in EDF format (originally recorded in TRC/SIG format, "
        "converted via custom sig2edf converter (sig) or FieldTrip (trc)). Raw EEG channel status in "
        "`*_channels.tsv` is derived from the preprocessing conversion logs when available.\n\n"
        "**Derivatives**\n\n"
        "- `resection-masks/` - Volumetric ground-truth resection masks derived from post-operative MRI or via alignment "
        "of intraoperative photographs via slicer-photo2cortex (https://github.com/shaverschuren/slicer-photo2cortex.git); "
        "this derivative also stores per-subject processing reports, comparison reports, atlas labels, harmonised masks, "
        "and dataset-level summary files/figures\n"
        "- `mri-preproc/` - Preprocessed MRI volumes: N4 bias-field correction, skull stripping (HD-BET), "
        "affine registration to MNI152 space (ANTs/FSL), resampling to 1 mm isotropic, "
        "intensity normalisation (robust z-score); final shape 160 x 192 x 160 voxels\n"
        "- `freesurfer-fastsurfer/` - Cortical surface reconstructions produced by FreeSurfer / FastSurfer; "
        "some subject folders also include `pic2mri_output/` scene files from intraoperative photo-to-cortex registration, "
        "kept in place so the (relative) Slicer scene references remain valid\n"
        "- `eeg-preproc/` - Preprocessed (unfiltered) and concatenated EEG recordings in EDF format\n"
        "- `eeg-spikes/` - Filtered interictal epileptiform discharge (spike) segments detected with Persyst 15 "
        "(Perception Score > 0.9), stored as NumPy arrays [n_segments x n_channels x n_samples] "
        "(1 s @ 256 Hz, per-subject z-score, 1-70 Hz or 70-120Hz with 50;60 Hz notch filter)\n\n"
        "The EEG conversion/preprocessing workflow also copies its MATLAB provenance script and per-subject block "
        "conversion logs into `code/eeg/` and `code/eeg_conversion_logs/`.\n\n"
        "**Source data** (`sourcedata/`)\n\n"
        "- Intraoperative cortex photographs (aECoG grid placement)\n\n"
        "## Sessions\n\n"
        "| Label | Description |\n"
        "|-------|-------------|\n"
        "| `ses-preop` | Pre-operative recordings |\n"
        "| `ses-postop` | Post-operative recordings |\n"
        "| `ses-intraop` | Intraoperative recordings / photographs |\n\n"
        "## Notes\n\n"
        "- The original EEG recordings were in TRC/SIG format and have been converted to EDF for BIDS compatibility. "
        "Some of the events were lost in the conversion.\n"
        "- MRI DICOM-to-NIfTI conversion was performed with `dicom2nifti` v2.6.2.\n"
        "- Subject identifiers follow the `sub-RESP****` scheme and correspond to RESPect-ids.\n"
    )
    write_text(BIDS_ROOT / "README", readme)

    write_participants_files(manifest)

    for subject_id, sub_df in manifest.groupby("subject_id"):
        if str(subject_id).lower() == "dataset":
            continue

        sessions_rows = []
        for session in sorted({normalize_session(row) for _, row in sub_df.iterrows()}):
            sessions_rows.append(
                {
                    "session_id": bids_session_entity(session),
                }
            )

        if sessions_rows:
            subject = bids_subject(str(subject_id))
            sessions_df = pd.DataFrame(sessions_rows, columns=["session_id"])
            write_tsv(BIDS_ROOT / subject / f"{subject}_sessions.tsv", sessions_df)


def write_derivative_dataset_descriptions() -> None:
    for pipeline_name in sorted(set(DERIVATIVE_PIPELINES.values())):
        description = "Files copied and organized from source manifest into a BIDS-compatible derivative structure."
        generated_by: list[dict[str, Any]] = [
            {
                "Name": "scripts_other/bids_conversion/convert_data.py",
                "Description": (
                    "Manifest-driven copy/relabel step implemented in "
                    "scripts_other/bids_conversion/convert_data.py."
                ),
            }
        ]

        if pipeline_name == "resection-masks":
            description = (
                "Ground-truth resection mask derivatives, per-subject processing reports, comparison reports, "
                "atlas labeling outputs, harmonised masks, and dataset-level summary artifacts generated from the "
                "ground-truth preprocessing pipeline."
            )
            generated_by.extend(
                [
                    {
                        "Name": "preprocessing/gt/preprocess_gt_loop.py",
                        "Description": (
                            "Batch ground-truth pipeline in preprocessing/gt/preprocess_gt_loop.py that runs "
                            "per-subject GT processing, computes global harmonisation thresholds, and writes "
                            "dataset-level summaries/figures."
                        ),
                    },
                    {
                        "Name": "preprocessing/gt/preprocess_gt.py",
                        "Description": (
                            "Per-subject GT processing in preprocessing/gt/preprocess_gt.py: mask selection, "
                            "atlas labeling, comparison metrics, and report generation."
                        ),
                    },
                    {
                        "Name": "preprocessing/gt/process_eeg_gt.py",
                        "Description": (
                            "GT coordinate summary generation in preprocessing/gt/process_eeg_gt.py "
                            "(gt_coords.json)."
                        ),
                    },
                    {
                        "Name": "slicer-photo2cortex",
                        "CodeURL": "https://github.com/shaverschuren/slicer-photo2cortex",
                        "Description": (
                            "Slicer-based intraoperative photo-to-cortex registration used for pic2mri-derived "
                            "ground-truth masks and scene outputs."
                        ),
                    },
                ]
            )
        elif pipeline_name == "mri-preproc":
            description = (
                "Preprocessed MRI derivatives created by the MRI preprocessing workflow and copied into "
                "BIDS derivative structure."
            )
            generated_by.extend(
                [
                    {
                        "Name": "preprocessing/mri/preprocess_mri.py",
                        "Description": (
                            "MRI preprocessing pipeline in preprocessing/mri/preprocess_mri.py "
                            "(N4 correction, registration, skull stripping, normalization, resampling, QC)."
                        ),
                    },
                    {
                        "Name": "ANTs",
                        "Description": "External image registration tools used by the MRI preprocessing pipeline.",
                    },
                    {
                        "Name": "FSL",
                        "Description": "External neuroimaging toolkit used by the MRI preprocessing pipeline.",
                    },
                    {
                        "Name": "HD-BET",
                        "Description": "External brain extraction tool used in the MRI preprocessing pipeline.",
                    },
                ]
            )
        elif pipeline_name == "freesurfer-fastsurfer":
            description = (
                "FreeSurfer/FastSurfer cortical surface reconstructions plus subject-local Slicer scene folders "
                "(pic2mri_output) used for intraoperative photo-to-cortex registration. The scene folders remain "
                "inside each subject directory so relative links continue to resolve correctly."
            )
            generated_by.extend(
                [
                    {
                        "Name": "FreeSurfer",
                        "Description": "External cortical surface reconstruction software used to generate subject reconstructions.",
                    },
                    {
                        "Name": "FastSurfer",
                        "Description": "External cortical surface reconstruction software used to generate subject reconstructions.",
                    },
                    {
                        "Name": "scripts_other/create_fastsurfer_ds.py",
                        "Description": (
                            "Subject selection/preparation helper in scripts_other/create_fastsurfer_ds.py for "
                            "the FreeSurfer/FastSurfer processing workflow."
                        ),
                    },
                    {
                        "Name": "slicer-photo2cortex",
                        "CodeURL": "https://github.com/shaverschuren/slicer-photo2cortex",
                        "Description": (
                            "Slicer-based intraoperative photo-to-cortex registration producing pic2mri_output "
                            "scene folders inside subject directories."
                        ),
                    },
                ]
            )
        elif pipeline_name == "eeg-preproc":
            description = (
                "EDF EEG derivatives produced by EEG conversion/preprocessing scripts and copied into BIDS, "
                "together with the MATLAB conversion provenance script and per-subject block logs used for "
                "channel-status annotation."
            )
            generated_by.extend(
                [
                    {
                        "Name": "preprocessing/eeg/preprocessing_convertEDF.m",
                        "Description": (
                            "FieldTrip-based TRC/EDF to EDF preprocessing/conversion pipeline in "
                            "preprocessing/eeg/preprocessing_convertEDF.m. Its output conversion TSVs are copied "
                            "into BIDS alongside the script and are used to annotate raw EEG channel status."
                        ),
                    },
                    {
                        "Name": "preprocessing/eeg/sig2edf_batch.py",
                        "Description": (
                            "SIG/STS to EDF conversion pipeline in preprocessing/eeg/sig2edf_batch.py "
                            "(HarmonieReader + MNE)."
                        ),
                    },
                    {
                        "Name": "FieldTrip",
                        "Description": "External MATLAB toolbox used in the EEG conversion workflow.",
                    },
                    {
                        "Name": "MNE-Python",
                        "Description": "External Python toolbox used in SIG/STS to EDF conversion workflow.",
                    },
                ]
            )
        elif pipeline_name == "eeg-spikes":
            description = (
                "EEG spike derivatives (timestamps and filtered/aligned spike segments) generated from Persyst exports "
                "and downstream Python preprocessing scripts."
            )
            generated_by.extend(
                [
                    {
                        "Name": "preprocessing/eeg/process_persyst_batch.ps1",
                        "Description": (
                            "Persyst batch processing/export wrapper in preprocessing/eeg/process_persyst_batch.ps1 "
                            "used to generate detection exports."
                        ),
                    },
                    {
                        "Name": "preprocessing/eeg/extract_event_timestamps.py",
                        "Description": (
                            "Persyst export parsing and timestamp extraction in preprocessing/eeg/extract_event_timestamps.py."
                        ),
                    },
                    {
                        "Name": "preprocessing/eeg/create_eeg_ds.py",
                        "Description": (
                            "Spike-segment filtering/alignment dataset generation in preprocessing/eeg/create_eeg_ds.py."
                        ),
                    },
                    {
                        "Name": "preprocessing/eeg/spike_localisation.py",
                        "Description": (
                            "Spike feature/localisation helper functions used in the EEG spike processing workflow."
                        ),
                    },
                    {
                        "Name": "Persyst 15",
                        "Description": "External clinical EEG software used for initial spike detection exports.",
                    },
                ]
            )

        generated_by[0]["Description"] = description

        payload = {
            "Name": pipeline_name,
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": generated_by,
            "SourceDatasets": [
                {
                    "URL": "n/a",
                    "Description": (
                        "Local RESPect source data and derivatives organized by RESP**** identifiers, including raw "
                        "MRI/EEG inputs, ground-truth masks and summaries, and FreeSurfer/FastSurfer subject directories."
                    ),
                }
            ],
        }
        write_json(BIDS_ROOT / "derivatives" / pipeline_name / "dataset_description.json", payload)

        if pipeline_name == "resection-masks":
            readme = (
                "# Resection masks derivative\n\n"
                "This derivative contains ground-truth outputs generated by the lesion preprocessing pipeline.\n\n"
                "## Contents\n\n"
                "- `sub-<id>/ses-postop/anat/*_desc-resection_mask.nii.gz` - final volumetric ground-truth masks\n"
                "- `sub-<id>/ses-postop/anat/*_processing_report.json` - per-subject processing report\n"
                "- `sub-<id>/ses-postop/anat/*_comparison.json` - unharmonised comparison metrics\n"
                "- `sub-<id>/ses-postop/anat/*_comparison_harmonised.json` - harmonised comparison metrics\n"
                "- `sub-<id>/ses-postop/anat/*_atlas_labeling_pic2mri.json` - atlas label summary for the pic2mri mask\n"
                "- `sub-<id>/ses-postop/anat/*_pic2mri_smooth.nii.gz` - smoothed probability map for thresholding\n"
                "- `sub-<id>/ses-postop/anat/*_pic2mri_harmonised.nii.gz` and `*_gt_mask_harmonised.nii.gz` - harmonised masks\n"
                "- `summary/processing_summary.json` - dataset-level processing summary\n"
                "- `summary/gt_hemi.json`, `summary/gt_lobe.json`, `summary/gt_coords.json` - dataset-level GT labels\n"
                "- `summary/comparison_summary_unharmonised.json`, `summary/comparison_summary_harmonised.json` - summary metrics\n"
                "- `summary/optimal_threshold.json` - harmonisation threshold selection\n"
                "- `figures/precision_recall_curve.png`, `figures/comparison_plot_extended.png` - overview figures\n\n"
                "The dataset was generated from the local ground-truth preprocessing outputs stored under `data/preprocessing/gt`. "
                "Any subject-level `pic2mri_output/` folders remain inside `derivatives/freesurfer-fastsurfer/sub-<id>/` so Slicer scene links stay intact."
            )
            write_text(BIDS_ROOT / "derivatives" / pipeline_name / "README", readme)
        elif pipeline_name == "freesurfer-fastsurfer":
            readme = (
                "# FreeSurfer/FastSurfer derivative\n\n"
                "This derivative contains subject-level cortical surface reconstructions produced by FreeSurfer or "
                "FastSurfer. Some subject folders also contain a `pic2mri_output/` subfolder with Slicer scene files "
                "from intraoperative photo-to-cortex registration. Those files are intentionally kept in the subject "
                "folder because the scene references are relative to that location.\n\n"
                "## Contents\n\n"
                "- `sub-<id>/` - FreeSurfer/FastSurfer subject directory\n"
                "- `sub-<id>/pic2mri_output/` - Slicer scene and support files for photo-to-cortex registration, when present\n\n"
                "The files in this derivative are copied from the source FreeSurfer/FastSurfer directories without "
                "changing their internal structure."
            )
            write_text(BIDS_ROOT / "derivatives" / pipeline_name / "README", readme)


def main() -> None:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)
    write_top_level_files(manifest)
    write_derivative_dataset_descriptions()


if __name__ == "__main__":
    main()
