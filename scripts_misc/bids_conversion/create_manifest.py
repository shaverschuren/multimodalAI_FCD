from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


# =============================================================================
# USER CONFIG
# =============================================================================

SUBJECT_LIST_CSV = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\selected_summary.csv")
SUBJECT_ID_COLUMN = "Participant Id"

OUT_MANIFEST_CSV = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\data_availability\BIDS_conversion\source_manifest.csv")
OUT_SUMMARY_TXT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\data_availability\BIDS_conversion\source_summary.txt")
OUT_SUMMARY_CSV = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\data_availability\BIDS_conversion\source_summary.csv")

# Raw data
RAW_EEG_EDF_ROOTS = [
    Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\sig2edf"),
    Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\trc2edf"),
]
RAW_MRI_PREOP_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\dataset_mri\pre_operative")
RAW_MRI_POSTOP_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\dataset_mri\post_operative")

# Source/non-BIDS auxiliary data
CORTEX_PICTURES_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\dataset_ECoG_pictures")

# Derivatives
GT_MASK_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\gt")
PREPROC_MRI_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\mri")
FREESURFER_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\dataset_fs")
PREPROC_EEG_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\EDFdata")
EEG_SPIKES_ROOT = Path(r"\\ds\data\HER\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\eeg\spikes")

GT_DATASET_SUMMARY_FILES = {
    "processing_summary.json",
    "gt_hemi.json",
    "gt_lobe.json",
    "gt_coords.json",
    "comparison_summary_unharmonised.json",
    "comparison_summary_harmonised.json",
    "optimal_threshold.json",
}

GT_DATASET_FIGURE_FILES = {
    "precision_recall_curve.png",
    "comparison_plot_extended.png",
}


# =============================================================================
# SEARCH SETTINGS
# =============================================================================

PATIENT_ID_REGEX = re.compile(r"^RESP.*$", re.IGNORECASE)

NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
EDF_EXTENSIONS = {".edf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
NPY_EXTENSIONS = {".npy"}

GT_MASK_FILENAME_TEMPLATE = "{subject_id}_gt_mask.nii.gz"

MRI_CONTRAST_KEYWORDS = {
    "T1w": ["t1", "t1w", "mprage", "spgr"],
    "T2w": ["t2", "t2w"],
    "FLAIR": ["flair"],
}

CORTEX_PRE_KEYWORDS = ["pre", "before"]
CORTEX_POST_KEYWORDS = ["post", "after"]


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ManifestRow:
    subject_id: str
    source_category: str
    datatype: str
    session: str
    modality: str
    source_path: str
    source_filename: str
    file_extension: str
    bids_derivative: bool
    bids_suffix: str
    run: Optional[str]
    notes: str


# =============================================================================
# BASIC HELPERS
# =============================================================================

def get_extension(path: Path) -> str:
    """Return .nii.gz as a single extension where relevant."""
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def path_text(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def load_selected_subjects(csv_path: Path) -> list[str]:
    subject_ids = (
        pd.read_csv(csv_path)[SUBJECT_ID_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return subject_ids


def subject_folder(root: Path, subject_id: str) -> Optional[Path]:
    """
    Return root / subject_id if it exists.

    This intentionally does not recursively search the root.
    """
    candidate = root / subject_id
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def list_files(folder: Optional[Path], extensions: Optional[set[str]] = None) -> list[Path]:
    """
    Recursively list files inside a known subject folder.

    This is allowed because the folder is already the patient-specific folder.
    """
    if folder is None:
        return []

    files = [p for p in folder.rglob("*") if p.is_file()]

    if extensions is not None:
        files = [p for p in files if get_extension(p) in extensions]

    return sorted(files)


def infer_mri_suffix(path: Path) -> str:
    txt = path_text(path)

    for suffix, keywords in MRI_CONTRAST_KEYWORDS.items():
        if contains_any(txt, keywords):
            return suffix

    return "UNKNOWN"


def infer_cortex_picture_phase(path: Path) -> str:
    """
    Intra-operative pictures may be pre- or post-resection.
    This is not the same as pre-op/post-op MRI session.
    """
    txt = path_text(path)

    has_pre = contains_any(txt, CORTEX_PRE_KEYWORDS)
    has_post = contains_any(txt, CORTEX_POST_KEYWORDS)

    if has_pre and not has_post:
        return "pre-resection"
    if has_post and not has_pre:
        return "post-resection"
    return "unknown-resection-phase"


# =============================================================================
# SCANNERS PER DATA SOURCE
# =============================================================================

def scan_raw_eeg(subject_id: str) -> list[ManifestRow]:
    files = []
    for root in RAW_EEG_EDF_ROOTS:
        folder = subject_folder(root, subject_id)
        files.extend(list_files(folder, EDF_EXTENSIONS))

    files = sorted(files)

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="raw_eeg_edf",
                datatype="eeg",
                session="ses-unknown",
                modality="eeg",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=False,
                bids_suffix="eeg",
                run=None,
                notes="Raw EEG EDF candidate; session/run assignment may need refinement",
            )
        )

    return rows


def scan_raw_mri_preop(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(RAW_MRI_PREOP_ROOT, subject_id)
    files = list_files(folder, NIFTI_EXTENSIONS)

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="raw_mri_preop",
                datatype="anat",
                session="ses-preop",
                modality="mri",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=False,
                bids_suffix=infer_mri_suffix(file),
                run=None,
                notes="Raw pre-op MRI NIfTI candidate",
            )
        )

    return rows


def scan_raw_mri_postop(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(RAW_MRI_POSTOP_ROOT, subject_id)
    files = list_files(folder, NIFTI_EXTENSIONS)

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="raw_mri_postop",
                datatype="anat",
                session="ses-postop",
                modality="mri",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=False,
                bids_suffix=infer_mri_suffix(file),
                run=None,
                notes="Raw post-op MRI NIfTI candidate",
            )
        )

    return rows


def scan_cortex_pictures(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(CORTEX_PICTURES_ROOT, subject_id)
    files = list_files(folder, IMAGE_EXTENSIONS)

    rows = []
    for file in files:
        phase = infer_cortex_picture_phase(file)
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="intraoperative_cortex_picture",
                datatype="sourcedata",
                session="ses-intraop",
                modality="photo",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=False,
                bids_suffix="photo",
                run=None,
                notes=f"Intra-operative cortex picture; resection phase: {phase}",
            )
        )

    return rows


def scan_gt_masks(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(GT_MASK_ROOT, subject_id)
    files = list_files(folder)

    rows = []

    expected_mask_name = GT_MASK_FILENAME_TEMPLATE.format(subject_id=subject_id)

    for file in files:
        if file.name == expected_mask_name:
            modality = "resection_mask"
            bids_suffix = "mask"
            notes = "Final ground-truth resection mask"
        elif file.name.endswith("_processing_report.json"):
            modality = "ground_truth_processing_report"
            bids_suffix = "report"
            notes = "Per-subject ground-truth processing report"
        elif file.name.endswith("_comparison.json"):
            modality = "ground_truth_comparison_report"
            bids_suffix = "comparison"
            notes = "Per-subject unharmonised comparison report"
        elif file.name.endswith("_comparison_harmonised.json"):
            modality = "ground_truth_comparison_harmonised_report"
            bids_suffix = "comparison"
            notes = "Per-subject harmonised comparison report"
        elif file.name.endswith("_atlas_labeling_pic2mri.json"):
            modality = "ground_truth_atlas_labeling"
            bids_suffix = "atlas"
            notes = "Per-subject atlas labeling report for pic2mri mask"
        elif file.name.endswith("_pic2mri_smooth.nii.gz"):
            modality = "ground_truth_smooth_map"
            bids_suffix = "smooth"
            notes = "Per-subject smoothed probability map for harmonisation"
        elif file.name.endswith("_pic2mri_harmonised.nii.gz"):
            modality = "ground_truth_harmonised_mask"
            bids_suffix = "harmonised"
            notes = "Per-subject harmonised pic2mri mask"
        elif file.name.endswith("_gt_mask_harmonised.nii.gz"):
            modality = "ground_truth_harmonised_gt_mask"
            bids_suffix = "harmonised"
            notes = "Per-subject harmonised ground-truth mask"
        else:
            modality = "ground_truth_auxiliary"
            bids_suffix = "UNKNOWN"
            notes = "Auxiliary ground-truth derivative file/log"

        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="ground_truth",
                datatype="derivative",
                session="ses-postop",
                modality=modality,
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=True,
                bids_suffix=bids_suffix,
                run=None,
                notes=notes,
            )
        )

    return rows


def scan_gt_dataset_outputs() -> list[ManifestRow]:
    files = list_files(GT_MASK_ROOT)

    rows = []
    for file in files:
        if file.parent != GT_MASK_ROOT:
            continue

        if file.name in GT_DATASET_SUMMARY_FILES:
            modality = "ground_truth_dataset_summary"
            bids_suffix = "summary"
            notes = "Dataset-level ground-truth summary artifact"
        elif file.name in GT_DATASET_FIGURE_FILES:
            modality = "ground_truth_dataset_figure"
            bids_suffix = "figure"
            notes = "Dataset-level ground-truth figure artifact"
        else:
            continue

        rows.append(
            ManifestRow(
                subject_id="dataset",
                source_category="ground_truth_dataset",
                datatype="derivative",
                session="ses-postop",
                modality=modality,
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=True,
                bids_suffix=bids_suffix,
                run=None,
                notes=notes,
            )
        )

    return rows


def scan_preproc_mri(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(PREPROC_MRI_ROOT, subject_id)
    files = list_files(folder)

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="preprocessed_mri",
                datatype="derivative",
                session="ses-unknown",
                modality="preprocessed_mri",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=True,
                bids_suffix=infer_mri_suffix(file) if get_extension(file) in NIFTI_EXTENSIONS else "UNKNOWN",
                run=None,
                notes="Preprocessed MRI derivative",
            )
        )

    return rows


def scan_freesurfer(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(FREESURFER_ROOT, subject_id)

    rows = []

    if folder is None:
        return rows

    # For FreeSurfer/FastSurfer, storing one row for the patient folder is usually more useful
    # than one row for thousands of internal files.
    rows.append(
        ManifestRow(
            subject_id=subject_id,
            source_category="freesurfer_fastsurfer",
            datatype="derivative",
            session="ses-unknown",
            modality="surface_reconstruction",
            source_path=str(folder),
            source_filename=folder.name,
            file_extension="directory",
            bids_derivative=True,
            bids_suffix="freesurfer",
            run=None,
            notes="FreeSurfer/FastSurfer subject directory",
        )
    )

    return rows


def scan_preproc_eeg(subject_id: str) -> list[ManifestRow]:
    folder = subject_folder(PREPROC_EEG_ROOT, subject_id)
    files = list_files(folder, EDF_EXTENSIONS)

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="preprocessed_eeg",
                datatype="derivative",
                session="ses-unknown",
                modality="preprocessed_eeg",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=True,
                bids_suffix="eeg",
                run=None,
                notes="Preprocessed/concatenated EEG derivative EDF",
            )
        )

    return rows


def scan_eeg_spikes(subject_id: str) -> list[ManifestRow]:
    """
    EEG spike derivatives are stored in one flat folder, for example:
      RESP1234_spikes_1-70Hz.npy
      RESP1234_spikes_70-120Hz.npy
    """
    if not EEG_SPIKES_ROOT.exists():
        return []

    pattern = f"{subject_id}_spikes_*.npy"
    files = sorted(EEG_SPIKES_ROOT.glob(pattern))

    rows = []
    for file in files:
        rows.append(
            ManifestRow(
                subject_id=subject_id,
                source_category="eeg_spikes",
                datatype="derivative",
                session="ses-unknown",
                modality="eeg_spikes",
                source_path=str(file),
                source_filename=file.name,
                file_extension=get_extension(file),
                bids_derivative=True,
                bids_suffix="spikes",
                run=None,
                notes="Processed EEG spike recording derivative",
            )
        )

    return rows


# =============================================================================
# POST-PROCESSING
# =============================================================================

def assign_eeg_runs(rows: list[ManifestRow]) -> list[ManifestRow]:
    """
    Assign run labels within each subject/session/source_category for EEG-like files.
    """
    eeg_categories = {"raw_eeg_edf", "preprocessed_eeg"}

    groups: dict[tuple[str, str, str], list[int]] = {}

    for idx, row in enumerate(rows):
        if row.source_category in eeg_categories:
            key = (row.subject_id, row.session, row.source_category)
            groups.setdefault(key, []).append(idx)

    for _, indices in groups.items():
        indices = sorted(indices, key=lambda i: rows[i].source_path)
        for run_nr, idx in enumerate(indices, start=1):
            rows[idx].run = f"run-{run_nr:02d}"

    return rows


def scan_subject(subject_id: str) -> list[ManifestRow]:
    rows = []

    rows.extend(scan_raw_eeg(subject_id))
    rows.extend(scan_raw_mri_preop(subject_id))
    rows.extend(scan_raw_mri_postop(subject_id))
    rows.extend(scan_cortex_pictures(subject_id))
    rows.extend(scan_gt_masks(subject_id))
    rows.extend(scan_preproc_mri(subject_id))
    rows.extend(scan_freesurfer(subject_id))
    rows.extend(scan_preproc_eeg(subject_id))
    rows.extend(scan_eeg_spikes(subject_id))

    return rows


def rows_to_dataframe(rows: list[ManifestRow]) -> pd.DataFrame:
    columns = [
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
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame([asdict(row) for row in rows], columns=columns)


# =============================================================================
# SUMMARY
# =============================================================================

def folder_exists_for_subject(root: Path, subject_id: str) -> bool:
    return subject_folder(root, subject_id) is not None


def count_rows(rows: list[ManifestRow], subject_id: str, source_category: str) -> int:
    return sum(
        1
        for row in rows
        if row.subject_id == subject_id and row.source_category == source_category
    )


def count_final_gt_masks(rows: list[ManifestRow], subject_id: str) -> int:
    return sum(
        1
        for row in rows
        if row.subject_id == subject_id
        and row.source_category == "ground_truth"
        and row.modality == "resection_mask"
    )


def format_present(count: int) -> str:
    return f"present ({count})" if count > 0 else "missing"


def build_subject_summary_record(subject_id: str, rows: list[ManifestRow]) -> dict[str, object]:
    raw_eeg = count_rows(rows, subject_id, "raw_eeg_edf")
    preop_mri = count_rows(rows, subject_id, "raw_mri_preop")
    postop_mri = count_rows(rows, subject_id, "raw_mri_postop")
    cortex_pictures = count_rows(rows, subject_id, "intraoperative_cortex_picture")
    gt_all = count_rows(rows, subject_id, "ground_truth")
    gt_final_mask = count_final_gt_masks(rows, subject_id)
    preproc_mri = count_rows(rows, subject_id, "preprocessed_mri")
    freesurfer = count_rows(rows, subject_id, "freesurfer_fastsurfer")
    preproc_eeg = count_rows(rows, subject_id, "preprocessed_eeg")
    eeg_spikes = count_rows(rows, subject_id, "eeg_spikes")

    missing_folder_notes = []

    checks = [
        ("raw pre-op MRI folder", RAW_MRI_PREOP_ROOT),
        ("raw post-op MRI folder", RAW_MRI_POSTOP_ROOT),
        ("cortex pictures folder", CORTEX_PICTURES_ROOT),
        ("ground-truth folder", GT_MASK_ROOT),
        ("preprocessed MRI folder", PREPROC_MRI_ROOT),
        ("FreeSurfer/FastSurfer folder", FREESURFER_ROOT),
        ("preprocessed EEG folder", PREPROC_EEG_ROOT),
    ]

    if not any(folder_exists_for_subject(root, subject_id) for root in RAW_EEG_EDF_ROOTS):
        missing_folder_notes.append("raw EEG EDF folders (sig2edf/trc2edf)")

    for label, root in checks:
        if not folder_exists_for_subject(root, subject_id):
            missing_folder_notes.append(label)

    return {
        "subject_id": subject_id,
        "raw_eeg_edfs": format_present(raw_eeg),
        "raw_preop_mri_niftis": format_present(preop_mri),
        "raw_postop_mri_niftis": format_present(postop_mri),
        "intraoperative_cortex_pictures": format_present(cortex_pictures),
        "ground_truth_folder_files": format_present(gt_all),
        "final_ground_truth_mask": format_present(gt_final_mask),
        "preprocessed_mri_derivatives": format_present(preproc_mri),
        "freesurfer_fastsurfer_directory": format_present(freesurfer),
        "preprocessed_eeg_edfs": format_present(preproc_eeg),
        "eeg_spikes_npy_files": format_present(eeg_spikes),
        "missing_patient_subfolders": "; ".join(missing_folder_notes),
    }


def build_subject_summary(subject_id: str, rows: list[ManifestRow]) -> str:
    summary = build_subject_summary_record(subject_id, rows)

    lines = [
        f"{subject_id}",
        f"  raw EEG EDFs: {summary['raw_eeg_edfs']}",
        f"  raw pre-op MRI NIfTIs: {summary['raw_preop_mri_niftis']}",
        f"  raw post-op MRI NIfTIs: {summary['raw_postop_mri_niftis']}",
        f"  intra-operative cortex pictures: {summary['intraoperative_cortex_pictures']}",
        f"  ground-truth folder files: {summary['ground_truth_folder_files']}",
        f"  final ground-truth mask: {summary['final_ground_truth_mask']}",
        f"  preprocessed MRI derivatives: {summary['preprocessed_mri_derivatives']}",
        f"  FreeSurfer/FastSurfer directory: {summary['freesurfer_fastsurfer_directory']}",
        f"  preprocessed/concatenated EEG EDFs: {summary['preprocessed_eeg_edfs']}",
        f"  EEG spike .npy files: {summary['eeg_spikes_npy_files']}",
    ]
    missing_folder_notes = summary["missing_patient_subfolders"]

    if missing_folder_notes:
        lines.append("  missing patient subfolders:")
        for label in str(missing_folder_notes).split("; "):
            lines.append(f"    - {label}")

    return "\n".join(lines)


def build_summary_table(selected_subjects: list[str], all_rows: list[ManifestRow]) -> pd.DataFrame:
    records = [build_subject_summary_record(subject_id, all_rows) for subject_id in selected_subjects]
    return pd.DataFrame(records)


def write_summary(
    selected_subjects: list[str],
    all_rows: list[ManifestRow],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Source-data availability summary",
        "=" * 55,
        "",
        f"Subject list CSV: {SUBJECT_LIST_CSV}",
        f"Number of selected subjects: {len(selected_subjects)}",
        f"Number of manifest rows: {len(all_rows)}",
        "",
        "Configured source roots:",
        f"  RAW_EEG_EDF_ROOTS: {', '.join(str(root) for root in RAW_EEG_EDF_ROOTS)}",
        f"  RAW_MRI_PREOP_ROOT: {RAW_MRI_PREOP_ROOT}",
        f"  RAW_MRI_POSTOP_ROOT: {RAW_MRI_POSTOP_ROOT}",
        f"  CORTEX_PICTURES_ROOT: {CORTEX_PICTURES_ROOT}",
        f"  GT_MASK_ROOT: {GT_MASK_ROOT}",
        f"  PREPROC_MRI_ROOT: {PREPROC_MRI_ROOT}",
        f"  FREESURFER_ROOT: {FREESURFER_ROOT}",
        f"  PREPROC_EEG_ROOT: {PREPROC_EEG_ROOT}",
        f"  EEG_SPIKES_ROOT: {EEG_SPIKES_ROOT}",
        "",
        "Per-subject summary:",
        "",
    ]

    for subject_id in selected_subjects:
        lines.append(build_subject_summary(subject_id, all_rows))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(
    selected_subjects: list[str],
    all_rows: list[ManifestRow],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary_table(selected_subjects, all_rows)
    summary_df.to_csv(out_path, index=False)


# =============================================================================
# ENTRYPOINT
# =============================================================================

def create_manifest() -> pd.DataFrame:
    OUT_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    selected_subjects = load_selected_subjects(SUBJECT_LIST_CSV)

    all_rows: list[ManifestRow] = []

    for subject_id in selected_subjects:
        if not PATIENT_ID_REGEX.match(subject_id):
            print(f"WARNING: selected subject does not match RESP**** pattern: {subject_id}")

        all_rows.extend(scan_subject(subject_id))

    all_rows.extend(scan_gt_dataset_outputs())

    all_rows = assign_eeg_runs(all_rows)

    manifest_df = rows_to_dataframe(all_rows)
    manifest_df.to_csv(OUT_MANIFEST_CSV, index=False)

    write_summary(
        selected_subjects=selected_subjects,
        all_rows=all_rows,
        out_path=OUT_SUMMARY_TXT,
    )
    write_summary_csv(
        selected_subjects=selected_subjects,
        all_rows=all_rows,
        out_path=OUT_SUMMARY_CSV,
    )

    print(f"Wrote manifest CSV: {OUT_MANIFEST_CSV}")
    print(f"Wrote summary TXT: {OUT_SUMMARY_TXT}")
    print(f"Wrote summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Subjects: {len(selected_subjects)}")
    print(f"Manifest rows: {len(all_rows)}")

    return manifest_df


def main() -> None:
    create_manifest()


if __name__ == "__main__":
    main()
