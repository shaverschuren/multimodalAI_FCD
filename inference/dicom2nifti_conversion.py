"""Convert DICOM folders to pre-op MRI NIfTI files.

Each direct subfolder inside --dicom_root is treated as one DICOM series.
Metadata is inspected to infer T1w/FLAIR and pseudo ID, and outputs are named:
<PSEUDO_ID>-preop-<T1w|FLAIR>.nii.gz
"""

import argparse
import gzip
import os
import re
import shutil
import tempfile
from glob import glob
from typing import Optional

from dicom2nifti import convert_directory
from dicom2nifti import settings
import pydicom
from tqdm import tqdm

def convert_dicom_to_nifti(dicom_dir: str, output_file: str) -> str:
    """Convert a DICOM directory to a single NIfTI file.

    Mirrors the conversion behavior from scripts_other/create_nifti_dataset.py,
    including a retry with slice increment validation disabled.
    """
    parent_dir = os.path.dirname(output_file)
    tmp_dir = tempfile.mkdtemp(prefix="tmp_dcm2nii_", dir=parent_dir)

    try:
        convert_directory(dicom_dir, tmp_dir, compression=False, reorient=True)
        nii_files = glob(os.path.join(tmp_dir, "*.nii"))

        if not nii_files:
            print(
                f"Warning: Standard conversion failed for {dicom_dir}. "
                "Retrying with slice increment validation disabled."
            )
            settings.disable_validate_slice_increment()
            try:
                convert_directory(dicom_dir, tmp_dir, compression=False, reorient=True)
            finally:
                settings.enable_validate_slice_increment()

            nii_files = glob(os.path.join(tmp_dir, "*.nii"))
            if not nii_files:
                raise RuntimeError("No NIfTI file generated after retry.")

        # Keep conversion behavior unchanged (uncompressed conversion), then gzip
        # so outputs follow the requested .nii.gz naming.
        with open(nii_files[0], "rb") as src, gzip.open(output_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return f"{dicom_dir}: Done"
    except Exception as exc:
        return f"Conversion failed for {dicom_dir}: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_pseudo_id(text: str) -> Optional[str]:
    match = re.search(r"(RESP\d+)", text.upper())
    return match.group(1) if match else None


def _find_first_dicom_file(dicom_dir: str) -> Optional[str]:
    for root, _, files in os.walk(dicom_dir):
        for filename in sorted(files):
            candidate = os.path.join(root, filename)
            try:
                pydicom.dcmread(candidate, stop_before_pixels=True, force=True)
                return candidate
            except Exception:
                continue
    return None


def infer_scan_type_and_pseudo_id(dicom_dir: str, folder_name: str) -> tuple[str, str]:
    """Infer scan type (T1w/FLAIR) and pseudo ID from DICOM metadata."""
    dicom_file = _find_first_dicom_file(dicom_dir)

    sequence_text = ""
    pseudo_id: Optional[str] = _extract_pseudo_id(folder_name)

    if dicom_file is not None:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True, force=True)
        series_description = str(getattr(ds, "SeriesDescription", "") or "")
        protocol_name = str(getattr(ds, "ProtocolName", "") or "")
        sequence_name = str(getattr(ds, "SequenceName", "") or "")
        scanning_sequence = str(getattr(ds, "ScanningSequence", "") or "")
        sequence_text = " ".join(
            [series_description, protocol_name, sequence_name, scanning_sequence]
        ).lower()

        if pseudo_id is None:
            for field in ("PatientID", "PatientName", "StudyID", "AccessionNumber"):
                value = str(getattr(ds, field, "") or "")
                pseudo_id = _extract_pseudo_id(value)
                if pseudo_id is not None:
                    break

    if pseudo_id is None:
        pseudo_id = folder_name

    if "flair" in sequence_text:
        scan_type = "FLAIR"
    elif "t1" in sequence_text or "mpr" in sequence_text or "spgr" in sequence_text:
        scan_type = "T1w"
    else:
        scan_type = "T1w"

    return scan_type, pseudo_id


def process_dicom_folders(dicom_root: str, output_root: str, overwrite: bool = False) -> None:
    """Convert all direct DICOM subfolders under dicom_root to NIfTI."""
    os.makedirs(output_root, exist_ok=True)

    dicom_folders = sorted(
        folder
        for folder in os.listdir(dicom_root)
        if os.path.isdir(os.path.join(dicom_root, folder))
    )

    print(f"Found {len(dicom_folders)} DICOM folders in {dicom_root}.")
    logs = []

    for folder in tqdm(dicom_folders, desc="DICOM folders"):
        dicom_dir = os.path.join(dicom_root, folder)
        try:
            scan_type, pseudo_id = infer_scan_type_and_pseudo_id(dicom_dir, folder)
            out_file = os.path.join(output_root, f"{pseudo_id}-preop-{scan_type}.nii.gz")
        except Exception as exc:
            logs.append(f"Metadata parse failed for {dicom_dir}: {exc}")
            continue

        if os.path.exists(out_file) and not overwrite:
            logs.append(f"File already exists: {out_file}")
            continue

        logs.append(convert_dicom_to_nifti(dicom_dir, out_file))

    log_path = os.path.join(output_root, "conversion_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        for log in logs:
            if log:
                log_file.write(log + "\n")

    print(f"Logs saved to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert direct DICOM subfolders in --dicom_root to .nii.gz files "
            "named as <PSEUDO_ID>-preop-<T1w|FLAIR>.nii.gz."
        )
    )
    parser.add_argument(
        "--dicom_root",
        required=True,
        help="Input directory containing DICOM subfolders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output directory where <folder_name>.nii files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output NIfTI files.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dicom_root):
        raise NotADirectoryError(f"dicom_root does not exist or is not a directory: {args.dicom_root}")

    process_dicom_folders(args.dicom_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
