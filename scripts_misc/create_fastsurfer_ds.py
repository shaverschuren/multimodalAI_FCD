import os
import nibabel as nib
from scipy.ndimage import affine_transform
import numpy as np
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
from datetime import datetime

"""
Create a dataset of subjects present in mri_dir but missing in fs_dir.
If a subject folder already exists in fs_dir, compare the T1w scan to the
FreeSurfer orig/001.mgz to check if they are identical so we can skip.
As these images are often preprocessed (e.g. resampled, de-faced etc.),
we use mutual information within a brain mask to compare them.

Edit the three paths below (in the __main__ section) before running.
"""

def _timestamp():
    """Return current timestamp as a human-readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def open_log(log_path):
    """Open log file for appending and return handle."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(log_path, "a", encoding="utf-8")
        f.write(f"\n{'='*80}\n[{_timestamp()}] Log started\n{'='*80}\n")
        f.flush()
        return f
    except Exception as e:
        tqdm.write(f"Could not open log file {log_path}: {e}")
        return None

def close_log(f):
    """Close log file if open."""
    if f:
        try:
            f.write(f"[{_timestamp()}] Log closed\n{'='*80}\n")
            f.close()
        except Exception:
            pass

def make_logger(log_f):
    """Return a logger function that writes to console and log file."""
    def logger(msg):
        tqdm.write(msg)
        if log_f:
            try:
                log_f.write(f"[{_timestamp()}] {msg}\n")
                log_f.flush()
            except Exception:
                pass
    return logger

def check_resolution(img_path, logger, min_voxel_size=2.0):
    """Check if the image has sufficient resolution (voxel size <= min_voxel_size mm)."""
    try:
        img = nib.load(img_path)
        zooms = img.header.get_zooms()[:3]
        if all(z <= min_voxel_size for z in zooms):
            return True
        else:
            return False
    except Exception as e:
        logger(f"Error checking resolution for {img_path}: {e}")
        return False

def compare_scans(img1_path, img2_path, brain_mask_path, logger, mi_thresh=0.9, bins=64):
    """
    Compare two NIfTI/MGZ images using mutual information within a brain mask.
    Automatically resamples the second image to the first if affines differ.
    Returns True if they plausibly match (same scan / same space), False otherwise.
    """

    logger("-"*40)
    logger(f"Comparing {os.path.basename(img1_path)} with {os.path.basename(img2_path)}")

    try:
        img1 = nib.load(img1_path)
        img2 = nib.load(img2_path)
        mask_img = nib.load(brain_mask_path)
    except Exception as e:
        logger(f"Error loading files: {e}")
        logger("-"*40)
        return False

    data1 = img1.get_fdata(dtype=np.float32)
    data2 = img2.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata().astype(bool)

    # --- If affines differ, resample img2 into img1 space ---
    if not np.allclose(img1.affine, img2.affine, atol=1e-3) or data1.shape != data2.shape:
        logger("Affine or shape differ — resampling second image to first.")
        A = np.linalg.inv(img2.affine) @ img1.affine
        matrix = A[:3, :3]
        offset = A[:3, 3]
        data2 = affine_transform(
            data2,
            matrix=matrix,
            offset=offset,
            output_shape=data1.shape,
            order=1
        )

    # --- Resample mask if needed ---
    if mask.shape != data1.shape:
        logger("Mask shape does not match image shape — resampling mask.")
        A_mask = np.linalg.inv(mask_img.affine) @ img1.affine
        matrix_mask = A_mask[:3, :3]
        offset_mask = A_mask[:3, 3]
        mask = affine_transform(
            mask.astype(np.float32),
            matrix=matrix_mask,
            offset=offset_mask,
            output_shape=data1.shape,
            order=0
        ) > 0.5

    if np.sum(mask) == 0:
        logger("Empty or invalid brain mask.")
        logger("-"*40)
        return False

    data1_masked = data1[mask]
    data2_masked = data2[mask]

    def mutual_information(x, y, bins=64):
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) == 0:
            return 0.0
        x = x[valid]
        y = y[valid]
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1, keepdims=True)
        py = np.sum(pxy, axis=0, keepdims=True)
        p_prod = px @ py
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / p_prod[nzs]))
        return float(mi)

    mi_value = mutual_information(data1_masked, data2_masked, bins=bins)
    logger(f"Mutual information (masked): {mi_value:.3f}")

    if mi_value >= mi_thresh:
        logger(">> MATCH")
        logger("-"*40)
        return True
    else:
        logger(">> DIFFERENT SCAN")
        logger("-"*40)
        return False


def main(mri_dir, fs_dir, out_dir, reprocess_list, only_reprocess=False):
    """Create dataset of subjects for FreeSurfer processing."""

    # Prepare log file
    log_path = out_dir / "conversion_logs.txt"
    log_f = open_log(log_path)
    logger = make_logger(log_f)

    logger(f"Starting dataset creation.")
    logger(f"Input MRI dir: {mri_dir}")
    logger(f"FreeSurfer dir: {fs_dir}")
    logger(f"Output dir: {out_dir}")

    # Make output dir and define subjects file path
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects_file = out_dir / "subjects_list.txt"

    # Read existing subjects file if present
    existing = set()
    if subjects_file.exists():
        with subjects_file.open("r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}

    to_add = []

    if not mri_dir.is_dir():
        logger(f"mri_dir does not exist or is not a directory: {mri_dir}")
        close_log(log_f)
        sys.exit(1)
    if not fs_dir.is_dir():
        logger(f"fs_dir does not exist or is not a directory: {fs_dir}")
        close_log(log_f)
        sys.exit(1)

    # Loop over subjects in mri_dir
    items = [p for p in sorted(mri_dir.iterdir()) if p.is_dir()]
    for item in tqdm(items, desc="Subjects", unit="subj"):
        name = item.name
        logger(f"PROCESSING: {name}")

        if (item / f"{name}-preop-T1w.nii").exists():
            sufficient_res = check_resolution(item / f"{name}-preop-T1w.nii", min_voxel_size=2.0, logger=logger)
            if not sufficient_res:
                logger(f"{name}: -preop-T1w.nii does not meet minimum resolution requirements, skipping.")
                logger("")
                continue
        else:
            logger(f"{name}: no -preop-T1w.nii file found, skipping.")
            logger("")
            continue
        
        if only_reprocess:
            if name not in reprocess_list:
                logger(f"{name}: only_reprocess is set and subject not in reprocess list, skipping.")
                logger("")
                continue
            else:
                logger(f"{name}: only_reprocess is set and subject is in reprocess list, proceeding with processing.")
        else:
            if (fs_dir / name).exists():
                logger(f"FS folder exists for subject {name}")
                # First, check if output is complete
                fs_output_complete = (
                    (fs_dir / name / "surf" / "lh.pial").exists() and
                    (fs_dir / name / "surf" / "rh.pial").exists() and
                    (fs_dir / name / "surf" / "lh.curv").exists() and
                    (fs_dir / name / "surf" / "rh.curv").exists() and
                    (fs_dir / name / "surf" / "lh.sulc").exists() and
                    (fs_dir / name / "surf" / "rh.sulc").exists() and
                    (fs_dir / name / "mri" / "brainmask.mgz").exists() and
                    (fs_dir / name / "mri" / "aparc+aseg.mgz").exists() and
                    (fs_dir / name / "mri" / "aparc.a2009s+aseg.mgz").exists() and
                    (fs_dir / name / "mri" / "wmparc.mgz").exists()
                )
                if not fs_output_complete:
                    logger(f"{name}: FreeSurfer output incomplete, will reprocess.")
                else:
                    logger(f"{name}: FreeSurfer output complete, checking for scan differences.")
                    # Determine scan to compare against
                    scan_to_compare = fs_dir / name / "mri" / "orig" / "001.mgz"
                    if not os.path.exists(scan_to_compare):
                        scan_to_compare = fs_dir / name / "mri" / "T1.mgz"
                    if not os.path.exists(scan_to_compare):
                        scan_to_compare = fs_dir / name / "mri" / "T1.nii"
                    # Compare if exists
                    if not os.path.exists(scan_to_compare):
                        logger(f"{name}: no suitable FreeSurfer scan found for comparison.")
                        identical = False
                    else:
                        identical = compare_scans(
                            item / f"{name}-preop-T1w.nii",
                            scan_to_compare,
                            brain_mask_path=fs_dir / name / "mri" / "brainmask.mgz",
                            logger=logger,
                        )
                    # Skip if identical and not in reprocess list
                    if identical and name not in reprocess_list:
                        logger(f"{name}: scans identical, skipping.")
                        logger("")
                        continue

        # Create output subject directory
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy T1w file
        t1_files = list(item.rglob("*-T1w.nii"))
        if not t1_files:
            logger(f"{name}: no -T1w.nii file found, skipping.")
            logger("")
            continue

        if len(t1_files) > 1:
            logger(f"{name}: multiple T1w files found, using first: {t1_files[0]}")

        src = t1_files[0]
        dest_file = dest / f"{name}_T1w.nii"

        if dest_file.exists():
            logger(f"{name}: destination already exists, skipping copy.")
        else:
            try:
                shutil.copy2(src, dest_file)
                logger(f"Copied: {src} -> {dest_file}")
            except Exception as e:
                logger(f"Failed to copy {src} -> {dest_file}: {e}")

        # Add to subjects.txt list
        if name not in existing:
            to_add.append(name)
            existing.add(name)
        
        # Blank line
        logger("")

    if to_add:
        with subjects_file.open("a", encoding="utf-8") as f:
            for name in to_add:
                f.write(name + "\n")
        logger(f"Added {len(to_add)} subjects to {subjects_file}")
    else:
        logger("No new subjects to add.")

    logger("Processing complete.")
    close_log(log_f)

if __name__ == "__main__":
    # Define directories
    mri_dir = Path("L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/dataset_mri/pre_operative")
    fs_dir = Path("L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/dataset_fs")
    out_dir = Path("L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/dataset_for_fastsurfer")

    main(mri_dir, fs_dir, out_dir, reprocess_list=[
        'RESP0266', 
        'RESP0285',
        'RESP0577',
        'RESP0657',
        'RESP0779',
        'RESP0878',
        'RESP1227'
    ], only_reprocess=True)
