"""
Full MRI preprocessing pipeline for pre-op T1w+FLAIR, post-op T1w and GT masks.

Builds upon standard tools: ANTs, FSL, HD-BET.
Assumes these are all installed and available in PATH.

Note: Runs much quicker on GPU for HD-BET, but CPU is also fine.

Includes:
- N4 bias field correction
- Rigid registration of FLAIR (and postop if provided) to T1w
- Skull stripping with HD-BET
- Affine registration to MNI
- Apply transforms to FLAIR/postop and GT mask (if provided)
- Resampling to 1mm isotropic
- Intensity normalization (robust z-score)
- Cropping/padding to uniform shape
- Storage in both Nifti and compressed npz
- QC montage generation

Author: Sjors Verschuren
Date: November 2025
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch import cuda  # type: ignore
from scipy.ndimage import zoom, center_of_mass
import time

# -------------------------------
# --- CONFIGURATION ---
# -------------------------------

TARGET_SPACING = (1.0, 1.0, 1.0)                   # In mm isotropic
TARGET_SHAPE   = (160, 192, 160)                   # final crop/pad shape

# Detect device for HD-BET
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
# MNI template path (sets automatically if FSLDIR is set)
FSL_DIR = os.environ.get('FSLDIR', '/home/bin/fsl')
MNI_TEMPLATE   = os.path.join(FSL_DIR, "data/standard/MNI152_T1_1mm_brain.nii.gz")

# -------------------------------
# --- HELPER FUNCTIONS ---
# -------------------------------

def run(cmd):
    """Run a shell command safely and print stdout/stderr on error."""
    print(f"[run] {cmd}")
    start_time = time.time()
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f}s")
    if proc.returncode != 0:
        print("Command failed. stdout:")
        print(proc.stdout.decode('utf-8', errors='replace') if proc.stdout else "Empty")
        print("stderr:")
        print(proc.stderr.decode('utf-8', errors='replace') if proc.stderr else "Empty")
        raise RuntimeError(f"Command failed with return code {proc.returncode}")
    return proc

def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(dtype=np.float32), nii.affine

def save_nii(data, affine, path):
    nib.Nifti1Image(data, affine).to_filename(path)

def intensity_norm(img, mask):
    # Compute median and MAD within brain mask
    vals = img[mask > 0]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-8
    std_rob = 1.4826 * mad

    # Clip extremes: [-6*std, +6*std] around median
    lo, hi = med - 6 * std_rob, med + 6 * std_rob
    img = np.clip(img, lo, hi)

    # Rescale to [0,1]
    img_norm = (img - lo) / (hi - lo)

    return img_norm.astype(np.float32), {"median": float(med), "std_rob": float(std_rob)}

def resample_to_spacing(img, affine, target_spacing, order=1):
    """
    Resample image data to target_spacing using scipy.zoom.
    order: interpolation order (1=linear for intensities, 0=nearest for masks)
    Returns resampled image and a new affine with diagonal target spacing.
    """
    target_spacing = np.asarray(target_spacing, dtype=float)

    # current voxel axis size (mm) per column of affine
    R = affine[:3, :3].copy()
    current_spacing = np.sqrt((R ** 2).sum(axis=0))  # length of each column

    # zoom factors (how many voxels along each axis after resampling)
    zoom_factors = current_spacing / target_spacing
    # perform zoom on numpy array: scipy.ndimage.zoom expects zoom factors per axis in input order
    img_res = zoom(img, zoom_factors, order=order)

    # Build new affine that preserves axis directions/rotation but uses new spacing
    # Unit direction vectors for the original columns:
    with np.errstate(divide='ignore', invalid='ignore'):
        dirs = R / current_spacing  # shape (3,3), dividing columns by scalar -> column unit vectors

    # Some safety: if any current_spacing==0, fallback to identity unit vectors
    for i in range(3):
        if not np.isfinite(current_spacing[i]) or current_spacing[i] == 0:
            dirs[:, i] = np.eye(3)[:, i]

    new_R = dirs * target_spacing  # scale unit direction columns by new voxel sizes
    new_affine = affine.copy()
    new_affine[:3, :3] = new_R

    # Adjust translation so that world coordinates of the image center remain (approximately) the same.
    old_shape = np.array(img.shape)
    new_shape = np.array(img_res.shape)
    old_center_vox = (old_shape - 1) / 2.0
    new_center_vox = (new_shape - 1) / 2.0

    old_center_world = affine[:3, :3] @ old_center_vox + affine[:3, 3]
    new_affine[:3, 3] = old_center_world - new_affine[:3, :3] @ new_center_vox

    return img_res, new_affine

def crop_or_pad(img, target_shape, affine):
    """
    Crop/pad that preserves world (or MNI) coordinates.

    Parameters
    ----------
    img : ndarray (DxHxW)
    target_shape : tuple of 3 ints
    affine : (4,4) ndarray

    Returns
    -------
    img_out : ndarray
        Cropped/padded image with shape = target_shape
    affine_out : ndarray
        Updated affine mapping voxel->world/MNI for the output image
    """

    img_shape = np.array(img.shape)
    target_shape = np.array(target_shape)

    diff = target_shape - img_shape

    # Amount to pad/crop on the "before" side
    before = np.floor(diff / 2).astype(int)

    # Split into pad_before ≥0 and crop_before ≥0
    pad_before  = np.clip(before, 0, None)
    crop_before = np.clip(-before, 0, None)

    # Amount to pad after cropping
    pad_after = np.clip(diff - pad_before, 0, None)

    # ----------------------------------------------------
    # 2. Crop (if needed)
    # ----------------------------------------------------
    slices = tuple(
        slice(c, c + min(t, img_shape[i] - c))
        for i, (c, t) in enumerate(zip(crop_before, target_shape))
    )
    cropped = img[slices]

    # ----------------------------------------------------
    # 3. Pad (if needed)
    # ----------------------------------------------------
    pad_width = [(pad_before[i], pad_after[i]) for i in range(3)]
    final = np.pad(cropped, pad_width, mode='constant')

    # ----------------------------------------------------
    # 4. UPDATE AFFINE
    # ----------------------------------------------------

    # Compute original and new centers in voxel coordinates
    old_center_vox = (img_shape - 1) / 2
    new_center_vox = (target_shape - 1) / 2

    # Convert to world coordinates
    R = affine[:3, :3]
    t = affine[:3, 3]

    old_center_world = R @ old_center_vox + t
    new_center_world_should_be = old_center_world

    # Solve for new translation t_new:
    t_new = old_center_world - R @ new_center_vox

    new_affine = affine.copy()
    new_affine[:3, 3] = t_new

    return final, new_affine

def qc_image(subj_id, t1_aff, t1_n4, flair_inT1, mask, mni_template, t1_mni, flair_mni, t1_final,
             flair_final, gt_inT1=None, t1_postop_mni=None, gt_mni=None, gt_final=None, out_path="qc.png"):
    """
    Generates a QC figure:
    - Top: N4 + brain mask outline (left) and N4 + GT overlay (right) side by side
    - Bottom: Four full-width transverse mosaics with GT overlay (except MNI template)
    """

    fig = plt.figure(figsize=(16, 16), facecolor='black')
    outer_gs = gridspec.GridSpec(9, 2, height_ratios=[1.3,0.1,0.3,0.3,0.3,0.3,0.1,0.5,0.5], width_ratios=[1,1], wspace=0.05, hspace=0.05)

    # ---------------------
    # Helper to extract slices with correct aspect ratio for anisotropic voxels
    def get_slices(img, affine=None, com=None):
        shape = np.array(img.shape)
        if com is None:
            com = shape // 2

        slices = [
            img[int(com[0]), :, :],  # axial: slice along first dimension (i, j, k) → (j, k)
            img[:, int(com[1]), :],  # sagittal: slice along second dimension → (i, k)
            img[:, :, int(com[2])]   # coronal: slice along third dimension → (i, j)
        ]
        
        if affine is not None:
            # Get voxel spacing directly from affine diagonal (assumes near-diagonal affine after registration)
            spacing = np.abs(np.diag(affine[:3, :3]))
            # Compute aspect ratios for each slice view
            # axial (dim 0 sliced): shows dims 1 and 2, aspect = spacing[2] / spacing[1]
            # sagittal (dim 1 sliced): shows dims 0 and 2, aspect = spacing[2] / spacing[0]
            # coronal (dim 2 sliced): shows dims 0 and 1, aspect = spacing[1] / spacing[0]
            aspect_axial = spacing[2] / spacing[1]
            aspect_sag = spacing[2] / spacing[0]
            aspect_cor = spacing[1] / spacing[0]
            aspects = [aspect_axial, aspect_sag, aspect_cor]
        else:
            aspects = [1.0, 1.0, 1.0]

        return slices, aspects

    # ---------------------
    # Compute global min/max for T1w and FLAIR for consistent windowing
    t1_vmin, t1_vmax = np.percentile(t1_n4[t1_n4 > 0], [1, 99])
    flair_vmin, flair_vmax = np.percentile(flair_inT1[flair_inT1 > 0], [1, 99])
    
    # ---------------------
    # Top-left grid: N4 + brain mask
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0, 0], wspace=0.0, hspace=0.0)
    mid = np.array(t1_n4.shape)//2
    for row, img in enumerate([t1_n4, flair_inT1]):
        vmin, vmax = (t1_vmin, t1_vmax) if row == 0 else (flair_vmin, flair_vmax)
        slices_img, aspects_img = get_slices(img, t1_aff, mid)
        slices_mask, aspects_mask = get_slices(mask, t1_aff, mid)
        for col, (slc, msk) in enumerate(zip(slices_img, slices_mask)):
            ax = fig.add_subplot(inner_gs[row, col], facecolor='black')
            ax.imshow(slc.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect=aspects_img[col])
            ax.contour(msk.T, colors='skyblue', linewidths=0.5)
            ax.axis('off')
            # Add label to leftmost plot only
            if col == 0:
                ax.text(-0.1, 0.5, ["T1w", "FLAIR"][row], transform=ax.transAxes,
                        fontsize=12, color='white', weight='bold', va='center', ha='right')

    # Top-right grid: N4 + GT overlay
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0, 1], wspace=0.0, hspace=0.0)
    if gt_inT1 is not None:
        # Empty GT masks yield NaN COM; fall back to center slices for robust QC.
        if np.any(gt_inT1 > 0):
            com_gt = np.array(center_of_mass(gt_inT1))
        else:
            com_gt = np.array(t1_n4.shape) // 2
        for row, img in enumerate([t1_n4, flair_inT1]):
            vmin, vmax = (t1_vmin, t1_vmax) if row == 0 else (flair_vmin, flair_vmax)
            slices_img, aspects_img = get_slices(img, affine=t1_aff, com=com_gt)
            slices_mask, aspects_mask = get_slices(mask, affine=t1_aff, com=com_gt)
            slices_gt, aspects_gt = get_slices(gt_inT1, affine=t1_aff, com=com_gt)
            for col, (slc, msk, ovl) in enumerate(zip(slices_img, slices_mask, slices_gt)):
                ax = fig.add_subplot(inner_gs[row, col], facecolor='black')
                ax.imshow(slc.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect=aspects_img[col])
                ax.contour(msk.T, colors='skyblue', linewidths=0.5)
                ax.contour(ovl.T, colors='red', linewidths=0.5)
                ax.axis('off')

    # ---------------------
    # 4 full-width rows: transverse MNI slices
    mni_imgs = [
        ("Template", mni_template, False),       # no GT overlay
        ("T1 preop", t1_mni, True),
        ("FLAIR preop", flair_mni, True),
        ("T1 postop", t1_postop_mni, True)    # optional
    ]

    row_counter = 2  # top row is 0, gap is 1, start mni rows at 2
    for title, img, overlay_gt in mni_imgs:
        if img is None:
            ax = fig.add_subplot(outer_gs[row_counter, :], facecolor='black')
            ax.text(0.5, 0.5, f"----- No {title} available -----", transform=ax.transAxes,
                    fontsize=10, color='white', weight='bold', va='center', ha='center')
        else:
            ax = fig.add_subplot(outer_gs[row_counter, :], facecolor='black')
            # Sample 16 slices evenly in z avoiding edges
            z_slices = np.linspace(20, img.shape[2]-20, 16).astype(int)
            mosaic = np.hstack([img[:, :, z].T for z in z_slices])
            ax.imshow(mosaic, cmap="gray", origin="lower")

            # Overlay GT if requested and provided
            if overlay_gt and gt_mni is not None:
                gt_mosaic = np.hstack([gt_mni[:, :, z].T for z in z_slices])
                ax.contour(gt_mosaic, colors="red", linewidths=0.3)

        if row_counter == 2:
            ax.set_title("MNI alignment", fontsize=14, color='white', weight='bold')

        ax.text(-0.01, 0.5, title, transform=ax.transAxes,
                fontsize=12, color='white', weight='bold', va='center', ha='right')

        ax.axis("off")
        row_counter += 1

    # ---------------------
    # Final two rows: transverse preprocessed final images with GT overlay
    row_counter += 1  # skip one row for spacing

    final_imgs = [
        ("T1w", t1_final, gt_final),
        ("FLAIR", flair_final, gt_final)
    ]

    for title, img, gt in final_imgs:
        ax = fig.add_subplot(outer_gs[row_counter, :], facecolor='black')
        z_slices = np.linspace(0, img.shape[2]-1, 12).astype(int)
        mosaic = np.hstack([img[:, :, z].T for z in z_slices])
        ax.imshow(mosaic, cmap="gray", origin="lower")

        # Overlay GT if provided
        if gt is not None:
            gt_mosaic = np.hstack([gt[:, :, z].T for z in z_slices])
            ax.contour(gt_mosaic, colors="red", linewidths=0.3)

        if title == "T1w":
            ax.set_title("Final preprocessed", fontsize=14, color='white', weight='bold')
        ax.text(-0.01, 0.5, title, transform=ax.transAxes,
                fontsize=12, color='white', weight='bold', va='center', ha='right')
        ax.axis("off")
        row_counter += 1

    fig.suptitle(f"{subj_id}: MRI Preprocessing QC", fontsize=16, color='white')

    plt.savefig(out_path, dpi=150, facecolor='black')
    plt.close(fig)

# -------------------------------
# --- MAIN PIPELINE ---
# -------------------------------

def preprocess_subject(
    subj_id, t1_path, flair_path, out_dir, gt_path=None, postop_path=None,
    keep_intermediates=False, overwrite=False, aggressive_flair_n4=False
):

    print(f"[{subj_id}] Start preprocessing...")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, subj_id)
    t1_n4 = base + "_T1w_N4.nii.gz"
    flair_n4 = base + "_FLAIR_N4.nii.gz"

    # 1. N4 bias correction
    print(f"[{subj_id}, 1/10] Running N4 bias field correction...")
    if overwrite or not (os.path.exists(t1_n4) and os.path.exists(flair_n4)):
        # T1w N4
        run(f"N4BiasFieldCorrection -d 3 -i {t1_path} -o {t1_n4} -v 1")
        # FLAIR N4 (aggressive optional)
        # --> Useful for some experimental FLAIRs in the dataset with strong bias fields.
        if aggressive_flair_n4:
            print(f"[{subj_id}]   Running two-stage aggressive N4 for FLAIR...")
            flair_stage1 = flair_n4.replace("_N4", "_N4stage1")
            flair_stage2 = flair_n4  # final output
            # Stage 1: coarse, large-scale inhomogeneity
            run(
                f"N4BiasFieldCorrection -d 3 "
                f"-i {flair_path} "
                f"-o {flair_stage1} "
                f"-s 4 "
                f"-b [200,3] "
                f"-c [200x100x50,1e-6] "
                f"-v 1"
            )
            # Stage 2: medium-scale residual bias
            run(
                f"N4BiasFieldCorrection -d 3 "
                f"-i {flair_stage1} "
                f"-o {flair_stage2} "
                f"-s 2 "
                f"-b [50,3] "
                f"-c [100x50x20,1e-6] "
                f"-v 1"
            )
        else:
            # default single-pass N4
            run(f"N4BiasFieldCorrection -d 3 -i {flair_path} -o {flair_n4} -v 1")
    else:
        print(f"  Skipping N4 correction, output files already exist.")

    # 2. Rigid registration (FLAIR -> T1)
    print(f"[{subj_id}, 2/10] Running rigid registration (FLAIR -> T1)...")
    reg_prefix_FLAIR2T1 = base + "_reg_FLAIRtoT1_"
    flair_inT1 = base + "_FLAIR_inT1.nii.gz"
    if overwrite or not os.path.exists(flair_inT1):
        run(
            f"antsRegistration "
            f" -d 3 "
            f" -r [{t1_n4},{flair_n4},1] "
            f" -m MI[{t1_n4},{flair_n4},1,32] "
            f" -t Rigid[0.1] "
            f" -c [1000x500x250x0,1e-6,10] "
            f" -s 3x2x1x0vox "
            f" -f 8x4x2x1 "
            f" -o [{reg_prefix_FLAIR2T1},{base}_FLAIR_inT1.nii.gz] "
            f" -n Linear "
            f" -v 1"
        )
        # Apply rigid affine (0GenericAffine.mat produced by reg_prefix)
        run(f"antsApplyTransforms -d 3 -i {flair_n4} -r {t1_n4} "
            f"-o {flair_inT1} -t {reg_prefix_FLAIR2T1}0GenericAffine.mat -n Linear")
    else:
        print(f"  Skipping FLAIR->T1 registration, output file already exists.")

    # 2b. If postop scan provided, also register to T1 (rigid)
    if postop_path is not None:
        print(f"[{subj_id}, 2b/10] Running rigid registration (postop -> T1)...")
        reg_prefix_postop2T1 = base + "_reg_postopToT1_"
        postop_inT1 = base + "_postop_inT1.nii.gz"
        if overwrite or not os.path.exists(postop_inT1):
            run(
                f"antsRegistration "
                f" -d 3 "
                f" -r [{t1_n4},{postop_path},1] "
                f" -m MI[{t1_n4},{postop_path},1,32] "
                f" -t Rigid[0.1] "
                f" -c [1000x500x250x0,1e-6,10] "
                f" -s 3x2x1x0vox "
                f" -f 8x4x2x1 "
                f" -o [{reg_prefix_postop2T1},{base}_postop_inT1.nii.gz] "
                f" -n Linear"
                f" -v 1"
            )
            run(f"antsApplyTransforms -d 3 -i {postop_path} -r {t1_n4} "
                f"-o {postop_inT1} -t {reg_prefix_postop2T1}0GenericAffine.mat -n Linear")
        else:
            print(f"  Skipping postop->T1 registration, output file already exists.")

    # 2c. If GT provided, resample to T1 space
    if gt_path is not None:
        print(f"[{subj_id}, 2c/10] Resampling GT mask to T1 space...")
        gt_inT1 = base + "_gt_inT1.nii.gz"
        if overwrite or not os.path.exists(gt_inT1):
            run(f"antsApplyTransforms -d 3 -i {gt_path} -r {t1_n4} "
                f"-o {gt_inT1} -n NearestNeighbor")
        else:
            print(f"  Skipping GT resampling, output file already exists.")

    # 3. Skull strip (HD-BET) on T1
    print(f"[{subj_id}, 3/10] Running skull stripping with HD-BET...")
    mask_path = base + "_T1w_brain_bet.nii.gz"
    t1_brain = base + "_T1w_brain.nii.gz"
    if overwrite or not (os.path.exists(mask_path) and os.path.exists(t1_brain)):
        run(f"hd-bet -i {t1_n4} -o {base}_T1w_brain.nii.gz -device {DEVICE} --save_bet_mask --verbose")
    else:
        print(f"  Skipping skull stripping, output files already exist.")

    # 4. Apply mask to FLAIR (T1-space)
    print(f"[{subj_id}, 4/10] Applying brain mask to FLAIR (and postop if provided)...")
    flair_brain = base + "_FLAIR_brain.nii.gz"
    if overwrite or not os.path.exists(flair_brain):
        run(f"fslmaths {flair_inT1} -mul {mask_path} {flair_brain}")
    else:
        print(f"  Skipping FLAIR masking, output file already exists.")
    # 4b. Also apply to postop if provided
    if postop_path is not None:
        postop_brain = base + "_postop_brain.nii.gz"
        if overwrite or not os.path.exists(postop_brain):
            run(f"fslmaths {postop_inT1} -mul {mask_path} {postop_brain}")
        else:
            print(f"  Skipping postop masking, output file already exists.")

    # 5. Spatial normalisation (Rigid + affine registration T1 -> MNI)
    print(f"[{subj_id}, 5/10] Running rigid + affine registration to MNI space...")
    norm_prefix = base + "_toMNI_"
    t1_mni = base + "_T1w_MNI.nii.gz"
    flair_mni = base + "_FLAIR_MNI.nii.gz"
    mask_mni = base + "_mask_MNI.nii.gz"
    gt_mni = base + "_gt_MNI.nii.gz" if gt_path is not None else None
    postop_mni = base + "_postop_MNI.nii.gz" if postop_path is not None else None
    if overwrite or not os.path.exists(t1_mni) or not os.path.exists(flair_mni) \
        or not os.path.exists(mask_mni) or (gt_path is not None and not os.path.exists(gt_mni)) \
            or (postop_path is not None and not os.path.exists(postop_mni)):
        run(
            f"antsRegistration "
            f"-d 3 "
            f"-r [{MNI_TEMPLATE},{t1_brain},1] "
            f"--winsorize-image-intensities [0.005,0.995] "
            # Rigid stage
            f"-m MI[{MNI_TEMPLATE},{t1_brain},1,32] "
            f"-t Rigid[0.1] "
            f"-c [1000x500x250x0,1e-6,10] "
            f"-s 3x2x1x0vox "
            f"-f 8x4x2x1 "
            # Affine stage
            f"-m MI[{MNI_TEMPLATE},{t1_brain},1,32] "
            f"-t Affine[0.1] "
            f"-c [1000x500x250x0,1e-6,10] "
            f"-s 3x2x1x0vox "
            f"-f 8x4x2x1 "
            f"-o [{norm_prefix},{base}_T1w_MNI.nii.gz] "
            f"-n Linear "
            f"-v 1"
        )
        
        # Apply transforms to all images
        # Get affine file
        affine_file = norm_prefix + "0GenericAffine.mat"
        # Format transforms for command: "-t warp -t affine"
        transforms_cmd = f"-t {affine_file}"
        # Apply transforms to T1, FLAIR, mask (linear interp for images, nearest for masks)
        run(f"antsApplyTransforms -d 3 -i {t1_brain} -r {MNI_TEMPLATE} -o {t1_mni} {transforms_cmd} -n Linear")
        run(f"antsApplyTransforms -d 3 -i {flair_brain} -r {MNI_TEMPLATE} -o {flair_mni} {transforms_cmd} -n Linear")
        run(f"antsApplyTransforms -d 3 -i {mask_path} -r {MNI_TEMPLATE} -o {mask_mni} {transforms_cmd} -n NearestNeighbor")
        # Apply same transforms to ground-truth if provided (nearest neighbor)
        if gt_path is not None:
            run(f"antsApplyTransforms -d 3 -i {gt_path} -r {MNI_TEMPLATE} -o {gt_mni} {transforms_cmd} -n NearestNeighbor")
        # Also to postop if provided (linear)
        if postop_path is not None:
            run(f"antsApplyTransforms -d 3 -i {postop_brain} -r {MNI_TEMPLATE} -o {postop_mni} {transforms_cmd} -n Linear")
    else:
        print(f"  Skipping MNI registration, output files already exist.")

    # 6. Load and resample to TARGET_SPACING isotropic (if needed)
    print(f"[{subj_id}, 6/10] Resampling to {TARGET_SPACING}mm isotropic...")
    t1_img, aff_mni = load_nii(t1_mni)
    flair_img, _ = load_nii(flair_mni)
    mask_img, _ = load_nii(mask_mni)
    if gt_path is not None:
        gt_img, _ = load_nii(gt_mni)
    else:
        gt_img = None
    if postop_path is not None:
        postop_img, _ = load_nii(postop_mni)
    else:
        postop_img = None

    # Resample intensities (linear) and masks (nearest)
    t1_iso, aff_res = resample_to_spacing(t1_img, aff_mni, TARGET_SPACING, order=1)
    flair_iso, _ = resample_to_spacing(flair_img, aff_mni, TARGET_SPACING, order=1)
    mask_iso, _ = resample_to_spacing(mask_img, aff_mni, TARGET_SPACING, order=0)
    mask_iso = (mask_iso > 0.5).astype(np.uint8)
    if gt_img is not None:
        gt_iso, _ = resample_to_spacing(gt_img, aff_mni, TARGET_SPACING, order=0)
        gt_iso = (gt_iso > 0.5).astype(np.uint8)
    else:
        gt_iso = None
    if postop_img is not None:
        postop_iso, _ = resample_to_spacing(postop_img, aff_mni, TARGET_SPACING, order=1)
    else:
        postop_iso = None

    # 7. Crop/pad to uniform shape
    print(f"[{subj_id}, 7/10] Cropping/padding to target shape {TARGET_SHAPE}...")
    t1_norm, aff_final = crop_or_pad(t1_iso, TARGET_SHAPE, affine=aff_res)
    flair_norm, _ = crop_or_pad(flair_iso, TARGET_SHAPE, affine=aff_res)
    postop_norm, _ = crop_or_pad(postop_iso, TARGET_SHAPE, affine=aff_res) if postop_iso is not None else (None, None)

    mask_final, _ = crop_or_pad(mask_iso, TARGET_SHAPE, affine=aff_res)
    gt_final, _ = crop_or_pad(gt_iso, TARGET_SHAPE, affine=aff_res) if gt_iso is not None else (None, None)

    # 8. Intensity normalization (robust [0-1] using brain mask)
    print(f"[{subj_id}, 8/10] Performing intensity normalization...")
    t1_final, t1_stats = intensity_norm(t1_norm, mask_final)
    flair_final, flair_stats = intensity_norm(flair_norm, mask_final)
    postop_final, postop_stats = intensity_norm(postop_norm, mask_final) if postop_norm is not None else (None, None)

    # 9. Save npz (include gt if present) and final nifti files
    print(f"[{subj_id}, 9/10] Saving preprocessed data to .npz...")
    out_npz = base + "_preproc.npz"
    save_dict = {
        "image": np.stack([t1_final, flair_final], axis=0).astype(np.float32),
        "mask": mask_final.astype(np.uint8),
        "affine": aff_final,
        "norm_params": {"T1": t1_stats, "FLAIR": flair_stats}
    }
    if gt_final is not None:
        save_dict["gt"] = gt_final.astype(np.uint8)
    if postop_final is not None:
        save_dict["postop"] = postop_final.astype(np.float32)
        save_dict["norm_params"]["postop"] = postop_stats

    np.savez_compressed(out_npz, **save_dict)

    # 9b. Save final preprocessed images as NIfTI
    print(f"[{subj_id}, 9b/10] Saving final final preprocessed NIfTI files...")
    save_nii(t1_final.astype(np.float32), aff_final, base + "_T1w_norm.nii.gz")
    save_nii(flair_final.astype(np.float32), aff_final, base + "_FLAIR_norm.nii.gz")
    save_nii(mask_final.astype(np.uint8), aff_final, base + "_brainmask_norm.nii.gz")
    if gt_final is not None:
        save_nii(gt_final.astype(np.uint8), aff_final, base + "_gt_norm.nii.gz")
    if postop_final is not None:
        save_nii(postop_final.astype(np.float32), aff_final, base + "_postop_norm.nii.gz")
    # 10. QC
    print(f"[{subj_id}, 10/10] Generating QC image...")
    qc_path = base + "_qc.png"
    qc_image(
        subj_id=subj_id,
        t1_aff=load_nii(t1_n4)[1],              # affine of native T1w
        t1_n4=load_nii(t1_n4)[0],               # N4-corrected T1 in native T1w space
        flair_inT1=load_nii(flair_inT1)[0],     # N4-corrected FLAIR in T1w space
        mask=load_nii(mask_path)[0],            # brain mask (aligned to T1w space)
        mni_template=load_nii(MNI_TEMPLATE)[0], # MNI template
        t1_mni=t1_img,                          # MNI-registered T1
        flair_mni=flair_img,                    # MNI-registered FLAIR
        t1_postop_mni=postop_img if postop_path is not None else None,
        t1_final=t1_final,
        flair_final=flair_final,
        gt_inT1=load_nii(gt_inT1)[0] if gt_path is not None else None,
        gt_mni=gt_img if gt_path is not None else None,
        gt_final=gt_final if gt_final is not None else None,
        out_path=qc_path
    )

    # Cleanup intermediate files
    if not keep_intermediates:
        # List intermediates
        intermediates = [t1_n4, flair_n4, flair_n4.replace("_N4", "_N4stage1"), t1_brain, flair_brain, t1_mni, flair_mni, mask_mni]
        if gt_mni is not None: intermediates.append(gt_mni)
        if postop_mni is not None: intermediates.append(postop_mni)
        # List of suffixes to remove
        sufs = ["0GenericAffine.mat", "Warped.nii.gz", "1Warp.nii.gz", "1InverseWarp.nii.gz"]
        # Iteratively check and remove files
        for f in os.listdir(out_dir):
            full_path = os.path.join(out_dir, f)
            if f in [os.path.basename(path) for path in intermediates] or any(f.endswith(suf) for suf in sufs):
                try:
                    if os.path.exists(full_path): os.remove(full_path)
                except Exception as e:
                    print(f"Warning: could not remove {f}: {e}")

    print(f"[{subj_id}] Done: {subj_id} → {out_npz}\n")

if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Full MRI preprocessing pipeline for T1w + FLAIR to MNI space"
    )
    # Define arguments
    parser.add_argument("subj_id", type=str, help="Subject identifier")
    parser.add_argument("t1_path", type=str, help="Path to T1-weighted NIfTI file")
    parser.add_argument("flair_path", type=str, help="Path to FLAIR NIfTI file")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--gt_path", type=str, default=None,
        help="Optional ground-truth mask NIfTI file"
    )
    parser.add_argument(
        "--postop_path", type=str, default=None,
        help="Optional postoperative scan NIfTI file"
    )
    parser.add_argument(
        "--keep_intermediates", action="store_true", default=False,
        help="Whether to keep intermediate files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Whether to overwrite existing outputs/intermediates"
    )
    parser.add_argument(
        "--aggressive_flair_n4",
        action="store_true",
        help="Use aggressive N4 correction for the FLAIR image (useful for severe bias fields in some experimental FLAIRs)."
    )
    args = parser.parse_args()

    # Run preprocessing
    preprocess_subject(
        subj_id=args.subj_id,
        t1_path=args.t1_path,
        flair_path=args.flair_path,
        out_dir=args.out_dir,
        gt_path=args.gt_path,
        postop_path=args.postop_path,
        keep_intermediates=args.keep_intermediates,
        aggressive_flair_n4=args.aggressive_flair_n4
    )
