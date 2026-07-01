"""
preprocess_gt.py

Usage (single patient only):

  python preprocess_gt.py --patient-id SUBJ01 \
    --mri-mask /path/to/SUBJ01_postresection.nii.gz \
    --pic-mask /path/to/SUBJ01_pic2mri.nii.gz \
    --manual-mask /path/to/SUBJ01_manual_mask.nii.gz \
    --atlas /path/to/atlas_labels.nii.gz \
    --atlas-lut /path/to/FreeSurferColorLUT.txt \
    --outdir /path/to/outdir \

Notes:
  - All masks are expected to be (or will be resampled to) atlas/FreeSurfer native T1 space.
  - Not all masks need to be provided; the script will choose the best available mask in order:
    1. post-resection MRI mask
    2. pic2mri mask
    3. manual mask
  - atlas_lut (optional) can be FreeSurferColorLUT.txt.

Author: Sjors Verschuren
Date: November 2025
"""
import os
import sys
import json
import argparse
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

try:
    from nibabel.processing import resample_from_to
    HAVE_RESAMPLE_FROM_TO = True
except Exception:
    HAVE_RESAMPLE_FROM_TO = False

# -------------------------
# I/O helpers
# -------------------------
def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img

def save_nifti(data: np.ndarray, affine: np.ndarray, outpath: str) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, outpath)

def save_npz_mask(mask_bool: np.ndarray, affine: np.ndarray, outpath: str) -> None:
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_vol = float(np.prod(voxel_sizes))
    np.savez_compressed(
        outpath,
        mask=mask_bool.astype(np.uint8),
        affine=affine,
        voxel_volume_mm3=voxel_vol
    )

# -------------------------
# Resampling
# -------------------------
def resample_to_target(src_img: nib.Nifti1Image, target_img: nib.Nifti1Image, order: int = 0) -> nib.Nifti1Image:
    if HAVE_RESAMPLE_FROM_TO:
        return resample_from_to(src_img, target_img, order=order)
    src_data = src_img.get_fdata()
    tgt_shape = target_img.shape
    factors = tuple(np.array(tgt_shape) / np.array(src_data.shape))
    resampled = ndimage.zoom(src_data, factors, order=order)
    return nib.Nifti1Image(resampled, target_img.affine)

# -------------------------
# Mask choice logic
# -------------------------
def choose_ground_truth(mri_mask_path: Optional[str], pic_mask_path: Optional[str], manual_mask_path: Optional[str]) -> Tuple[Optional[str], str]:
    if mri_mask_path and os.path.exists(mri_mask_path):
        return mri_mask_path, "mri_mask"
    if pic_mask_path and os.path.exists(pic_mask_path):
        return pic_mask_path, "pic2mri"
    if manual_mask_path and os.path.exists(manual_mask_path):
        return manual_mask_path, "manual_mask"
    return None, "none"

# -------------------------
# Metrics
# -------------------------
def binarize_mask(data: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (data > thr).astype(np.uint8)

def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom

def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union

def precision(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return tp / (tp + fp + 1e-8)

def recall(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp / (tp + fn + 1e-8)

def fbeta_score(gt: np.ndarray, pred: np.ndarray, beta: float = 1.0) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision_v = tp / (tp + fp + 1e-8)
    recall_v = tp / (tp + fn + 1e-8)
    
    if precision_v == 0 and recall_v == 0:
        return 0.0

    beta2 = beta ** 2
    return (1 + beta2) * (precision_v * recall_v) / (beta2 * precision_v + recall_v + 1e-8)

def relative_volume_difference(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    v_gt = gt.sum()
    v_pred = pred.sum()
    if v_gt == 0:
        return np.nan  # undefined if no lesion in GT
    return (v_pred - v_gt) / (v_gt + 1e-8)

def volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_vol_mm3 = np.prod(voxel_sizes)
    return mask.sum() * voxel_vol_mm3 / 1000.0

def surface_voxel_coordinates(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    eroded = ndimage.binary_erosion(mask, structure=struct)
    surface = mask.astype(bool) & (~eroded)
    idx = np.array(np.nonzero(surface)).T
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)
    hom = np.concatenate([idx, np.ones((idx.shape[0], 1))], axis=1)
    coords = (affine @ hom.T).T[:, :3]
    return coords

def hausdorff_distance_mm(a: np.ndarray, b: np.ndarray, affine: np.ndarray) -> float:
    ca = surface_voxel_coordinates(a, affine)
    cb = surface_voxel_coordinates(b, affine)
    if ca.shape[0] == 0 and cb.shape[0] == 0:
        return 0.0
    if ca.shape[0] == 0 or cb.shape[0] == 0:
        return float('inf')
    d_ab = cdist(ca, cb)
    d_ba = cdist(cb, ca)
    return float(max(d_ab.min(axis=1).max(), d_ba.min(axis=1).max()))

def assd_mm(a: np.ndarray, b: np.ndarray, affine: np.ndarray) -> float:
    ca = surface_voxel_coordinates(a, affine)
    cb = surface_voxel_coordinates(b, affine)
    if ca.shape[0] == 0 and cb.shape[0] == 0:
        return 0.0
    if ca.shape[0] == 0 or cb.shape[0] == 0:
        return float('inf')
    d_ab = cdist(ca, cb)
    d_ba = cdist(cb, ca)
    return float(0.5 * (d_ab.min(axis=1).mean() + d_ba.min(axis=1).mean()))

# -------------------------
# Atlas labeling
# -------------------------
def atlas_vector(atlas_block: dict, level: str):
    """
    Convert atlas_labeling['region_counts'] into a normalized vector at the chosen level:
    level ∈ {'hemisphere', 'lobe', 'gyrus'}.
    Returns: (labels, vector)
    """
    if ("region_counts" not in atlas_block) or len(atlas_block["region_counts"]) == 0:
        return [], np.array([])

    counts = defaultdict(float)

    for item in atlas_block["region_counts"]:
        key = item.get(level, None)
        if not key:   # empty string or None
            key = "unknown"
        counts[key] += item["count"]

    labels = sorted(counts.keys())
    vec = np.array([counts[label] for label in labels], dtype=float)

    # Normalize (proportion of lesion volume)
    s = vec.sum()
    if s > 0:
        vec /= s

    return labels, vec

def atlas_similarity_levels(gt_atlas: dict, pic_atlas: dict):
    """
    Compute similarity at hemisphere, lobe, and gyrus levels,
    excluding all white-matter / unlabeled structures
    (i.e., any region where lobe == "").

    Returns a dict with 3 scalar metrics.
    """

    # Apply filtering
    gt_clean = filter_nonwm(gt_atlas)
    pic_clean = filter_nonwm(pic_atlas)

    results = {}

    for level in ["hemisphere", "lobe", "gyrus"]:
        # Build vectors
        gt_labels, gt_vec = atlas_vector(gt_clean, level)
        pic_labels, pic_vec = atlas_vector(pic_clean, level)

        # Build joint label set
        all_labels = sorted(set(gt_labels) | set(pic_labels))

        # Align vectors to same label order
        def align(vec_labels, vec, all_labels):
            mapping = {l: i for i, l in enumerate(vec_labels)}
            out = np.zeros(len(all_labels), dtype=float)
            for i, label in enumerate(all_labels):
                if label in mapping:
                    out[i] = vec[mapping[label]]
            return out

        gt_aligned = align(gt_labels, gt_vec, all_labels)
        pic_aligned = align(pic_labels, pic_vec, all_labels)

        # Check if top regions are the same
        gt_top = gt_clean.get("top_region", {})
        pic_top = pic_clean.get("top_region", {})

        if gt_top and pic_top:
            # Compare by label_id for exact match
            results[f"atlas_top_region_same_{level}"] = float(gt_top.get(level) == pic_top.get(level))
        else:
            results[f"atlas_top_region_same_{level}"] = 0.0

        # Compute cosine similarity and Pearson correlation
        results[f"atlas_cos_sim_{level}"] = cosine_sim(gt_aligned, pic_aligned)
        if len(gt_aligned) == 1:
            if np.all(gt_aligned == pic_aligned):
                results[f"atlas_pearsonr_{level}"] = 1.0
            else:
                results[f"atlas_pearsonr_{level}"] = 0.0
        else:
            results[f"atlas_pearsonr_{level}"] = pearsonr(gt_aligned, pic_aligned)[0]

    return results

def filter_nonwm(atlas_dict):
    """Remove entries where lobe == '' (white matter, ventricles, etc.)."""
    filtered = {
        **atlas_dict,
        "region_counts": [
            rc for rc in atlas_dict.get("region_counts", [])
            if rc.get("lobe", "") not in ("", None)
        ]
    }
    return filtered

def cosine_sim(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length for cosine similarity")
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return np.nan
    return float(np.dot(vec1, vec2) / denom)

def infer_hemisphere_from_name(name: str) -> Optional[str]:
    n = name.lower()
    if any(n.startswith(p) for p in ['lh', 'left', 'l-', 'l_']) or '-lh-' in n or '_lh_' in n:
        return 'left'
    if any(n.startswith(p) for p in ['rh', 'right', 'r-', 'r_']) or '-rh-' in n or '_rh_' in n:
        return 'right'
    if 'left' in n: return 'left'
    if 'right' in n: return 'right'
    return None

def infer_lobe_from_name(name: str) -> Optional[str]:
    n = name.lower()

    # Frontal lobe
    if any(k in n for k in [
        "front", "precentral", "orbital", "rectus", "suborbital",
        "frontomargin", "transv_frontopol"
    ]):
        return "frontal"

    # Parietal lobe
    if any(k in n for k in [
        "pariet", "postcentral", "precuneus", "supramar", "angular", "subparietal"
    ]):
        return "parietal"

    # Temporal lobe
    if any(k in n for k in [
        "temp", "temporal", "fusifor", "parahip", "transverse", "plan_"
    ]):
        return "temporal"

    # Occipital lobe
    if any(k in n for k in [
        "occip", "cuneus", "calcarine", "lingual", "parieto_occipital"
    ]):
        return "occipital"

    # Cingulate cortex
    if "cingul" in n:
        return "cingulate"

    # Insula
    if "insul" in n or "insula" in n:
        return "insula"

    # Medial wall / subcallosal / special regions
    if any(k in n for k in ["medial_wall", "subcallosal", "pericallosal"]):
        return "medial"

    return None

def infer_gyrus_from_name(name: str) -> Optional[str]:
    # TODO: Check if common gyral names are necessary
    return name

def load_atlas_lut(lut_path: Optional[str]) -> Dict[int, Dict[str, str]]:
    if lut_path is None:
        lut_path = os.path.join(os.environ.get('FREESURFER_HOME', ''), 'FreeSurferColorLUT.txt')
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT file not found at {lut_path}")

    mapping = {}
    with open(lut_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                label_id = int(parts[0])
                label_name = parts[1]
                mapping[label_id] = {
                    'name': label_name,
                    'hemisphere': infer_hemisphere_from_name(label_name) or '',
                    'lobe': infer_lobe_from_name(label_name) or '',
                    'gyrus': infer_gyrus_from_name(label_name) or ''
                }
    return mapping

def label_from_overlap(mask: np.ndarray, atlas_data: np.ndarray, lut: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    mask_bool = mask.astype(bool)
    atlas_labels, counts = np.unique(atlas_data[mask_bool], return_counts=True)
    sel = atlas_labels != 0
    atlas_labels, counts = atlas_labels[sel], counts[sel]
    breakdown = []
    for lab, ct in zip(atlas_labels, counts):
        info = lut.get(int(lab), {})
        breakdown.append({
            'label_id': int(lab),
            'label_name': info.get('name', ''),
            'count': int(ct),
            'hemisphere': info.get('hemisphere', ''),
            'lobe': info.get('lobe', ''),
            'gyrus': info.get('gyrus', '')
        })
    breakdown_sorted = sorted(breakdown, key=lambda x: -x['count'])

    # Find top region
    if breakdown_sorted:
        # Find top region that has a lobe assignment (no WM as this is not parcellated and thus big)
        top_region = None
        for region in breakdown_sorted:
            if region.get('lobe') and region['lobe'] != '':
                top_region = region
                break
        if top_region is None:
            # Fallback to first region if no lobe-assigned regions found
            top_region = breakdown_sorted[0]
    else:
        return None

    # Result dict
    result = {'region_counts': breakdown_sorted, 'top_region': top_region}
    return result

# -------------------------
# Main processing
# -------------------------
def process_patient(patient_id, mri_mask, pic_mask, manual_mask, atlas, atlas_lut, outdir, logger=print):
    os.makedirs(outdir, exist_ok=True)
    chosen_path, reason = choose_ground_truth(mri_mask, pic_mask, manual_mask)
    result = {'patient_id': patient_id, 'chosen_mask_reason': reason}
    logger(f"[{patient_id}] Chosen mask: {chosen_path} (reason: {reason})")

    atlas_data, atlas_affine, atlas_img = load_nifti(atlas)
    atlas_data = atlas_data.astype(int)
    lut = load_atlas_lut(atlas_lut)

    if chosen_path is None:
        result['message'] = 'No ground truth mask available'
        return result

    mask_img = nib.load(chosen_path)
    need_resample = (mask_img.shape != atlas_img.shape) or (not np.allclose(mask_img.affine, atlas_img.affine, atol=1e-5))
    if need_resample:
        logger(f"[{patient_id}] Resampling chosen mask to atlas space.")
        mask_img = resample_to_target(mask_img, atlas_img, order=0)

    logger(f"[{patient_id}] Binarizing and saving ground truth mask.")
    mask_bin = binarize_mask(mask_img.get_fdata())
    out_mask_nii = os.path.join(outdir, f"{patient_id}_gt_mask.nii.gz")
    # out_mask_npz = os.path.join(outdir, f"{patient_id}_gt_mask.npz")
    save_nifti(mask_bin, atlas_affine, out_mask_nii)
    # save_npz_mask(mask_bin, atlas_affine, out_mask_npz)
    result.update({
        'chosen_mask_path': chosen_path,
        'written_nifti': out_mask_nii,
        # 'written_npz': out_mask_npz,
        'mask_voxels': int(mask_bin.sum()),
        'mask_volume_ml': volume_ml(mask_bin, atlas_affine)
    })

    logger(f"[{patient_id}] Performing atlas labeling of the chosen ground-truth lesion.")
    result['atlas_labeling_gt'] = label_from_overlap(mask_bin, atlas_data, lut)

    # If both post-op MRI and pic2mri masks exist, compute comparison metrics (unharmonised)
    if mri_mask and os.path.exists(mri_mask) and pic_mask and os.path.exists(pic_mask):
        logger(f"[{patient_id}] Both MRI and pic2mri masks are available. Computing comparison metrics.")

        # Load the other image (pic2mri)
        pic_img = nib.load(pic_mask)
        if (pic_img.shape != atlas_img.shape) or (not np.allclose(pic_img.affine, atlas_img.affine, atol=1e-5)):
            pic_img = resample_to_target(pic_img, atlas_img, order=0)

        logger(f"[{patient_id}] Computing atlas labels for second mask.")
        # --- Compute atlas labeling for the pic2mri mask ---
        pic_bin = binarize_mask(pic_img.get_fdata())
        pic_labeling = label_from_overlap(pic_bin, atlas_data, lut)
        result['atlas_labeling_pic2mri'] = pic_labeling
        # Also save to JSON for convenience
        pic_json = os.path.join(outdir, f"{patient_id}_atlas_labeling_pic2mri.json")
        with open(pic_json, "w") as f:
            json.dump(pic_labeling, f, indent=2)
        result['atlas_labeling_pic2mri_path'] = pic_json

        # Compute comparison metrics
        comp = {
            'A -> B': f"A: {pic_mask}, B: {chosen_path}",
            'dice': dice_coef(mask_bin, pic_bin),
            'jaccard': jaccard_index(mask_bin, pic_bin),
            'F1 (B=GT)': fbeta_score(mask_bin, pic_bin, beta=1.0),
            'precision (B=GT)': precision(mask_bin, pic_bin),
            'recall (B=GT)': recall(mask_bin, pic_bin),
            'rVD (B=GT)': relative_volume_difference(mask_bin, pic_bin),
            'hausdorff_mm': hausdorff_distance_mm(mask_bin, pic_bin, atlas_affine),
            'assd_mm': assd_mm(mask_bin, pic_bin, atlas_affine)
        }
        # Atlas similarity levels
        level_sims = atlas_similarity_levels(
            result["atlas_labeling_gt"],
            result["atlas_labeling_pic2mri"]
        )
        comp.update(level_sims)

        # Save comparison metrics
        result['comparison'] = comp
        comp_path = os.path.join(outdir, f"{patient_id}_comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(comp, f, indent=2)
        result['comparison_report_path'] = comp_path
        logger(f"[{patient_id}] Comparison metrics saved to {comp_path}")

    # Create smoothed probability map for pic2mri masks (for harmonized thresholding later)
    # Convert sigma from mm to voxels
    if pic_mask and os.path.exists(pic_mask):
        # If only pic2mri is available, use gt mask as pic_raw
        if reason == "pic2mri":
            pic_bin = mask_bin
        # Else, use pic2mri from comparison
        else:
            pass

        voxel_sizes = np.sqrt((atlas_affine[:3, :3] ** 2).sum(axis=0))
        sigma_voxels = 3.0 / voxel_sizes  # 3mm in voxel units
        # Apply filter
        pic_smooth = gaussian_filter(pic_bin.astype(float), sigma=sigma_voxels)

        # Save smoothed probability map (used later for global thresholding)
        smooth_out_path = os.path.join(outdir, f"{patient_id}_pic2mri_smooth.nii.gz")
        save_nifti(pic_smooth, atlas_affine, smooth_out_path)
        result['pic2mri_smooth_path'] = smooth_out_path
        result['pic2mri_smooth_mean'] = float(np.nanmean(pic_smooth))

    outjson = os.path.join(outdir, f"{patient_id}_processing_report.json")
    with open(outjson, 'w') as f:
        json.dump(result, f, indent=2)
    logger(f"[{patient_id}] Processing complete. Report written to {outjson}")
    return result

# Main entry point
def main():
    p = argparse.ArgumentParser(description="Process single-patient lesion ground truth and compute atlas labels.")
    p.add_argument('--patient-id', type=str, required=True, help='Patient ID')
    p.add_argument('--mri-mask', type=str, help='Path to post-resection MRI mask')
    p.add_argument('--pic-mask', type=str, help='Path to pic2mri mask')
    p.add_argument('--manual-mask', type=str, help='Path to manual mask')
    p.add_argument('--atlas', type=str, required=True, help='Atlas labels NIfTI')
    p.add_argument('--atlas-lut', type=str, default=None, help='Path to FreeSurferColorLUT.txt')
    p.add_argument('--outdir', type=str, required=True, help='Output directory')
    args = p.parse_args()

    process_patient(args.patient_id, args.mri_mask, args.pic_mask, args.manual_mask, args.atlas, args.atlas_lut, args.outdir)

if __name__ == '__main__':
    main()
