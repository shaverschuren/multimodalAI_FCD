import numpy as np
import nibabel as nib
from scipy.ndimage import label
import os
import re
import glob
import json
from tqdm import tqdm

# Hard-code MNI brain bounds in mm (from MNI152 template)
MNI_EXTENT_MM = np.array([90, 126, 72])  # x, y, z half-ranges


def _world_corner_bounds(shape, affine):
    """Return per-axis world-coordinate min/max from image corners."""
    d, h, w = shape
    corners = np.array([
        [0, 0, 0, 1],
        [d - 1, 0, 0, 1],
        [0, h - 1, 0, 1],
        [0, 0, w - 1, 1],
        [d - 1, h - 1, w - 1, 1],
    ], dtype=np.float64)
    world = (affine @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def validate_mni_grid_geometry(mask_path, shape, affine, tolerance_mm=5.0):
    """
    Ensure the image grid spans both negative and positive coordinates per axis.

    A legacy variant in this project uses a 256^3 affine with x in [90, 345],
    which produces unrealistically large MNI means (mu). Those files are rejected.
    """
    mins, maxs = _world_corner_bounds(shape, affine)

    # Require each axis to cross 0 in world/MNI space for centered grids.
    # This catches non-centered legacy affines (e.g. x=[90,345]).
    crosses_zero = (mins <= tolerance_mm) & (maxs >= -tolerance_mm)
    if not np.all(crosses_zero):
        raise ValueError(
            "Detected non-centered world grid; expected a centered MNI-like grid where "
            "each axis spans both negative and positive values. "
            f"File: {mask_path}; shape={tuple(shape)}; "
            f"world mins={mins.tolist()}, maxs={maxs.tolist()}, affine={affine.tolist()}"
        )

    return mins, maxs

def get_largest_connected_component(mask):
    labeled, num = label(mask)
    if num == 0:
        raise ValueError("Empty mask")

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label)

def get_mask_distribution_stats(mask_path):
    """
    Compute mean (μ) and per-axis std (σ) of the lesion in MNI space.

    Returns:
    --------
    dict with:
      - mu: (x, y, z) in mm
      - sigma: (σx, σy, σz) in mm
      - volume_ml
    """
    img = nib.load(mask_path)
    data = img.get_fdata()
    affine = img.affine
    shape = img.shape[:3]

    grid_mins, grid_maxs = validate_mni_grid_geometry(mask_path, shape, affine)

    mask = data > 0
    mask = get_largest_connected_component(mask)

    # voxel coordinates of lesion
    voxels = np.column_stack(np.nonzero(mask))

    # convert to MNI (mm)
    voxels_h = np.c_[voxels, np.ones(len(voxels))]
    mni_coords = (affine @ voxels_h.T).T[:, :3]

    # mean and std
    mu = mni_coords.mean(axis=0)
    sigma = mni_coords.std(axis=0)

    # Validate that mu is within expected MNI bounds (with 10mm tolerance)
    tolerance = 10
    if np.any(np.abs(mu) > MNI_EXTENT_MM + tolerance):
        raise ValueError(
            f"Computed mu {mu} is outside expected MNI bounds ±{MNI_EXTENT_MM}. "
            f"Check image registration and affine matrix."
        )

    # volume
    # Use column norms so this stays correct even if the affine contains rotation.
    voxel_dims = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume_mm3 = np.prod(voxel_dims)
    volume_ml = len(voxels) * voxel_volume_mm3 / 1000

    return {
        "mu": mu,
        "sigma": sigma,
        "volume_ml": volume_ml,
        "grid_world_min": grid_mins,
        "grid_world_max": grid_maxs,
    }

def process_gt_masks_to_json(root_folder, output_json_path, folder_pattern=r'.*'):
    """
    Process ground truth masks from subdirectories and save results to JSON.
    
    Parameters:
    -----------
    root_folder : str
        Path to the root folder containing subdirectories
    output_json_path : str
        Path where the JSON file will be saved
    folder_pattern : str
        Regular expression pattern to match subdirectories (default: match all)
    """
    results = {}
    
    # Compile the regex pattern
    pattern = re.compile(folder_pattern)
    
    # Get all subdirectories that match the pattern
    matching_dirs = []
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path) and pattern.match(item):
            matching_dirs.append(item)
    
    # Loop through matching directories
    for item in tqdm(matching_dirs, desc="Processing folders"):
        item_path = os.path.join(root_folder, item)
        # Search for *_gt_norm.nii.gz files
        gt_files = glob.glob(os.path.join(item_path, '*_gt_norm.nii.gz'))
        
        if gt_files:
            # Use the first matching file
            gt_file = gt_files[0]
            
            try:
                stats = get_mask_distribution_stats(gt_file)

                mu = stats["mu"]
                sigma = stats["sigma"]

                results[item] = {
                    "mni_mu_mm": {
                        "x": float(mu[0]),
                        "y": float(mu[1]),
                        "z": float(mu[2])
                    },
                    "mni_sigma_mm": {
                        "x": float(sigma[0]),
                        "y": float(sigma[1]),
                        "z": float(sigma[2])
                    },
                    "normalized_mu": {
                        "x": float(mu[0] / MNI_EXTENT_MM[0]),
                        "y": float(mu[1] / MNI_EXTENT_MM[1]),
                        "z": float(mu[2] / MNI_EXTENT_MM[2])
                    },
                    "normalized_sigma": {
                        "x": float(sigma[0] / MNI_EXTENT_MM[0]),
                        "y": float(sigma[1] / MNI_EXTENT_MM[1]),
                        "z": float(sigma[2] / MNI_EXTENT_MM[2])
                    },
                    "volume_ml": float(stats["volume_ml"]),
                    "gt_file": gt_file,
                    "normalization": {
                        "mni_extent_mm": MNI_EXTENT_MM.tolist(),
                        "grid_world_min": stats["grid_world_min"].tolist(),
                        "grid_world_max": stats["grid_world_max"].tolist(),
                    }
                }

            except Exception as e:
                tqdm.write(f"\033[91mError processing {item}: {str(e)}\033[0m")
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} patients. Results saved to {output_json_path}")
    return results


# CLI entry point
if __name__ == "__main__":

    # Hardcoded paths
    root_folder = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri"
    output_json_path = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\gt\\gt_coords.json"
    folder_pattern = r'^RESP\d{4}$'

    # Process
    process_gt_masks_to_json(root_folder, output_json_path, folder_pattern)
