"""
create_MNI_sensitivity_fields.py
Create MNI sensitivity fields for EEG electrodes based on their spatial distribution.

This script generates sensitivity maps for EEG electrodes positioned according to the MNI standard brain template.
Uses EEG electrode positions projected on the MNI brain, as published by Koessler et al., 2009.
We model the electrode sensitivity as a Gaussian kernel centered at each electrode position, which might be an oversimplification
but this is a try-out for now.

Author: Sjors Verschuren
Date: December 2025
"""

import os
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from datasets.eeg import CHANNEL_LABEL_TO_INT, validate_electrode_coordinates

GAUSSIAN_SIGMA_MM = 30.0


def load_mni_brain(template_path):
    """
    Load the MNI standard brain in .nii.gz format.
    """
    img = nib.load(template_path)
    data = img.get_fdata()
    spatial_unit, _ = img.header.get_xyzt_units()
    _, sform_code = img.header.get_sform(coded=True)
    if spatial_unit != "mm" or int(sform_code) != 4:
        raise ValueError(
            "Expected an MNI152 template with millimetre units and an MNI sform "
            f"(units={spatial_unit!r}, sform_code={int(sform_code)})."
        )
    return data, img.affine, img.header


def create_sensitivity_map(electrode_coords, mni_brain, affine, sigma_mm=GAUSSIAN_SIGMA_MM):
    """
    Create a sensitivity map for each electrode based on its spatial distribution.
    The electrode sensitivity is modeled as a Gaussian kernel.
    
    Parameters:
    - electrode_coords: Channel names mapped to x/y/z scalp coordinates in millimetres
    - mni_brain: The MNI brain template data (numpy array)
    - affine: The affine transformation for the MNI brain
    - sigma_mm: Isotropic Gaussian standard deviation (in mm)
    
    Returns:
    - sensitivity_maps: A list of sensitivity maps (one per electrode)
    """
    sensitivity_maps = {}
    inverse_affine = np.linalg.inv(affine)
    
    for electrode in tqdm(electrode_coords.keys(), desc="Creating sensitivity maps"):
        # Convert electrode coordinate to voxel space
        world_coord = [electrode_coords[electrode][axis] for axis in "xyz"]
        voxel_coord = nib.affines.apply_affine(inverse_affine, world_coord)
        
        # Create a 3D Gaussian kernel centered at the electrode's voxel position
        sensitivity_map = np.zeros(mni_brain.shape)
        # Place a point source at the electrode position
        voxel_coord_int = np.round(voxel_coord).astype(int)
        if (0 <= voxel_coord_int[0] < sensitivity_map.shape[0] and 
            0 <= voxel_coord_int[1] < sensitivity_map.shape[1] and 
            0 <= voxel_coord_int[2] < sensitivity_map.shape[2]):
            sensitivity_map[voxel_coord_int[0], voxel_coord_int[1], voxel_coord_int[2]] = 1.0
        # Apply Gaussian smoothing to create the sensitivity field
        voxel_sizes = nib.affines.voxel_sizes(affine)
        sensitivity_map = gaussian_filter(
            sensitivity_map,
            sigma=sigma_mm / voxel_sizes,
            mode='constant', cval=0, truncate=4.0,
        )
        sensitivity_map = sensitivity_map * (mni_brain > 0)
        maximum = np.max(sensitivity_map)
        if maximum <= 0:
            raise ValueError(f"Sensitivity field for {electrode} does not intersect the MNI brain.")
        sensitivity_map /= maximum

        # Add the sensitivity map to the overall brain map
        sensitivity_maps[electrode] = sensitivity_map

    return sensitivity_maps

def save_sensitivity_map(sensitivity_map, affine, output_filename):
    """
    Save a single sensitivity map as a .nii.gz file.
    """
    # Create NIfTI image
    img = nib.Nifti1Image(sensitivity_map, affine=affine)
    nib.save(img, output_filename)

def main():
    eeg_dir = Path(__file__).resolve().parent
    mni_template_path = eeg_dir / 'MNI152_T1_1mm_brain.nii.gz'
    electrode_data_json = eeg_dir / 'standard_1005_mni_coordinates.json'
    output_dir = PROJECT_DIR.parent / 'data' / 'preprocessing' / 'eeg' / 'sensitivity_maps'

    # Load electrode coordinates from JSON file
    with open(electrode_data_json, 'r', encoding='utf-8') as f:
        electrode_data = json.load(f)

    source = electrode_data.get('source', {})
    coordinate_unit = source.get('unit')
    if coordinate_unit != 'mm':
        raise ValueError(f"Expected scalp coordinates in millimetres, got {coordinate_unit!r}.")
    coordinate_system = source.get('coordinate_system', 'unspecified')
    if 'mni' not in coordinate_system.casefold():
        warnings.warn(
            f"Coordinate metadata declares {coordinate_system!r}, not MNI. "
            "Geometric checks will be applied, but confirm this frame is registered to the template before using the maps.",
            UserWarning,
            stacklevel=2,
        )

    mni_brain, affine, header = load_mni_brain(mni_template_path)
    electrode_coords = {
        channel: electrode_data['data'][channel]['mni']
        for channel in sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
    }
    validate_electrode_coordinates(electrode_coords, mni_brain, affine, header)

    # Create and save sensitivity maps
    sensitivity_maps = create_sensitivity_map(
        electrode_coords, mni_brain, affine, sigma_mm=GAUSSIAN_SIGMA_MM,
    )
    
    for electrode, sensitivity_map in tqdm(sensitivity_maps.items(), desc="Saving sensitivity maps"):
        output_filename = os.path.join(output_dir, f'sensitivity_map_MNI_{electrode}.nii.gz')
        save_sensitivity_map(sensitivity_map, affine, output_filename)
        tqdm.write(f"Saved sensitivity map for channel {electrode} to {output_filename}")

if __name__ == "__main__":
    main()
