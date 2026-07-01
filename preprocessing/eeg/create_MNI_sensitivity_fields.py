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
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import json
from tqdm import tqdm

def load_mni_brain(template_path):
    """
    Load the MNI standard brain in .nii.gz format.
    """
    img = nib.load(template_path)
    data = img.get_fdata()
    return data, img.affine

def create_sensitivity_map(electrode_coords, mni_brain, affine, kernel_size=10):
    """
    Create a sensitivity map for each electrode based on its spatial distribution.
    The electrode sensitivity is modeled as a Gaussian kernel.
    
    Parameters:
    - electrode_coords: A dictionary with electrode names as keys and MNI coordinates under 'mni' key
    - mni_brain: The MNI brain template data (numpy array)
    - affine: The affine transformation for the MNI brain
    - kernel_size: Standard deviation of the Gaussian kernel (in mm)
    
    Returns:
    - sensitivity_maps: A list of sensitivity maps (one per electrode)
    """
    sensitivity_maps = {}
    
    for electrode in tqdm(electrode_coords.keys(), desc="Creating sensitivity maps"):
        # Convert electrode coordinate to voxel space
        voxel_coord = np.dot(np.linalg.inv(affine), np.append(list(electrode_coords[electrode]['mni'].values()), 1))[:3]
        
        # Create a 3D Gaussian kernel centered at the electrode's voxel position
        sensitivity_map = np.zeros(mni_brain.shape)
        # Place a point source at the electrode position
        voxel_coord_int = np.round(voxel_coord).astype(int)
        if (0 <= voxel_coord_int[0] < sensitivity_map.shape[0] and 
            0 <= voxel_coord_int[1] < sensitivity_map.shape[1] and 
            0 <= voxel_coord_int[2] < sensitivity_map.shape[2]):
            sensitivity_map[voxel_coord_int[0], voxel_coord_int[1], voxel_coord_int[2]] = 1.0
        # Apply Gaussian smoothing to create the sensitivity field
        sensitivity_map = gaussian_filter(sensitivity_map, sigma=kernel_size, mode='constant', cval=0, truncate=4.0)
        # Renormalize the sensitivity map
        sensitivity_map /= np.max(sensitivity_map)

        # Mask the sensitivity map with the MNI brain template (only include non-zero brain voxels)
        sensitivity_map = sensitivity_map * (mni_brain > 0)

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
    # Define path to MNI-125 standard brain template
    mni_template_path = os.path.join('preprocessing', 'eeg', 'MNI152_T1_1mm_brain.nii.gz')
    electrode_data_json = os.path.join('preprocessing', 'eeg', '10_10_to_MNI.json')
    output_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\sensitivity_maps"

    # Load electrode coordinates from JSON file
    with open(electrode_data_json, 'r') as f:
        electrode_data = json.load(f)

    # Load MNI brain template
    mni_brain, affine = load_mni_brain(mni_template_path)

    # Create and save sensitivity maps
    sensitivity_maps = create_sensitivity_map(electrode_data['data'], mni_brain, affine, kernel_size=10)
    
    for electrode, sensitivity_map in tqdm(sensitivity_maps.items(), desc="Saving sensitivity maps"):
        output_filename = os.path.join(output_dir, f'sensitivity_map_MNI_{electrode}.nii.gz')
        save_sensitivity_map(sensitivity_map, affine, output_filename)
        tqdm.write(f"Saved sensitivity map for channel {electrode} to {output_filename}")

if __name__ == "__main__":
    main()
