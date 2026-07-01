import os
import json
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import affine_transform

def load_nifti(filepath):
    """Load NIfTI file and return data array."""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine

def create_channel_labels(ground_truth_path, sensitivity_map_dir, output_dir, patient_id, available_channels=None):
    """
    Create channel labels by multiplying ground truth with electrode sensitivity maps.
    
    Args:
        ground_truth_path: Path to ground truth .nii.gz file
        sensitivity_map_dir: Directory containing electrode sensitivity .nii.gz files
        output_dir: Directory to save output .json files
        patient_id: Patient identifier
        available_channels: Optional list of channels to include (default: all found)
    """
    # Load ground truth map
    gt_data, gt_affine = load_nifti(ground_truth_path)
    
    # Dictionary to store channel labels
    channel_labels = {}
    
    # Process each sensitivity map in the directory
    sensitivity_files = list(Path(sensitivity_map_dir).glob("*.nii.gz"))
    
    for sens_file in sensitivity_files:
        # Extract channel name from filename (remove .nii.gz extension)
        channel_name = sens_file.stem[20:].replace('.nii', '')

        # Skip if not in available channels
        if available_channels is not None:
            if channel_name.lower() not in [ch.lower() for ch in available_channels]:
                continue
        
        # Load sensitivity map
        sens_data, sens_affine = load_nifti(sens_file)
        
        # Resample ground truth to match sensitivity map affine and shape (only if needed, store result back in gt_data for next iterations)
        if not np.allclose(gt_affine, sens_affine) or gt_data.shape != sens_data.shape:
            # Resample ground truth to match sensitivity map
            transform_matrix = np.linalg.inv(sens_affine) @ gt_affine
            gt_data = affine_transform(gt_data, transform_matrix[:3, :3], 
                               offset=transform_matrix[:3, 3], 
                               output_shape=sens_data.shape)
            gt_affine = sens_affine

        # Ensure dimensions match
        if gt_data.shape != sens_data.shape:
            tqdm.write(f"\033[93mWarning: Shape mismatch for {channel_name}. GT: {gt_data.shape}, Sens: {sens_data.shape}\033[0m")
            continue
        
        # Multiply ground truth by sensitivity map
        overlap = gt_data * sens_data
        
        # Calculate sum of overlap and store
        overlap_sum = np.sum(overlap)
        channel_labels[channel_name] = float(overlap_sum)

    # Normalize to sum to 1
    total_sum = sum(channel_labels.values())
    if total_sum > 0:
        channel_labels = {ch: val / total_sum for ch, val in channel_labels.items()}
    # Remove channels with <1% contribution
    channel_labels = {ch: val for ch, val in channel_labels.items() if val >= 0.01}
    # Renormalize
    total_sum = sum(channel_labels.values())
    if total_sum > 0:
        channel_labels = {ch: val / total_sum for ch, val in channel_labels.items()}
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{patient_id}_channel_labels.json")
    
    with open(output_file, 'w') as f:
        json.dump(channel_labels, f, indent=2)
    
    tqdm.write(f"Channel labels saved to: {output_file}")
    if len(channel_labels) > 0:
        tqdm.write(f"Number of channels with overlap: {len(channel_labels)}")
    else:
        tqdm.write(f"\033[93mWarning: No channels with sufficient overlap found for patient {patient_id}.\033[0m")
    
    return channel_labels


def main(patients_dir, sensitivity_map_dir, output_dir):
    
    labels = {}

    # Loop over all patient directories
    for patient_folder in tqdm(list(patients_dir.iterdir()), desc="Processing patients"):
        if not patient_folder.is_dir():
            continue
        if not "RESP" in patient_folder.name:
            tqdm.write(f"Skipping non-RESP folder: {patient_folder.name}")
            continue
            
        patient_id = patient_folder.name
        tqdm.write(f"Processing patient: {patient_id}")
        
        # Expected file structure for each patient
        ground_truth_path = patient_folder / f"{patient_id}_gt_norm.nii.gz"
        
        # Check if required files exist
        if not ground_truth_path.exists():
            tqdm.write(f"\033[93mWarning: Ground truth file not found for {patient_id}: {ground_truth_path}\033[0m")
            continue
            
        if not sensitivity_map_dir.exists():
            tqdm.write(f"\033[93mWarning: Sensitivity maps directory not found: {sensitivity_map_dir}\033[0m")
            continue
        
        try:
            labels[patient_id] = create_channel_labels(
                ground_truth_path=str(ground_truth_path),
                sensitivity_map_dir=str(sensitivity_map_dir),
                output_dir=str(output_dir),
                patient_id=patient_id,
                available_channels=[
                "Fp1", "Fp2", "F9", "F10", "F7", "F3", "Fz", "F4", "F8",
                "T7", "C3", "Cz", "C4", "T8",
                "P7", "P3", "Pz", "P4", "P8",
                "O1", "O2"
                ]
            )
        except Exception as e:
            tqdm.write(f"\033[91mError processing patient {patient_id}: {e}\033[0m")

    return labels

if __name__ == "__main__":

    # Hard-code paths
    patients_dir = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri")
    sensitivity_map_dir = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\sensitivity_maps")
    output_dir = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\channel_labels")
    output_file = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\gt\\gt_channels.json")

    # Run main function
    labels = main(patients_dir, sensitivity_map_dir, output_dir)

    # Save all labels to a single JSON file
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)
    # Also copy to output directory for reference
    shutil.copyfile(output_file, output_dir / "gt_channels.json")