import os
import shutil
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

def convert_bonn_ds(input_dir, output_dir, include_hc=False, only_hc=False):
    """
    Convert BIDS compatible database to preprocessing pipeline format.
    Args:
        input_dir: Path to the BIDS dataset directory (e.g., ds004199)
        output_dir: Path to the output directory where converted data will be stored
        include_hc: If True, include healthy control subjects (those without '_roi' files)
        only_hc: If True, only process healthy control subjects (those without '_roi' files)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    
    patient_count = 1
    
    for subject_dir in tqdm(sorted(subject_dirs), desc="Processing subjects"):
        anat_dir = subject_dir / 'anat'
        
        if not anat_dir.exists():
            continue
        
        # Find files in anat directory
        nii_files = list(anat_dir.glob('*.nii.gz'))
        
        # Check if ground truth exists
        gt_files = [f for f in nii_files if '_roi' in f.name]

        if gt_files and only_hc:
            continue  # Skip subjects with ground truth if only_hc is True

        if not gt_files:
            if not include_hc:
                continue  # Skip subjects without ground truth if include_hc is False
            else:
                gt_files = []  # No GT files for this subject, but we will include it as HC
        
        # Find T1w and FLAIR scans
        t1w_files = [f for f in nii_files if 'T1w' in f.name and '_roi' not in f.name]
        flair_files = [f for f in nii_files if 'FLAIR' in f.name and '_roi' not in f.name]
        
        # Create new subject directory name
        new_subject_name = subject_dir.name.replace('sub-', 'Bonn')
        new_subject_dir = output_path / new_subject_name
        new_subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy T1w files
        for t1w_file in t1w_files:
            new_name = f"{new_subject_name}-preop-T1w.nii.gz"
            shutil.copy2(t1w_file, new_subject_dir / new_name)
            tqdm.write(f"Copied {t1w_file.name} -> {new_name}")
        
        # Copy FLAIR files
        for flair_file in flair_files:
            new_name = f"{new_subject_name}-preop-FLAIR.nii.gz"
            shutil.copy2(flair_file, new_subject_dir / new_name)
            tqdm.write(f"Copied {flair_file.name} -> {new_name}")
        
        # Copy or generate empty ground truth file
        if gt_files:
            for gt_file in gt_files:
                new_name = f"{new_subject_name}_gt_mask.nii.gz"
                shutil.copy2(gt_file, new_subject_dir / new_name)
                tqdm.write(f"Copied {gt_file.name} -> {new_name}")
        else:
            # Create an empty ground truth file (same shape as T1w or FLAIR)
            reference_file = t1w_files[0] if t1w_files else (flair_files[0] if flair_files else None)
            if reference_file:
                new_name = f"{new_subject_name}_gt_mask.nii.gz"
                shutil.copy2(reference_file, new_subject_dir / new_name)  # Copy reference file
                # Now overwrite the copied file with zeros
                img = nib.load(new_subject_dir / new_name)
                data = img.get_fdata()
                data[:] = 0  # Set all values to zero
                nib.save(nib.Nifti1Image(data, img.affine), new_subject_dir / new_name)
                tqdm.write(f"Created empty ground truth for {new_subject_name}")
        
        patient_count += 1
    
    print(f"Processed {patient_count - 1} patients with ground truth data")

if __name__ == "__main__":
    input_directory = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\mri\openneuro\ds004199")
    output_directory = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\ds_for_snellius_preprocess")
    
    convert_bonn_ds(input_directory, output_directory, include_hc=False, only_hc=False)