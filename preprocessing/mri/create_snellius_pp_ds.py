import os
import shutil
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

def copy_patient_data(post_op_dir, pre_op_dir, gt_dir, out_dir, overwrite=False):
    """
    Copy .nii files from post_operative subfolders and contents from 
    corresponding pre_operative subfolders to output directory.
    """
    post_op_path = Path(post_op_dir)
    pre_op_path = Path(pre_op_dir)
    out_path = Path(out_dir)
    gt_path = Path(gt_dir)

    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Iterate through subfolders in pre-operative directory
    for patient_folder in tqdm(list(pre_op_path.iterdir()), desc="Processing patients"):
        if patient_folder.is_dir():
            patient_id = patient_folder.name
            if patient_id not in ["RESP0230", "RESP0507", "RESP0083", "RESP0286", "RESP1149"]:  # TODO: remove this line later
                continue
            tqdm.write(f"Processing patient: {patient_id}")

            # Create corresponding output subfolder
            out_patient_dir = out_path / patient_id
            out_patient_dir.mkdir(parents=True, exist_ok=True)

            # Copy .nii/.nii.gz files from pre_operative subfolder
            for file in patient_folder.iterdir():
                if file.suffix in ['.nii', '.nii.gz']:
                    compressed_name = file.stem + '.nii.gz' if file.suffix == '.nii' else file.name
                    if overwrite or not ((out_patient_dir / file.name).exists() or (out_patient_dir / compressed_name).exists()):
                        shutil.copy2(file, out_patient_dir)

            # Copy contents from corresponding post_operative subfolder
            post_op_patient_dir = post_op_path / patient_id
            if post_op_patient_dir.exists() and post_op_patient_dir.is_dir():
                for item in post_op_patient_dir.iterdir():
                    if item.suffix in ['.nii', '.nii.gz']:
                        compressed_name = item.stem + '.nii.gz' if item.suffix == '.nii' else item.name
                        if overwrite or not ((out_patient_dir / item.name).exists() or (out_patient_dir / compressed_name).exists()):
                            shutil.copy2(item, out_patient_dir)

            # Copy ground truth mask if it exists
            gt_file = gt_path / patient_id / f"{patient_id}_gt_mask.nii.gz"
            if gt_file.exists():
                compressed_name = gt_file.stem + '.nii.gz' if gt_file.suffix == '.nii' else gt_file.name
                if overwrite or not ((out_patient_dir / gt_file.name).exists() or (out_patient_dir / compressed_name).exists()):
                    shutil.copy2(gt_file, out_patient_dir)

            # Compress output files to .nii.gz if they are .nii
            for file in out_patient_dir.iterdir():
                if file.suffix == '.nii':
                    img = nib.load(file)
                    nib.save(img, file.with_suffix('.nii.gz'))
                    file.unlink()  # Remove the original .nii file

# Example usage
if __name__ == "__main__":
    post_operative_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_mri\\post_operative"
    pre_operative_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_mri\\pre_operative"
    gt_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\gt"
    output_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ds_for_snellius_preprocess"
    
    copy_patient_data(post_operative_dir, pre_operative_dir, gt_dir, output_dir, overwrite=False)