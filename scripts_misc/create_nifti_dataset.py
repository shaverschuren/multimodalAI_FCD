import os
import shutil
import pandas as pd
from glob import glob
from dicom2nifti import convert_directory
from dicom2nifti import settings
from tqdm import tqdm  

def convert_dicom_to_nifti(dicom_dir, output_file):
    """Convert a DICOM directory to a NIfTI file."""
    try:
        os.mkdir(os.path.join(os.path.dirname(output_file), "tmp"))
        convert_directory(dicom_dir, os.path.join(os.path.dirname(output_file), "tmp"), compression=False, reorient=True)
        nii_files = glob(os.path.join(os.path.dirname(output_file), 'tmp', '*.nii'))
        if nii_files:
            shutil.move(nii_files[0], output_file)
        else:
            print(f"Warning: Standard conversion failed for {dicom_dir}. Retrying with slice increment validation disabled.")
            settings.disable_validate_slice_increment()
            convert_directory(dicom_dir, os.path.join(os.path.dirname(output_file), "tmp"), compression=False, reorient=True)
            settings.enable_validate_slice_increment()
            nii_files = glob(os.path.join(os.path.dirname(output_file), 'tmp', '*.nii'))
            if nii_files:
                shutil.move(nii_files[0], output_file)
            else:
                raise RuntimeError("No NIfTI file generated after retry.")
        shutil.rmtree(os.path.join(os.path.dirname(output_file), "tmp"))
        return f"{dicom_dir}: Done"
    except Exception as e:
        shutil.rmtree(os.path.join(os.path.dirname(output_file), "tmp"))
        return f"Conversion failed for {dicom_dir}: {e}"

def process_csv(csv_path, dicom_root, output_root, scan_type):
    """Process csv file with selected DICOM paths to convert raw DICOM scans to NIfTI format."""

    # Read csv and store in dataframe
    df = pd.read_csv(csv_path, sep=';')
    print(f"Processing {len(df)} scans from {csv_path} into {output_root}...")

    # Loop over csv rows
    logs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{scan_type.capitalize()} scans"):
        # Define paths and filenames
        dicom_dir = os.path.join(dicom_root, row['Type'], row['session'], str(row['scan']), 'DICOM')

        # Create subject directory if it doesn't exist
        subject_dir = os.path.join(output_root, row['patient_id'])
        os.makedirs(subject_dir, exist_ok=True)

        # Define output NIfTI filename
        if scan_type == 'preop':
            out_file = os.path.join(subject_dir, f"{row['patient_id']}-preop-{row['Type']}.nii")
        else:
            out_file = os.path.join(subject_dir, f"{row['patient_id']}-postop-{row['Type']}.nii")

        # Convert DICOM to NifTI if not already done
        if not os.path.exists(out_file):
            logs.append(convert_dicom_to_nifti(dicom_dir, out_file))
        else:
            logs.append(f"File already exists: {out_file}")

    # Output logs to file
    with open(os.path.join(output_root, f"conversion_log.txt"), 'w') as log_file:
        for log in logs:
            if log:
                log_file.write(log + '\n')
    print(f"Logs saved to {os.path.join(output_root, f'conversion_log.txt')}")

if __name__ == "__main__":

    # Define paths
    pre_op_csv_path = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull\\pre_op_scans_manual_select_final.csv'
    post_op_csv_path = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull\\post_op_scans_manual_select_final.csv'
    dicom_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull\\raw'
    output_preop = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_mri\\pre_operative'
    output_postop = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_mri\\post_operative'

    # Process pre-op scans
    process_csv(pre_op_csv_path, dicom_root, output_preop, scan_type='preop')

    # Process post-op scans
    process_csv(post_op_csv_path, dicom_root, output_postop, scan_type='postop')
