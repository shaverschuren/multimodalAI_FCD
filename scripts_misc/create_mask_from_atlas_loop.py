import os
import subprocess
import yaml
from PIL import Image
import nibabel as nib
import numpy as np
import shutil

# ------------------------
# FreeSurfer label LUT parsing
# ------------------------
def load_freesurfer_lut(lut_path):
    """
    Load FreeSurferColorLUT.txt and return a dict mapping label names to numbers.
    """
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    name_to_number = {}
    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                number = int(parts[0])
                name = parts[1]
                name_to_number[name] = number
    return name_to_number

# ------------------------
# Mask creation function
# ------------------------
def create_masks_from_aparc(aparc_path, wmseg_path, labels_list, output_dir):
    """
    Create binary masks from a FreeSurfer aparc+aseg.mgz file.
    """
    if not os.path.exists(aparc_path):
        raise FileNotFoundError(f"Aparc file not found: {aparc_path}")
    if not os.path.exists(wmseg_path):
        raise FileNotFoundError(f"WM segmentation file not found: {wmseg_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {aparc_path} ...")
    img_aparc = nib.load(aparc_path)
    data_aparc = img_aparc.get_fdata().astype(np.int32)

    print(f"Loading {wmseg_path} ...")
    img_wmseg = nib.load(wmseg_path)
    data_wmseg = img_wmseg.get_fdata().astype(np.int32)

    mask = np.zeros(data_aparc.shape, dtype=np.uint8)

    for lbl in labels_list:
        print(f"Creating mask for label {lbl} ...")
        mask_lbl_aparc = (data_aparc == lbl).astype(np.uint8)
        mask_lbl_wmseg = (data_wmseg == lbl).astype(np.uint8)

        if not np.any(mask_lbl_aparc) and not np.any(mask_lbl_wmseg):
            print(f"Warning: label {lbl} not found in aparc or wmseg data.")
            continue

        mask += mask_lbl_aparc + mask_lbl_wmseg
        mask = np.clip(mask, 0, 1)

    mask_path = os.path.join(output_dir, f"resection_mask_atlas_based.nii.gz")
    nib.save(nib.Nifti1Image(mask, img_aparc.affine), mask_path)
    print(f"Saved mask to {mask_path}")

# ------------------------
# Helper functions
# ------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def show_resection_photo(photo_path):
    if os.path.exists(photo_path):
        img = Image.open(photo_path)
        img.show(title='Resection Photo')
    else:
        print(f'Resection photo not found: {photo_path}')

def open_slicer_gui(slicer_path, subject_dir):
    """
    Open Slicer in the subject directory.
    User can manually select T1.mgz, aparc+aseg.mgz, wmparc.mgz.
    Not loading automatically to use SlicerFreeSurfer extension. 
    """
    print("Opening Slicer for manual inspection...")
    subprocess.Popen([slicer_path], cwd=os.path.join(subject_dir, 'atlas_based_output'))
    input("Check which labels to use and press Enter to continue...")

def parse_labels_input(labels_input, name_to_number):
    """
    Convert user input (label names, comma separated) to list of numeric FreeSurfer labels.
    """
    labels_list = []
    for name in labels_input.split(','):
        name = name.strip()
        if name in name_to_number:
            labels_list.append(name_to_number[name])
        else:
            print(f"Warning: label name '{name}' not found in LUT. Skipping.")
    return labels_list

# ------------------------
# Patient processing
# ------------------------
def process_subject(subject_dir, subject_id, slicer_path, pic_data_dir, lut_dict):
    atlas_file = os.path.join(subject_dir, 'pic2mri_output', 'atlas_based.txt')
    if not os.path.exists(atlas_file):
        return

    t1_file = os.path.join(subject_dir, 'mri', 'T1.mgz')
    aparc_file = os.path.join(subject_dir, 'mri', 'aparc.a2009s+aseg.mgz')
    wmseg_file = os.path.join(subject_dir, 'mri', 'wmparc.mgz')
    missing = [f for f in [t1_file, aparc_file, wmseg_file] if not os.path.exists(f)]
    if missing:
        print(f'Warning: missing files for {subject_id}: {missing}. Skipping.')
        return

    # Symlink required files to slicer_in directory (for quicker loading)
    slicer_in_dir = os.path.join(subject_dir, 'atlas_based_output', 'slicer_in')
    os.makedirs(slicer_in_dir, exist_ok=True)
    shutil.copy2(t1_file, os.path.join(slicer_in_dir, os.path.basename(t1_file)))
    shutil.copy2(aparc_file, os.path.join(slicer_in_dir, os.path.basename(aparc_file)))
    shutil.copy2(wmseg_file, os.path.join(slicer_in_dir, os.path.basename(wmseg_file)))

    print(f'\nProcessing subject: {subject_id}')

    # Show resection photo
    resection_photo = os.path.join(subject_dir, "pic2mri_output", f'{subject_id}_resection_photo.jpg')
    show_resection_photo(resection_photo)

    # Step 1: Open Slicer GUI for manual inspection
    open_slicer_gui(slicer_path, subject_dir)

    # Step 2: Ask for FreeSurfer label names
    labels_input = input(f'Enter FreeSurfer label names to extract for {subject_id} (comma separated): ')
    labels_list = parse_labels_input(labels_input, lut_dict)
    if not labels_list:
        print(f'No valid labels entered for {subject_id}. Skipping mask extraction.')
        return

    # Step 3: Create masks using nibabel
    output_dir = os.path.join(subject_dir, 'atlas_based_output')
    create_masks_from_aparc(aparc_file, wmseg_file, labels_list, output_dir)

# ------------------------
# Main loop
# ------------------------
def main():
    config_path = os.path.join('res_pic2mri', 'config.yaml')
    config = load_config(config_path)
    slicer_path = config['slicer_exe_path']
    mri_data_dir = config['mri_data_dir']
    pic_data_dir = config['pic_data_dir']

    # Path to FreeSurfer LUT
    lut_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ext', 'FreeSurfer', 'FreeSurferColorLUT.txt')
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f'FreeSurferColorLUT.txt not found at {lut_path}')
    lut_dict = load_freesurfer_lut(lut_path)

    for subject_id in os.listdir(mri_data_dir):
        subject_dir = os.path.join(mri_data_dir, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        process_subject(subject_dir, subject_id, slicer_path, pic_data_dir, lut_dict)

if __name__ == '__main__':
    main()
