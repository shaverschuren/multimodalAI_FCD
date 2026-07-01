import os
import pandas as pd
import shutil
from tqdm import tqdm

# ieeg_search.py
# Author: Sjors
# Description: Script to check and copy all available iEEG files for selected patients.

# Define paths
metadata_csv = os.path.join("L:\\", "her_knf_golf", "Wetenschap", "newtransport", "Sjors", "data", "selection", "selected_summary.csv")
aecog_dir = os.path.join("L:\\", "Respect-leijten", "5_BIDS", "acute_ECoG")
cecog_dir = os.path.join("L:\\", "Respect-leijten", "5_BIDS", "chronic_ECoG")
out_root = os.path.join("L:\\", "her_knf_golf", "Wetenschap", "newtransport", "Sjors", "data", "ieeg_search")
# Overwrite arg
overwrite = False


def safe_copy(src, dst, ignore_anat=True):
    """Copy file or directory from src to dst, handling both files and directories."""
    # Copy all contents from source to destination
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        if os.path.isdir(src_path):
            # Ignore anat files if specified
            ignore_pattern = shutil.ignore_patterns('*anat*','*.nii', '*.nii.gz') if ignore_anat else None
            # Copy directory
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True, ignore=ignore_pattern)
        else:
            # Copy file
            shutil.copy2(src_path, dst_path)

# Load metadata
df = pd.read_csv(metadata_csv)

# Loop over patients and look for iEEG data
print(f"Checking {len(df)} patients for iEEG data...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    # Get patient id
    patient_id = row['Participant Id']

    # Check whether already done
    if not (os.path.exists(os.path.join(out_root, f"sub-{patient_id}")) and not overwrite):

        # Check for acute ECoG
        aecog_dir_pt = os.path.join(aecog_dir, f"sub-{patient_id}")
        if os.path.exists(aecog_dir_pt):
            # Specify destination folder
            dest_dir = os.path.join(out_root, f"sub-{patient_id}", "ses-intraop")
            # Create destination folder if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            # Copy all contents from source to destination
            safe_copy(aecog_dir_pt, dest_dir)

        # Check for chronic ECoG
        cecog_dir_pt = os.path.join(cecog_dir, f"sub-{patient_id}")
        if os.path.exists(cecog_dir_pt):
            # Specify destination folder
            dest_dir = os.path.join(out_root, f"sub-{patient_id}", "ses-preop")
            # Create destination folder if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            # Copy all contents from source to destination
            safe_copy(cecog_dir_pt, dest_dir)

print("Done. Results saved to:", out_root)