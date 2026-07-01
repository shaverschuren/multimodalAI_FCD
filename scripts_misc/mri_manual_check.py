import os
import pandas as pd
import pydicom
import random

import matplotlib.pyplot as plt

# File paths
autoselect_csv = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ria_pull\\pre_op_scans_autoselect.csv'
manual_select_csv = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ria_pull\\pre_op_scans_manual_select.csv'
output_csv = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ria_pull\\mri_manual_check_results.csv'
t1w_dicom_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ria_pull\\raw\\T1w'
flair_dicom_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\ria_pull\\raw\\FLAIR'

def on_key(event):
    """ Close figure on 'escape' key press"""
    if event.key == 'escape':
        plt.close(fig)

# Load patient data
df = pd.read_csv(autoselect_csv)

# Check whether output file exists, if so load previous results if wanted
if os.path.exists(output_csv):
    reload = input(f"{output_csv} exists. Reload previous results? (y/n): ").strip().lower()
    if reload == 'y':
        results_df = pd.read_csv(output_csv)
        results = results_df.to_dict('records')
    else:
        results = []
else:
    # Initialize results list
    results = []

# Loop over patients
for patient_id in df['patient_id'].unique():
    # Skip if already done
    if any(r['patient_id'] == patient_id for r in results):
        print(f"Skipping {patient_id}, already done.")
        continue
    # Else, process
    print(f"Checking {patient_id}:")

    # Extract patient data
    patient_data = df[df['patient_id'] == patient_id]
    t1w_session_data = patient_data[patient_data['Type'] == 'T1w']
    flair_session_data = patient_data[patient_data['Type'] == 'FLAIR']

    # Extract scan paths and metadata
    scan_paths = [
        f"{os.path.join(t1w_dicom_root, t1w_session_data['session'].values[0], t1w_session_data['scan'].values[0], 'DICOM')}",
        f"{os.path.join(flair_dicom_root, flair_session_data['session'].values[0], flair_session_data['scan'].values[0], 'DICOM')}"
    ]
    voxel_sizes = [session_data["VoxelSize"].values[0] for session_data in [t1w_session_data, flair_session_data]]
    image_sizes = [session_data["Dimensions"].values[0] for session_data in [t1w_session_data, flair_session_data]]
    session_dates = [session_data["StudyDate"].values[0] for session_data in [t1w_session_data, flair_session_data]]
    session_series = [session_data["SeriesDescription"].values[0] for session_data in [t1w_session_data, flair_session_data]]

    # Plot middle slice of each scan
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    for i, scan_path in enumerate(scan_paths):
        if not os.path.exists(scan_path):
            axes[i].set_title(f"Missing: {scan_path}")
            axes[i].axis('off')
            continue

        # List all DICOM files in the folder
        dcm_files = [f for f in os.listdir(scan_path) if f.endswith('.dcm')]
        if not dcm_files:
            axes[i].set_title(f"No DICOMs: {scan_path}")
            axes[i].axis('off')
            continue

        # Read DICOM files and get their slice locations to display middle one (only 20% to speed up)
        slices = []
        sample_size = max(1, int(0.2 * len(dcm_files)))
        sampled_files = random.sample(dcm_files, sample_size)
        for f in sampled_files:
            dcm_fp = os.path.join(scan_path, f)
            ds = pydicom.dcmread(dcm_fp, stop_before_pixels=True, force=True)
            # Use SliceLocation if available, else ImagePositionPatient[2]
            if hasattr(ds, 'SliceLocation'):
                loc = ds.SliceLocation
            elif hasattr(ds, 'ImagePositionPatient'):
                loc = ds.ImagePositionPatient[2]
            else:
                loc = None
            
            slices.append((loc, f))
        # Filter out slices with missing location
        slices = [s for s in slices if s[0] is not None]
        if not slices:
            axes[i].set_title(f"No valid slices: {scan_path}")
            axes[i].axis('off')
            continue
        # Sort by location and pick the middle one
        slices.sort(key=lambda x: x[0])
        mid_idx = len(slices) // 2
        dcm_file = os.path.join(scan_path, slices[mid_idx][1])
        ds = pydicom.dcmread(dcm_file)
        img = ds.pixel_array

        # Display
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Res: {voxel_sizes[i]}, Size: {image_sizes[i]}")
        axes[i].text(0.5, -0.05, f"Series: {session_series[i]}", ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        axes[i].text(0.5, -0.15, f"Date: {session_dates[i]}", ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        axes[i].axis('off')

    # Set title, connect key event and show
    plt.suptitle(f'Patient: {patient_id}')
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # Ask for quality input
    ok = input("Quality OK (1) or not OK: Low resolution (2), Artefacts (3), Weird date (4)? ")
    results.append({'patient_id': patient_id, 'quality_ok': int(ok)})

    # Save results after each patient
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Summarize results
results_df = pd.DataFrame(results)
print("Done! Summary:")
print(f"Quality OK (1): {results_df['quality_ok'].value_counts().get(1, 0)}")
print(f"Low resolution (2): {results_df['quality_ok'].value_counts().get(2, 0)}")
print(f"Artefacts (3): {results_df['quality_ok'].value_counts().get(3, 0)}")
print(f"Weird date (4): {results_df['quality_ok'].value_counts().get(4, 0)}")

# Make pre_op_scans_manual_select.csv template to fill in later
manual_select_df = pd.read_csv(autoselect_csv)
for idx, row in results_df.iterrows():
    if row['quality_ok'] != 1:
        manual_select_df.loc[manual_select_df['patient_id'] == row['patient_id'], manual_select_df.columns.difference(['patient_id'])] = row['quality_ok']
# Store
manual_select_df.to_csv(manual_select_csv, index=False)