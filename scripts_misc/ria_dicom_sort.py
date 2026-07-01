import os
import pandas as pd
import pydicom
from tqdm import tqdm

# ria_dicom_sort.py
# Author: Sjors
# Description: Script to sort and extract metadata from DICOM files extracted from RIA.
#              Selects pre-op and post-op scans from sessions *closest to* pathology dates from the selection CSV.
#              Selects based on criteria: MPR preferred, isotropic voxels, highest resolution.
#              Will be manually checked later, as sometimes these sessions don't contain the best images.

# Setup paths
selection_csv_path = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\selection\\selected_summary.csv'
t1w_dicom_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull\\raw\\T1w'
flair_dicom_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull\\raw\\FLAIR'
output_csv_root = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull'

dicom_roots = [t1w_dicom_root, flair_dicom_root]

def sort_dicom_scans(df_total, typed_scans):
    """Sort scans of a scan type (T1w or FLAIR)"""

    # Init drop_indices
    drop_indices = []

    # If possible, drop scans with very non-iso voxels (i.e. slice thickness is more than 3x in-plane resolution or >2mm)
    non_iso = typed_scans[
        typed_scans['VoxelSize'].apply(
            lambda x: (
                float(x.split(' x ')[2]) / float(x.split(' x ')[0]) > 3.0
                or float(x.split(' x ')[2]) > 2.0
            )
        )
    ].index
    if len(non_iso) < len(typed_scans):
        typed_scans = typed_scans.drop(non_iso)
        drop_indices.extend(non_iso)

    # Check for native iso-3D scans
    is_3d_scans = typed_scans[typed_scans['SeriesDescription'].str.lower().str.contains('3d|vol|iso')]
    # If one, keep that and drop the rest
    if len(is_3d_scans) == 1:
        drop_indices.extend(typed_scans.index.difference(is_3d_scans.index))
        df_total = df_total.drop(drop_indices)
        return df_total
    # Elif multiple, continue with those
    elif len(is_3d_scans) > 1:
        typed_scans = is_3d_scans
    # If no 3D, check for MPR transversal scans and prefer those
    else:
        # Search for MPR transversal
        mpr_scans = typed_scans[typed_scans['SeriesDescription'].str.lower().str.contains('mpr tra|t mpr')]
        # If one, keep that one and drop the rest
        if len(mpr_scans) == 1:
            drop_indices.extend(typed_scans.index.difference(mpr_scans.index))
            df_total = df_total.drop(drop_indices)
            return df_total
        # Elif multiple, continue with those
        elif len(mpr_scans) > 1:
            typed_scans = mpr_scans
        # Else, continue with all scans
        else:
            pass

    # Finally, sort by VoxelSize (smallest first), keep the first (smallest), drop the rest
    scans_sorted = typed_scans.copy()
    scans_sorted['VoxelVolume'] = scans_sorted['VoxelSize'].apply(
        lambda x: float(x.split(' x ')[0]) * float(x.split(' x ')[1]) * float(x.split(' x ')[2])
    )
    scans_sorted = scans_sorted.sort_values('VoxelVolume')
    drop_indices.extend(scans_sorted.index[1:])
    df_total = df_total.drop(drop_indices)

    return df_total

# Init lists
data = []
no_preop_mri_ids = []
dcm_errors = []

# Check if dicom metadeta CSV already exists
intermediate_csv_path = os.path.join(output_csv_root, 'dicom_metadata_all.csv')
if os.path.exists(intermediate_csv_path):
    use_existing = input(f"Metadata CSV '{intermediate_csv_path}' exists. Use existing and skip metadata read? (y/n): ").strip().lower()
    if use_existing == 'y':
        df = pd.read_csv(intermediate_csv_path)
        df_all = df.copy()
        print(f"Loaded metadata from {intermediate_csv_path}")
        skip_metadata_read = True
    else:
        skip_metadata_read = False
else:
    skip_metadata_read = False

# If not skipping, read metadata from DICOM files
if not skip_metadata_read:
    # Print some info
    print(f"Scanning {len(dicom_roots)} DICOM directories for metadata...\n")
    # Loop through sessions, both for extracted FLAIR and T1w
    for dicom_root in dicom_roots:
        sessions = [s for s in os.listdir(dicom_root) if os.path.isdir(os.path.join(dicom_root, s)) and s.startswith('RESP')]
        print(f"Found {len(sessions)} sessions in {dicom_root}")
        # Loop through sessions
        for session in tqdm(sessions, desc="Sessions"):
            # Define path and RESPnr
            session_path = os.path.join(dicom_root, session)
            patient_id = session.split('_')[0]
            # List scans
            scans = [scan for scan in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, scan))]
            # Loop through scans
            for scan in scans:
                # Define paths
                scan_path = os.path.join(session_path, scan)
                dicom_dir = os.path.join(scan_path, 'DICOM')
                # Check and read DICOM files
                if not os.path.isdir(dicom_dir):
                    continue
                dicom_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
                # Check whether series
                if not dicom_files:
                    continue
                if len(dicom_files) == 1:
                    continue  # Skip, not a series
                # Read first DICOM file for metadata
                dicom_file_path = os.path.join(dicom_dir, dicom_files[0])
                try:
                    ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
                except Exception as e:
                    dcm_errors.append(f"Error reading {dicom_file_path}: {e}\n")
                # Try to get date and time, not available for all
                try:
                    study_date = pd.to_datetime(getattr(ds, 'StudyDate', ''), format='%Y%m%d', errors='coerce').date()
                    study_time = pd.to_datetime(getattr(ds, 'StudyTime', ''), format='%H%M%S', errors='coerce').time()
                except Exception as e:
                    dcm_errors.append(f"Missing date/time in {dicom_file_path}: {e}\n")
                    study_date = 'N/A'
                    study_time = 'N/A'
                # Store metadata
                try:
                    info = {
                        'patient_id': patient_id,
                        'session': session,
                        'scan': scan,
                        'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                        'StudyDate': study_date,
                        'StudyTime': study_time,
                        'Dimensions': f"{getattr(ds, 'Rows', 0):>4d} x {getattr(ds, 'Columns', 0):>4d} x {len(dicom_files):>4d}",
                        'VoxelSize': f"{float(getattr(ds, 'PixelSpacing', [0, 0])[0]):.2f} x {float(getattr(ds, 'PixelSpacing', [0, 0])[1]):.2f} x {float(getattr(ds, 'SliceThickness', 0)):.2f}",
                        'Type': dicom_root.split('\\')[-1]
                    }
                    data.append(info)
                except Exception as e:
                    dcm_errors.append(f"Unexpected error extracting metadata from {dicom_file_path}: {e}\n")


    # Store in dataframe and sort
    df = pd.DataFrame(data)
    df = df.sort_values(by=['patient_id', 'session', 'scan']).reset_index(drop=True)

    # Save intermediate CSV
    df_all = df.copy()  # Keep a copy of all data before selection
    df_all.to_csv(intermediate_csv_path, index=False)
    print(f"\nIntermediate metadata saved to {intermediate_csv_path}")

# Selection of scans to use

# Keep only sessions that have both T1w and FLAIR scans (so n unique types == 2)
sessions_with_types = df.groupby(['patient_id', 'session'])['Type'].nunique()
valid_sessions = sessions_with_types[sessions_with_types == 2].index
df = df.set_index(['patient_id', 'session'])
df = df.loc[valid_sessions].reset_index()

# Loop over valid sessions to handle cases with multiple scans per type
for (patient_id, session), group in df.groupby(['patient_id', 'session']):
    t1w_scans = group[group['Type'] == 'T1w']
    flair_scans = group[group['Type'] == 'FLAIR']
    # Select T1w scan if multiple
    if len(t1w_scans) > 1:
        df = sort_dicom_scans(df, t1w_scans)
    # Select FLAIR scan if multiple
    if len(flair_scans) > 1:
        df = sort_dicom_scans(df, flair_scans)

# For session selection, read pathology dates
selection_df = pd.read_csv(selection_csv_path)
pathology_dates = selection_df.set_index('Participant Id')['P9Path01']
# For each participant, keep only the session with StudyDate before but closest to pathology date as pre-op MRI
selected_indices = []
for participant_id, path_date in pathology_dates.items():
    # Get scans for this participant
    scans = df[df['patient_id'] == participant_id]
    # Filter scans before pathology date
    scans_before = scans[
        pd.to_datetime(scans['StudyDate'], format='mixed').dt.date < pd.to_datetime(path_date, format='mixed').date()
    ]
    if scans_before.empty: 
        scans_before = scans[pd.to_datetime(scans['StudyDate'], format='mixed').dt.date == pd.to_datetime(path_date, format='mixed').date()]
    # Append session closest to pathology date
    if not scans_before.empty:
        # Append all scans from the session with StudyDate closest to pathology date
        closest_date = scans_before['StudyDate'].max()
        session_scans = scans_before[scans_before['StudyDate'] == closest_date]
        selected_indices.extend(session_scans.index.tolist())
    else:
        # No pre-op MRI found before pathology date
        participant_exists = participant_id in df_all['patient_id'].values
        # Log participant with reason
        if participant_exists:
            no_preop_mri_ids.append((participant_id, "Excluded by autoselect"))
        else:
            no_preop_mri_ids.append((participant_id, "Not in RIA"))

# Filter df to keep only selected scans
df = df.loc[selected_indices].reset_index(drop=True)

# Also search post-op scans (after pathology date)
postop_indices = []
for participant_id, path_date in pathology_dates.items():
    # Get scans for this participant
    scans = df_all[df_all['patient_id'] == participant_id]
    # Filter scans after pathology date
    scans_after = scans[pd.to_datetime(scans['StudyDate'], format='mixed').dt.date == pd.to_datetime(path_date, format='%d-%m-%Y').date()]
    if scans_after.empty: 
            scans_after = scans[
        pd.to_datetime(scans['StudyDate'], format='mixed').dt.date > pd.to_datetime(path_date, format='mixed').date()
    ]
    # Append session closest to pathology date
    if not scans_after.empty:
        # Get all scans from session soonest after pathology date
        closest_date = scans_after['StudyDate'].min()
        session_scans = scans_after[scans_after['StudyDate'] == closest_date]
        # Get highest resolution T1 (MPR if possible)
        t1w_postop_scans = session_scans[session_scans['Type'] == 'T1w']
        if len(t1w_postop_scans) > 1:
            t1w_postop_scans = sort_dicom_scans(t1w_postop_scans, t1w_postop_scans)
        postop_indices.extend(t1w_postop_scans.index.tolist())

# Create postop df
df_postop = df_all.loc[postop_indices].reset_index(drop=True)

# Save to CSV
output_csv_path = os.path.join(output_csv_root, 'pre_op_scans_autoselect.csv')
postop_csv_path = os.path.join(output_csv_root, 'post_op_scans_autoselect.csv')
df.to_csv(output_csv_path, index=False)
df_postop.to_csv(postop_csv_path, index=False)
print(f"Selected pre-op scans saved to {output_csv_path}")
print(f"Selected post-op scans saved to {postop_csv_path}")

# Report errors and participants without MRI before pathology date for later manual check
if no_preop_mri_ids:
    print(f"\n\033[91mWARNING: {len(no_preop_mri_ids)} participants have missing pre-op MRI. See 'err_missing_preop_mri.txt' for details.\033[0m")
    with open(os.path.join(output_csv_root, 'err_missing_preop_mri.txt'), 'w') as f:
        for pid, reason in no_preop_mri_ids:
            f.write(f"{pid}: {reason}\n")
    with open(os.path.join(output_csv_root, 'err_missing_mri_in_ria.txt'), 'w') as f:
        for pid, reason in no_preop_mri_ids:
            if reason == "Not in RIA": 
                f.write(f"{pid},")
if dcm_errors:
    print(f"\033[91mWARNING: {len(dcm_errors)} DICOM read errors occurred. See 'err_dcm_read.txt' for details.\033[0m")
    with open(os.path.join(output_csv_root, 'err_dcm_read.txt'), 'w') as f:
        for error in dcm_errors:
            f.write(f"{error}\n")
