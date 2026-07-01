import pandas as pd
import os
import re
from collections import defaultdict

# mri_search.py
# Author: Sjors
# Description: Script to sort all available .nii files and check which are missing for selected patients.
#              Uses predefined .txt files with paths to all .nii files, created using recursive shell search
#              of all .nii or .nii.gz files with "RESP" in full path. Also extract Freesurfer paths.

# Define dirs
data_dir = os.path.join("L:\\", "her_knf_golf", "Wetenschap", "newtransport", "Sjors", "data")
search_dir = os.path.join(data_dir, "mri_search")

# Define files
selection_csv = os.path.join(data_dir, "selection", "selected_summary.csv")
nii_path_files = [
    os.path.join(search_dir, "newtransport_nifti_paths.txt"),
    os.path.join(search_dir, "Respect-leijten-BIDS_nifti_paths.txt"),
    os.path.join(search_dir, "Respect-leijten-aECoG_nifti_paths.txt"),
    os.path.join(search_dir, "Respect-leijten-cECoG_nifti_paths.txt"),
]

# Get selected RESP-nrs
df = pd.read_csv(selection_csv)
resp_ids = df["Participant Id"].dropna().unique().tolist()

# Read all nii paths
nii_paths = []
for file_path in nii_path_files:
    with open(file_path, "r", encoding="utf-16") as f:
        content = f.read()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        nii_paths.extend(lines)

# Create a dictionary to hold paths for each RESP number
resp_paths = defaultdict(list)
# Match paths to RESP numbers
resp_pattern = re.compile(r'RESP(\d{4})')
for path in nii_paths:
    match = resp_pattern.search(path)
    if match:
        resp_num = f'RESP{match.group(1)}'
        if resp_num in resp_ids:
            resp_paths[resp_num].append(path)
# Add RESP ids with no matching paths as None
for resp_id in resp_ids:
    if resp_id not in resp_paths or not resp_paths[resp_id]:
        resp_paths[resp_id] = None

# Convert resp_paths to DataFrame
df_out = pd.DataFrame([
    {
        "Participant Id": resp_id,
        "Has MRI": 1 if paths is not None else 0,
        "Freesurfer": int(any("freesurfer" in p.lower() for p in paths)) if paths is not None else 0,
        "Respect-leijten": int(any("respect-leijten" in p.lower() for p in paths)) if paths is not None else 0,
        "Respect-leijten paths": "\n".join(p for p in paths if "respect-leijten" in p.lower()) if paths is not None else "",
        "All NII Paths": "\n".join(paths) if paths is not None else ""
    }
    for resp_id, paths in resp_paths.items()
])
df_out = df_out.sort_values(by="Participant Id").reset_index(drop=True)

# Extract and save Freesurfer paths separately
freesurfer_rows = []
for _, row in df_out.iterrows():
    if row["Freesurfer"] == 1:
        paths = row["All NII Paths"].split('\n')
        fs_paths = [p for p in paths if "freesurfer" in p.lower()]
        for fs_path in fs_paths:
            # Freesurfer dir is two levels up
            fs_dir = os.path.dirname(os.path.dirname(fs_path))
            freesurfer_rows.append({
                "Participant Id": row["Participant Id"],
                "Freesurfer Path": fs_dir
            })
# Save to df and drop duplicates
df_freesurfer = pd.DataFrame(freesurfer_rows)
df_freesurfer = df_freesurfer.drop_duplicates()
# If there's two separate freesurfer dirs for one participant, keep the cECoG one (CCEP and cECoG are duplicates)
duplicates = df_freesurfer[df_freesurfer.duplicated(subset=["Participant Id"], keep=False)]["Participant Id"].unique()
df_freesurfer = df_freesurfer[
    ~(
        df_freesurfer["Participant Id"].isin(duplicates) &
        ~df_freesurfer["Freesurfer Path"].str.contains("chronic_ECoG", case=False)
    )
]

# Extract and save T1w paths separately
t1w_rows = []
for _, row in df_out.iterrows():
    if row["Respect-leijten"] == 1:
        paths = row["Respect-leijten paths"].split('\n')
        source_t1w_paths = [
            p for p in paths
            if "freesurfer" not in p.lower() and "_CT" not in p and "t1" in p.lower()
        ]
        if len(source_t1w_paths) == 1:
            t1w_rows.append({
                "Participant Id": row["Participant Id"],
                "T1w Path": source_t1w_paths[0]
            })
        elif len(source_t1w_paths) > 1:
            # Prefer non-defaced scans (will skull-strip anyways)
            non_deface = [p for p in source_t1w_paths if "deface" not in p]
            if len(non_deface) == 1:
                t1w_rows.append({
                    "Participant Id": row["Participant Id"],
                    "T1w Path": non_deface[0]
                })
            elif len(non_deface) > 1:
                chosen_path = next((p for p in non_deface if "chronic_ECoG" in p), non_deface[0])
                t1w_rows.append({
                    "Participant Id": row["Participant Id"],
                    "T1w Path": chosen_path
                })
            else:
                # If no non-defaced, check whether good outcome.
                good_outcome = next((p for p in source_t1w_paths if "good outcome" in p), None)
                if good_outcome:
                    t1w_rows.append({
                        "Participant Id": row["Participant Id"],
                        "T1w Path": good_outcome
                    })
        else:
            print(f"Warning: No T1w path found for {row['Participant Id']}. Skipping.")
# Save to df
df_t1w = pd.DataFrame(t1w_rows)

# Define output paths
output_csv = os.path.join(search_dir, "available_nii_files.csv")
freesurfer_csv = os.path.join(search_dir, "freesurfer_paths.csv")
t1w_csv = os.path.join(search_dir, "t1w_paths.csv")
# Store to csv
df_out.to_csv(output_csv, index=False)
df_freesurfer.to_csv(freesurfer_csv, index=False)
df_t1w.to_csv(t1w_csv, index=False)
# Print completion message
print(f"OK. Saved summary .csv files to {search_dir}")