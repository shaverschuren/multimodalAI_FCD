#!/bin/bash
# This script copies T1w NII files for each RESP-id listed in a CSV file.
# The CSV should have columns: RESP-id, source_path.
# For each entry, it creates a destination directory and copies the contents from the source directory.
# Usage: bash ./copy_t1w_nii.sh

# Path to CSV containing T1w paths
csv_path="L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\mri_search\t1w_paths.csv"
# Destination base directory
t1w_dir="L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\mri_search\T1w"

# Read CSV, skip header
tail -n +2 "$csv_path" | while IFS=, read -r resp_id src_path; do
    # Remove carriage return characters from variables
    resp_id=$(echo "$resp_id" | tr -d '\r')
    src_path=$(echo "$src_path" | tr -d '\r')
    # Add _deface if applicable
    if [[ "${src_path,,}" == *deface* ]]; then
        dest_path="${t1w_dir}\\${resp_id}_T1w_deface.nii"
    else
        dest_path="${t1w_dir}\\${resp_id}_T1w.nii"
    fi
    # Copy file
    echo "Copying T1w NII file for $resp_id:"
    cp -v "$src_path" "$dest_path"
done