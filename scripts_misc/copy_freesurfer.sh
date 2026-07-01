#!/bin/bash
# This script copies FreeSurfer output directories for each RESP-id listed in a CSV file.
# The CSV should have columns: RESP-id, source_directory.
# For each entry, it creates a destination directory and copies the contents from the source directory.
# Usage: bash ./copy_freesurfer.sh

# Path to CSV containing freesurfer paths
csv_path="L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\mri_search\freesurfer_paths.csv"
# Destination base directory
freesurfer_dir="L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\freesurfer"

# Read CSV, skip header
tail -n +2 "$csv_path" | while IFS=, read -r resp_id src_dir; do
    # Remove carriage return characters from variables
    resp_id=$(echo "$resp_id" | tr -d '\r')
    src_dir=$(echo "$src_dir" | tr -d '\r')
    dest_dir="${freesurfer_dir}\\${resp_id}"
    # Normalize path
    if [ ! -d "$src_dir" ]; then
        src_dir="${src_dir//\\//}"
    fi
    # Copy dir
    echo "Copying freesurfer dir for $resp_id: $src_dir"
    mkdir -p "$dest_dir"
    cp -r "$src_dir"/* "$dest_dir"/
done