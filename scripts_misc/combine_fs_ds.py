import os
import shutil
import pandas as pd
from tqdm import tqdm

fastsurfer_dir = os.path.join("..", "data", "fastsurfer_out")
freesurfer_dir = os.path.join("..", "data", "freesurfer")
out_dir = os.path.join("..", "data", "dataset_fs")
selected_csv = os.path.join("..", "data", "selection", "selected_summary.csv")

# Load selected patient IDs
df_selected = pd.read_csv(selected_csv, dtype=str)
selected_ids = df_selected["Participant Id"].unique()

print(f"Total selected patient IDs: {len(selected_ids)}")
for pid in tqdm(selected_ids, desc="Processing patients", unit="pt"):
    pid = str(pid).strip()
    fastsurfer_path = os.path.join(fastsurfer_dir, pid)
    freesurfer_path = os.path.join(freesurfer_dir, pid)
    out_path = os.path.join(out_dir, pid)

    has_fastsurfer = os.path.isdir(fastsurfer_path)
    has_freesurfer = os.path.isdir(freesurfer_path)
    has_out = os.path.isdir(out_path)

    try:
        # Skip if already there
        if has_out:
            tqdm.write(f"\033[93mWARNING:\033[0m {pid}: Output directory already exists, skipping.")
            continue

        os.makedirs(out_path, exist_ok=True)
        if has_fastsurfer:
            tqdm.write(f"\033[92mINFO:\033[0m {pid}: Moving FastSurfer data. {fastsurfer_path} -> {out_path}")
            shutil.move(fastsurfer_path, out_path)
        else:
            if has_freesurfer:
                tqdm.write(f"\033[92mINFO:\033[0m {pid}: Moving FreeSurfer data. {freesurfer_path} -> {out_path}")
                shutil.move(freesurfer_path, out_path)
            else:
                tqdm.write(f"\x1b[91mERROR:\x1b[0m {pid}: No data found for patient ID")
    except Exception as e:
        tqdm.write(f"\x1b[91mERROR:\x1b[0m {pid}: Exception: {e}")
