import os
import shutil
import pandas as pd
from tqdm import tqdm

OVERWRITE_EXISTING = False

data_dir = os.path.join("L:\\", "Respect-leijten", "0_Reports", "Pictures_aECoG")
out_dir = os.path.join("..", "data", "dataset_ECoG_pictures")
selected_csv = os.path.join("..", "data", "selection", "selected_summary.csv")

# absolute / normalized source path for safety checks (do NOT modify data_dir)
data_dir_abs = os.path.normcase(os.path.abspath(data_dir))

# Load selected patient IDs
df_selected = pd.read_csv(selected_csv, dtype=str)
selected_ids = df_selected["Participant Id"].unique()

print(f"Total selected patient IDs: {len(selected_ids)}")
for pid in tqdm(selected_ids, desc="Processing patients", unit="pt"):
    pid = str(pid).strip()
    data_path = os.path.join(data_dir, pid)
    out_path = os.path.join(out_dir, pid)

    data_path_abs = os.path.normcase(os.path.abspath(data_path))
    out_path_abs = os.path.normcase(os.path.abspath(out_path))

    # Safety: ensure out_path is not the same as or inside data_dir
    if out_path_abs == data_path_abs or out_path_abs.startswith(data_dir_abs + os.sep):
        tqdm.write(f"\x1b[91mERROR: {pid}: Unsafe output path (would modify source). Skipping.\x1b[0m")
        continue

    has_data = os.path.isdir(data_path)
    has_out = os.path.isdir(out_path)

    try:
        if has_data and not has_out:
            tqdm.write(f"\033[92mINFO:\033[0m {pid}: Copying data for patient ID")
            shutil.copytree(data_path, out_path)
        elif has_data and has_out:
            if OVERWRITE_EXISTING:
                tqdm.write(f"\033[93mWARNING:\033[0m {pid}: Output directory already exists, deleting to replace.")
                # Extra safety: ensure we are not deleting anything inside the source
                if out_path_abs.startswith(data_dir_abs + os.sep) or out_path_abs == data_dir_abs:
                    tqdm.write(f"\x1b[91mERROR:\x1b[0m {pid}: Attempt to delete path inside source. Skipping.")
                    continue
                shutil.rmtree(out_path)
                shutil.copytree(data_path, out_path)
            else:
                tqdm.write(f"\033[93mWARNING:\033[0m {pid}: Output directory already exists, skipping copy.")
        elif not has_data and has_out:
            tqdm.write(f"\033[92mINFO:\033[0m {pid}: Data already there but not processed")
        else:
            tqdm.write(f"\x1b[91mERROR:\x1b[0m {pid}: No data found for patient ID")
    except Exception as e:
        tqdm.write(f"\x1b[91mERROR:\x1b[0m {pid}: Exception: {e}")
