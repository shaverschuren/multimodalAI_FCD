"""Batch export MRB segmentations to NIfTI using 3D Slicer."""

import subprocess
from pathlib import Path
from tqdm import tqdm

# Configuration
SLICER_BIN = "C:\\Users\\sversch6\\AppData\\Local\\slicer.org\\Slicer 5.8.1\\Slicer.exe"
SCRIPT = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\src\\scripts_other\\export_mrb_to_nifti.py"
DATA_DIR = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\gt\\validated_segs\\mrb")
OUT_DIR = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\gt\\validated_segs\\nifti")

OUT_DIR.mkdir(exist_ok=True)

for patient_dir in tqdm(sorted(DATA_DIR.iterdir()), desc="Patients"):
    if not patient_dir.is_dir():
        continue

    mrb_files = list(patient_dir.glob("*.mrb"))
    if not mrb_files:
        tqdm.write(f"Skipping {patient_dir.name}: no MRB found")
        continue

    mrb_path = mrb_files[0]
    patient_out = OUT_DIR / patient_dir.name
    patient_out.mkdir(exist_ok=True)

    cmd = [
        SLICER_BIN,
        "--no-main-window",
        "--python-script", SCRIPT,
        str(mrb_path),
        str(patient_out),
    ]

    tqdm.write(f"Processing {patient_dir.name}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        tqdm.write(f"\033[31mWarning: Processing {patient_dir.name} failed with exit code {result.returncode}\033[0m")

print("All patients processed.")
