import os
import io
import pandas as pd
import getpass
import openpyxl
import msoffcrypto
import shutil
from tqdm import tqdm

# create_acog_pics_ds.py
# Author: Sjors
# Description: Script to create dataset for ACOG PICS from selected patient IDs
#              by coupling RESP-nrs with AZU-nrs from codelist and then searching
#              the raw "schetsen" data from "L:\her_knf_golf\Overige\everyone\Frans"
#              for matching AZU-nrs. Afterwards, we save all pseudonymized data dirs. 

# Set dirs and parameters
data_dir = os.path.join("..", "data", "tmp", "ECoG_pictures_raw")
out_dir = os.path.join("..", "data", "dataset_ECoG_pictures")
selected_csv = os.path.join("..", "data", "selection", "selected_summary.csv")
codelist_path = os.path.join("L:\\","her_knf_golf","Wetenschap","newtransport","0_EpiLab","4_RESPectDB","1_Codelist","20250327_RESPect_lijst_final_retrospectief+prospectief_incl_info - kopie.xlsx")

# Ensure out_dir exists
os.makedirs(out_dir, exist_ok=True)

# Load codelist in read-only mode (!) + ask for password if needed
sheets_dict = {}

# Decrypt codelist
password = getpass.getpass("Please enter codelist password: ")
decrypted = io.BytesIO()
with open(codelist_path, "rb") as f:
    office_file = msoffcrypto.OfficeFile(f)
    office_file.load_key(password=password)
    office_file.decrypt(decrypted)
# Read all sheets into dictionary of DataFrames
for sheet_name in openpyxl.load_workbook(decrypted, read_only=True).sheetnames:
    skiprows = 1 if sheet_name == "RESPect_prospectief_digitaal_IC" else 0
    decrypted.seek(0)
    df = pd.read_excel(
        decrypted,
        sheet_name=sheet_name,
        dtype=str,
        engine="openpyxl",
        skiprows=skiprows,
        usecols=[0, 1]
    )
    sheets_dict[sheet_name] = df

# Load selected patient IDs
df_selected = pd.read_csv(selected_csv, dtype=str)
selected_ids = df_selected["Participant Id"].unique()

# Find and couple RESP-nrs with AZU-nrs
coupled_ids = []
for resp_id in selected_ids:
    found = False
    # Loop over sheets to find RESP_id
    for sheet_df in sheets_dict.values():
        # Search proper columns
        resp_col = "RESP_number" if "RESP_number" in sheet_df.columns else "RESP"
        azu_col = "AZU_number" if "AZU_number" in sheet_df.columns else "AZU_nummer"
        # Search for matching RESP_id
        match = sheet_df[sheet_df[resp_col] == resp_id]
        # If found, get AZU number and break
        if not match.empty:
            azu_number = match.iloc[0][azu_col]
            coupled_ids.append({"RESP_number": resp_id, "AZU_number": azu_number})
            found = True
            break
    # If not found, throw warning and add with None
    if not found:
        print(f"\033[93mWARNING:\033[0m {resp_id} not found in codelist.")
        coupled_ids.append({"RESP_number": resp_id, "AZU_number": None})
# Store in dataframe
coupled_df = pd.DataFrame(coupled_ids)
coupled_df = coupled_df.sort_values(by="RESP_number").reset_index(drop=True)

# Now, read the raw "schetsen" data and match AZU-nrs
print(f"Found {len(os.listdir(data_dir))} subdirectories in raw data dir.")
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    print(f"Processing subdir: {subdir_path}")
    for patient_dir in tqdm(os.listdir(subdir_path), desc=f"Processing {subdir}", unit="pt"):
        patient_dir_path = os.path.join(subdir_path, patient_dir)
        if os.path.isdir(patient_dir_path):
            # Extract AZU number from dir name
            try:
                azu_number = patient_dir.split("_")[-1]  # Assuming dir name ends with AZU number
                if len(azu_number) != 7 or not azu_number.isdigit():
                    azu_number = patient_dir.split("_")[-2]  # Try second last part
                    if len(azu_number) != 7 or not azu_number.isdigit():
                        azu_number = patient_dir.split("_")[-3]  # Try third last part
                        if len(azu_number) != 7 or not azu_number.isdigit():
                            raise IndexError
            except IndexError:
                tqdm.write(f"\033[93mWARNING:\033[0m Could not extract AZU number from {patient_dir}.")
                continue

            # Check if this AZU number is in our coupled_df
            if azu_number in coupled_df["AZU_number"].values:
                resp_number = coupled_df[coupled_df["AZU_number"] == azu_number]["RESP_number"].values[0]

                # Copy dir to out_dir with new name
                new_dir_name = resp_number
                new_dir_path = os.path.join(out_dir, new_dir_name)

                # Copy the entire directory tree
                if not os.path.exists(new_dir_path):
                    shutil.copytree(patient_dir_path, new_dir_path)
                    tqdm.write(f"Copied {patient_dir_path} to {new_dir_path}")
                else:
                    tqdm.write(f"\033[93mWARNING:\033[0m Directory {new_dir_path} already exists.")
                    new_dir_path += "_2"
                    tqdm.write(f"Renaming to {new_dir_path}")
                    if not os.path.exists(new_dir_path):
                        shutil.copytree(patient_dir_path, new_dir_path)
                        tqdm.write(f"Copied {patient_dir_path} to {new_dir_path}")
                    else:
                        tqdm.write(f"\033[93mWARNING:\033[0m Directory {new_dir_path} already exists.")
                        new_dir_path = new_dir_path.rstrip("_2") + "_3"
                        tqdm.write(f"Renaming to {new_dir_path}")
                        if not os.path.exists(new_dir_path):
                            shutil.copytree(patient_dir_path, new_dir_path)
                            tqdm.write(f"Copied {patient_dir_path} to {new_dir_path}")
                        else:
                            tqdm.write(f"\033[91mERROR:\033[0m Could not copy {patient_dir_path}, even with new name {new_dir_path}.")

            else:
                tqdm.write(f"INFO: AZU number {azu_number} not found in coupled IDs.")