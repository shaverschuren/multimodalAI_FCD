import os
import pandas as pd
from util.plot import plot_patient_entry_density

# analysis_castor_exp.py
# Author: Sjors
# Description: Script to select patients for Multimodal AI project 
#              Imports RESPectDB CASTOR export CSV files and filters patients.

# Set dirs and parameters
export_nr = "_export_20251009"
import_dir = os.path.join("..", "data", "selection", "RESPectDB_CASTOR_export_2025_10_09")
export_dir = os.path.join("..", "data", "selection")
# Toggle plotting of entry density
show_plot = False

# Data import
print("\n=========== Data import ===========\n")

# Extract csv's
csv_files = [f for f in os.listdir(import_dir) if f.startswith('RESPectDB') and f.endswith('.csv')]
# Import in df's
dataframes = {}
for filename in csv_files:
    # Skip unwanted files
    if any(x in filename.lower() for x in ["old", "respque", "wada", "toestemming", "invoeren"]):
        continue
    # Get path and df name
    filepath = os.path.join(import_dir, filename)
    df_name = os.path.splitext(filename)[0].replace(export_nr, "")
    # Read csv's, use first row as header
    df = pd.read_csv(filepath, header=0, sep=';', low_memory=False)
    # Move important but hard-to-read columns to the back
    cols_to_move = ["Participant Status", "Repeating Data Creation Date", "Repeating data Name Custom", "Repeating data Parent"]
    df = df[[col for col in df.columns if col not in cols_to_move] + [col for col in cols_to_move if col in df.columns]]
    # Store in dict
    dataframes[df_name] = df

# Sort DataFrames by "Participant Id" if the column exists (RESP-nr) and print some info
print("Imported files:")
for name, df in dataframes.items():
    # Sort
    if "Participant Id" in df.columns:
        df_sorted = df.sort_values(by="Participant Id")
        dataframes[name] = df_sorted.reset_index(drop=True)
    # Count unique subjects
    unique_subjects = df["Participant Id"].nunique() if "Participant Id" in df.columns else "N/A"

    print(f"  {name:<40.40} : {unique_subjects:>5} subjects, {df.shape[0]:>5} rows x {df.shape[1]:>3} columns")

# Patient selection
print("\n======== Patient selection ========\n")

# Create list to store dropped participant ids, reasons, and report info for later manual evaluation
dropped_ids = []
def add_dropped_ids(ids, reason, reports):
    for pid, report in zip(ids, reports):
        dropped_ids.append({"Participant Id": pid, "reason": reason, "report": report})

# Get initial patient ID list from any FCD proven by pathology
df_pathology = dataframes.get("RESPectDB_Pathology")
df_pathology["P9Path01"] = pd.to_datetime(df_pathology["P9Path01"], format="%d-%m-%Y", errors="coerce")
patient_ids = df_pathology.loc[df_pathology["P9Path03c#Focal Cortical Dysplasia"] == 1, "Participant Id"].unique()
non_fcd_ids = df_pathology.loc[~df_pathology["Participant Id"].isin(patient_ids)]
add_dropped_ids(non_fcd_ids["Participant Id"].tolist(), "non-FCD", non_fcd_ids["P9Path10"].tolist())
print(f"Patients with PA-proven FCD:\t\t{len(patient_ids)}")
# Remove tuberous sclerosis patients
ts_ids = df_pathology.loc[df_pathology["P9Path03c#Tuberous Sclerosis"] == 1, "Participant Id"].unique()
patient_ids = [id for id in patient_ids if id not in ts_ids]
add_dropped_ids(ts_ids, "tuberous sclerosis", df_pathology.loc[df_pathology["Participant Id"].isin(ts_ids), 'P9Path10'].tolist())
print(f"Patients with non-TS PA-proven FCD:\t{len(patient_ids)}")
# Update pathology dataframe
df_pathology = df_pathology[df_pathology["Participant Id"].isin(patient_ids)].reset_index(drop=True)

# Define cols with major PA diagnoses to check for double pathology
pathology_cols = [
    'P9Path03#Mesial temporal sclerosis (MTS)',
    'P9Path03#CNS tumors',
    'P9Path03#Malformations of cortical development',
    'P9Path03#Others',
    'P9Path03#No abnormalities'
]
# Remove duplicate PA reports
duplicate_mask = df_pathology.duplicated(subset=['Participant Id'] + pathology_cols, keep='last')
add_dropped_ids(df_pathology.loc[duplicate_mask, 'Participant Id'].tolist(), "duplicate pathology report", df_pathology.loc[duplicate_mask, 'P9Path10'].tolist())
df_pathology = df_pathology[~duplicate_mask]
# Remove multiple report dual pathology patients
multi_report_mask = df_pathology.duplicated(subset=['Participant Id'], keep=False)
add_dropped_ids(df_pathology.loc[multi_report_mask, 'Participant Id'].tolist(), "multiple dual pathology reports", df_pathology.loc[multi_report_mask, 'P9Path10'].tolist())
df_pathology = df_pathology[~multi_report_mask]
# Remove empty PA reports
empty_mask = df_pathology[pathology_cols].sum(axis=1) == 0
add_dropped_ids(df_pathology.loc[empty_mask, 'Participant Id'].tolist(), "empty pathology report", df_pathology.loc[empty_mask, 'P9Path10'].tolist())
df_pathology = df_pathology[~empty_mask]
# Remove single report dual pathology patients
dual_path_mask = df_pathology[pathology_cols].sum(axis=1) != 1
add_dropped_ids(df_pathology.loc[dual_path_mask, 'Participant Id'].tolist(), "dual pathology in single report", df_pathology.loc[dual_path_mask, 'P9Path10'].tolist())
df_pathology = df_pathology[~dual_path_mask]
# Define ID's with FCS without dual pathology
patient_ids = df_pathology["Participant Id"].unique()
print(f"Patients with single-pathology FCD:\t{len(patient_ids)}\n")

# Manual removal of patients
manual_remove_ids = [
    # ------- PA-based removals -------
    'RESP0674', # No FCD
    'RESP1061', # No FCD, cavernoma
    'RESP0017', # Hemimegencephaly
    'RESP0710', # Hemimegencephaly
    'RESP0674', # No FCD, vascular gliosis
    'RESP0335', # FCD IIIb
    'RESP0354', # TSC
    'RESP0900', # FCD III
    'RESP1047', # FCD IIId
    'RESP1069', # FCD IIId
    'RESP1606', # FCD III
    # -------- Other removals ---------
    'RESP0134', # ILAE 4
    'RESP0285', # Secundair HS
    'RESP0306', # ILAE 4
    'RESP0315', # ILAE 5
    'RESP0322', # Missing GT
    'RESP0454', # ILAE 5
    'RESP0468', # ILAE 4
    'RESP0477', # ILAE 5
    'RESP0606', # ILAE 4
    'RESP0607', # ILAE 5
    'RESP0608', # ILAE 4
    'RESP0677', # ILAE 4
    'RESP0685', # Functionele HS
    'RESP0692', # Missing GT
    'RESP0731', # ILAE 5
    'RESP0867', # ILAE 5
    'RESP0892', # ILAE 4
    'RESP0909', # ILAE 4
    'RESP0968', # ILAE 4
    'RESP1073', # ILAE 5
    'RESP1074', # ILAE 5
    'RESP1113', # TPO disconnectie
    'RESP1134', # Multifocaal
    'RESP1149', # ILAE 4
    'RESP1447', # HS
    'RESP1474', # Missing MRI + EEG
    'RESP1493', # Missing GT
    # ---- Missing data-based removals (not applicable anymore) ----
    # 'RESP0810', # Missing FLAIR
    # 'RESP0875', # Missing FLAIR
    # 'RESP0961', # Missing FLAIR
    # 'RESP1474', # Missing T1w + FLAIR
]
patient_ids = [id for id in patient_ids if id not in manual_remove_ids]
add_dropped_ids(manual_remove_ids, "manual removal", df_pathology.loc[df_pathology["Participant Id"].isin(manual_remove_ids), "P9Path10"].tolist())
print(f"Patients after manual removal:\t\t{len(patient_ids)}")

# Create dataframe for patients that were dropped, for manual evaluation
df_dropped = pd.DataFrame(dropped_ids)

# Manual addition of patients
manual_add_ids = [
    'RESP0216', # FCD II, bad Castor entry
    # 'RESP0867', # FCD IIb, bad Castor entry -> Exclude anyways because of outcome ILAE 5
    'RESP0244', # FCD IIb, extra empty PA report
    'RESP0535', # FCD IIb, extra empty PA report
    # 'RESP0608', # FCD IIb, extra report on re-resection with reactive changes -> Exclude anyways because of outcome ILAE 5
    # 'RESP0731', # FCD IIb, extra report on re-resection with reactive changes -> Exclude anyways because of outcome ILAE 5
    'RESP0896', # FCD IIb, IIa in one report (no balloon cells)
    'RESP0942', # FCD IIb, multiple reports, earlier one said mMCD
    'RESP1054', # FCD IIb, multiple reports, earlier one said mMCD 
    'RESP1141', # FCD II, extra empty PA report
    # 'RESP0477', # FCD Ib, with reactive changes due to grid implantation -> Exclude anyways because of outcome ILAE 5
    'RESP1021', # FCD II, with reactive changes due to grid implantation
]
patient_ids = patient_ids + manual_add_ids
print(f"Patients after manual addition:\t\t{len(patient_ids)}\n")

# For these manual add-ins, retrieve their pathology data from the original dataframe and add to df_pathology
df_pathology_orig = dataframes.get("RESPectDB_Pathology")
manual_add_entries = df_pathology_orig[df_pathology_orig["Participant Id"].isin(manual_add_ids)].copy()
# Convert "P9Path01" to datetime for sorting
manual_add_entries["P9Path01"] = pd.to_datetime(manual_add_entries["P9Path01"], format="%d-%m-%Y", errors="coerce")
# For patients with multiple entries, keep only the most recent one
manual_add_entries = manual_add_entries.sort_values("P9Path01").groupby("Participant Id", as_index=False).last()
# Add these entries to df_pathology
df_pathology = pd.concat([df_pathology, manual_add_entries], ignore_index=True)
df_pathology = df_pathology.drop_duplicates(subset=["Participant Id"], keep="last").sort_values("Participant Id").reset_index(drop=True)

# Filter pathology dataframe to only include selected patients
df_pathology = df_pathology[df_pathology["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"Pathology:    {df_pathology.shape[0]:>3} entries for {df_pathology['Participant Id'].nunique():>3} patients")
# Create surgery dataframe
df_surgery = dataframes.get("RESPectDB_Brain_Surgery_after_08")
df_surgery = df_surgery[df_surgery["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"Surgery:      {df_surgery.shape[0]:>3} entries for {df_surgery['Participant Id'].nunique():>3} patients")
# Create MRI dataframe
df_mri = dataframes.get("RESPectDB_Structural_MRI")
df_mri = df_mri[df_mri["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"MRI:          {df_mri.shape[0]:>3} entries for {df_mri['Participant Id'].nunique():>3} patients")
# Create outcome dataframe
df_outcome = dataframes.get("RESPectDB_Seizure_Outcome")
df_outcome = df_outcome[df_outcome["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"Outcome:      {df_outcome.shape[0]:>3} entries for {df_outcome['Participant Id'].nunique():>3} patients")
# Create demographics dataframe
df_demo = dataframes.get("RESPectDB")
df_demo = df_demo[df_demo["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"Demographics: {df_demo.shape[0]:>3} entries for {df_demo['Participant Id'].nunique():>3} patients")
# Create seizure type dataframe
df_seizure = dataframes.get("RESPectDB_Seizure_Type")
df_seizure = df_seizure[df_seizure["Participant Id"].isin(patient_ids)].reset_index(drop=True)
print(f"Seizure type: {df_seizure.shape[0]:>3} entries for {df_seizure['Participant Id'].nunique():>3} patients")

# Process data: Check for unexpected values, remove redundant info, merge dataframes
print("\n========= Processing data =========\n")

# Pathology
print("Pathology:\t\t", end="")
remove_prefixes = ["P9Path12", "P9Path03", "P9Path05", "P9Path06", "P9Path07", "P9Path08", "P9Path09"]
warnings_list = []
for col in df_pathology.columns:
    if any(col.startswith(prefix) for prefix in remove_prefixes) and not col.endswith("_a"):
        col_sum = df_pathology[col].fillna(0).astype(int).sum()
        if col_sum == 0:
            # Drop columns with only zeros
            df_pathology = df_pathology.drop(columns=[col])
        elif col_sum % 777 == 0:
            # Drop columns with only zeros or 777 (not applicable)
            df_pathology = df_pathology.drop(columns=[col])
        elif col in ["P9Path03c#Focal Cortical Dysplasia", "P9Path03#Malformations of cortical development"]:
            pass # Keep these columns
        else:
            # Collect warnings for expected zero columns with non-zero entries
            patient_ids_with_nonzero = df_pathology.loc[df_pathology[col].fillna(0).astype(int) != 0, "Participant Id"].tolist()
            warnings_list.append(f"Column '{col}' not dropped. Non-zero for patient ids: {patient_ids_with_nonzero}")
# Warn if unexpected data
if warnings_list:
    # Write warnings to a text file in the export folder
    warnings_path = os.path.join(export_dir, "pathology_warnings.txt")
    with open(warnings_path, "w", encoding="utf-8") as f:
        for w in warnings_list:
            f.write(w + "\n")
    print(f"\033[93mWARNING:\033[0m Some unexpected data. See {warnings_path}")
else:
    print("\033[92mOK\033[0m")

# Surgery
print("Surgery:\t\t", end="")
remove_prefixes = ["P4EpSG09", "P4EpSG15", "P4EpSG16", "P4EpSG17", "P04AEDuse", "P4EpSG012"]
for col in df_surgery.columns:
    if any(col.startswith(prefix) for prefix in remove_prefixes):
        df_surgery = df_surgery.drop(columns=[col])
# Identify patients with multiple surgeries
counts = df_surgery["Participant Id"].value_counts()
multi_ids = counts[counts > 1].index.tolist()
# For these patients, drop sEEG and grid im/explantations (P4EpSG14a is True or NaN)
df_surgery = df_surgery[
    ~((df_surgery["Participant Id"].isin(multi_ids)) & (df_surgery["P4EpSG14a"].astype(bool) != False))
].reset_index(drop=True)
print("\033[92mOK\033[0m")
# Convert surgery date to datetime
df_surgery["P4EpSG02"] = pd.to_datetime(df_surgery["P4EpSG02"], format="%d-%m-%Y", errors="coerce")
# Create surgery summary df for merging later
df_surgery_summary = pd.DataFrame({"Participant Id": df_surgery["Participant Id"].unique()})
# Count number of surgeries per patient
df_surgery_summary["P4_custom_num_surgeries"] = df_surgery.groupby("Participant Id").size().reindex(df_surgery_summary["Participant Id"]).fillna(0).astype(int).values
# Add column "all_conclusions", combining all surgery conclusion entries, sorted by date
all_conclusions = []
for pid in df_surgery_summary["Participant Id"]:
    entries = df_surgery[df_surgery["Participant Id"] == pid].copy()
    entries = entries.sort_values("P4EpSG02", ascending=False)
    conclusions = entries["P4EpSG10"].dropna().astype(str).tolist()
    combined = "\n\n".join(conclusions)
    all_conclusions.append(combined)
df_surgery_summary["P4_custom_all_conclusions"] = all_conclusions

# MRI
print("MRI:\t\t\t", end="")
remove_prefixes = ["P11MRI10"]
warnings_list = []
for col in df_mri.columns:
    if any(col.startswith(prefix) for prefix in remove_prefixes) and not col.endswith("_a"):
        col_sum = df_mri[col].fillna(0).astype(int).sum()
        if col_sum == 0:
            # Drop columns with only zeros
            df_mri = df_mri.drop(columns=[col])
        elif col in [
            "P11MRI10c#Focal Cortical Dysplasia", "P11MRI10#Malformations of cortical development",
            "P11MRI10#No abnormalities", "P11MRI10#Status post surgery"
        ]:
            pass  # Keep these columns
        else:
            # Collect warnings for expected zero columns with non-zero entries
            patient_ids_with_nonzero = df_mri.loc[df_mri[col].fillna(0).astype(int) != 0, "Participant Id"].tolist()
            warnings_list.append(f"Column '{col}' not dropped. Non-zero for patient ids: {patient_ids_with_nonzero}")
# Warn if unexpected data
if warnings_list:
    # Write warnings to a text file in the export folder
    warnings_path = os.path.join(export_dir, "mri_warnings.txt")
    with open(warnings_path, "w", encoding="utf-8") as f:
        for w in warnings_list:
            f.write(w + "\n")
    print(f"\033[93mWARNING:\033[0m Some unexpected data. See {warnings_path}")
else:
    print("\033[92mOK\033[0m")
# Convert MRI date to datetime
df_mri["P11MRI01"] = pd.to_datetime(df_mri["P11MRI01"], format="%d-%m-%Y", errors="coerce")
# Create MRI summary df for merging later
df_mri_summary = pd.DataFrame({"Participant Id": df_mri["Participant Id"].unique()})
# Count number of MRIs per patient
df_mri_summary["P11_custom_num_mris"] = df_mri.groupby("Participant Id").size().reindex(df_mri_summary["Participant Id"]).fillna(0).astype(int).values
# Add column "pre_or_MRI": 1 if any entry for this participant has P11MRI12 == 1, else 0
pre_or_mri_map = df_mri.groupby("Participant Id")["P11MRI12"].apply(lambda x: int((x == 1).any()))
df_mri_summary["P11_custom_pre_or_MRI"] = df_mri_summary["Participant Id"].map(pre_or_mri_map).fillna(0).astype(int)
# Add column "post_or_MRI": 1 if any entry for this participant has P11MRI12 > 1, else 0
post_or_mri_map = df_mri.groupby("Participant Id")["P11MRI12"].apply(lambda x: int((x > 1).any()))
df_mri_summary["P11_custom_post_or_MRI"] = df_mri_summary["Participant Id"].map(post_or_mri_map).fillna(0).astype(int)
# Add column "pre_or_abnormality", taking last known pre-op MRI abnormality (P11MRI07) that is not 6 (unknown)
pre_or_abnormality = []
for pid in df_mri_summary["Participant Id"]:
    # Filter for pre-op MRI entries
    entries = df_mri[(df_mri["Participant Id"] == pid) & (df_mri["P11MRI12"] == 1)]
    entries = entries.copy()
    # Loop over entries, sort by date, take last valid P11MRI07 value that is not 6 (unknown)
    while not entries.empty:
        # Take last entry
        last_entry = entries.sort_values("P11MRI01").iloc[-1]
        # Get P11MRI07 ("abnormality") value
        value = last_entry["P11MRI07"]
        if pd.notna(value) and int(value) != 6:
            # If now unknown, take this value and stop
            pre_or_abnormality.append(value)
            break
        else:
            # If unknown, remove this one and continue with next last entry
            entries = entries[entries.index != last_entry.name]
    if entries.empty:
        pre_or_abnormality.append(None)
        continue
df_mri_summary["P11_custom_pre_or_abnormality"] = pre_or_abnormality
# Add column "all_conclusions", combining all MRI conclusion entries, sorted by date
all_conclusions = []
for pid in df_mri_summary["Participant Id"]:
    entries = df_mri[df_mri["Participant Id"] == pid].copy()
    entries = entries.sort_values("P11MRI01", ascending=False)
    conclusions = entries["P11MRI11"].dropna().astype(str).tolist()
    combined = "\n\n".join(conclusions)
    all_conclusions.append(combined)
df_mri_summary["P11_custom_all_conclusions"] = all_conclusions

# Outcome
print("Outcome:\t\t", end="")
# Remove redundant columns
remove_prefixes = ["P23", "P10Out04", "P10Out05", "P10Out08", "P10Out09", "P10Out10", "P10Out11", "P10Out17"]
for col in df_outcome.columns:
    if any(col.startswith(prefix) for prefix in remove_prefixes):
        df_outcome = df_outcome.drop(columns=[col])
# Remove rows with "unknown" outcome entries (seizure free & Engel & ILAE)
df_outcome["P10Out02"] = df_outcome["P10Out02"].fillna(666)
df_outcome["P10Out12"] = df_outcome["P10Out12"].fillna(666)
df_outcome["P10Out13"] = df_outcome["P10Out13"].fillna(666)
df_outcome = df_outcome[~(
    (df_outcome["P10Out02"].astype(int) == 666)      # Unknown seizure freedom
    & (df_outcome["P10Out12"].astype(int) == 666)    # Unknown Engel
    & (df_outcome["P10Out13"].astype(int) == 666)    # Unknown ILAE
)].reset_index(drop=True)
# Remove multiple outcome entries per patient, keep only the last by date
# Convert date column to datetime
df_outcome["P10Out01"] = pd.to_datetime(df_outcome["P10Out01"], format="%d-%m-%Y", errors="coerce")
df_outcome["P10Out000"] = pd.to_datetime(df_outcome["P10Out000"], format="%d-%m-%Y", errors="coerce")
# Sort by date and keep last entry per patient
df_outcome = df_outcome.sort_values(by=["Participant Id", "P10Out01"]).groupby("Participant Id", as_index=False).last().reset_index(drop=True)
# Calculate total follow-up time in months
df_outcome["P10custom_followup_months"] = (
    (df_outcome["P10Out01"] - df_outcome["P10Out000"]).dt.days / 30.44
)
print("\033[92mOK\033[0m")

# Demographics
print("Demographics:\t\t", end="")
# Keep only P0 and P2 columns
cols_to_keep = [col for col in df_demo.columns if col.startswith("P0D")]
df_demo = df_demo[["Participant Id"] + cols_to_keep]
print("\033[92mOK\033[0m")

# Seizure type
print("Seizure type:\t\t", end="")
# Keep only relevant columns
cols_to_keep = [col for col in df_seizure.columns if col.startswith("P3Seiz")]
df_seizure = df_seizure[["Participant Id"] + cols_to_keep]
df_seizure = df_seizure.sort_values(by="Participant Id").reset_index(drop=True)
# Create seizure type summary df for merging
df_seizure_summary = df_seizure.copy()
df_seizure_summary["P3custom_all_types"] = df_seizure_summary["Participant Id"].map(
    lambda pid: "\n".join(
        df_seizure[
            (df_seizure["Participant Id"] == pid) & (df_seizure["P3Seiz12"] == 1)
        ]["P3Seiz13"].dropna().astype(str)
    )
)
df_seizure_summary = df_seizure_summary[["Participant Id", "P3custom_all_types"]]
df_seizure_summary = df_seizure_summary.drop_duplicates(subset=["Participant Id"]).reset_index(drop=True)
print("\033[92mOK\033[0m")

# Merge dataframes into single summary dataframe
print("\nMerging dataframes:\t", end="")
df_merge = df_pathology.merge(df_outcome, on="Participant Id", how="left")
df_merge = df_merge.merge(df_demo, on="Participant Id", how="left")
df_merge = df_merge.merge(df_surgery_summary, on="Participant Id", how="left")
df_merge = df_merge.merge(df_mri_summary, on="Participant Id", how="left")
df_merge = df_merge.merge(df_seizure_summary, on="Participant Id", how="left")
# Drop redundant columns
remove_prefixes = ["Participant Status", "Repeating Data Creation Date", "Repeating data Name Custom", "Repeating data Parent"]
cols_to_drop = [col for col in df_merge.columns if any(col.startswith(prefix) for prefix in remove_prefixes)]
df_merge = df_merge.drop(columns=cols_to_drop)
print("\033[92mOK\033[0m")

# Save results
print("\nSaving results...")

# Plot entry density to check whether periods of data collection are missing, save figure
dataframes_to_plot = [
    ("Pathology", df_pathology),
    ("Surgery", df_surgery),
    ("MRI", df_mri),
    ("Outcome", df_outcome)
]
# dataframes_to_plot = [
#     ("Pathology", dataframes.get("RESPectDB_Pathology")),
#     ("Surgery", dataframes.get("RESPectDB_Brain_Surgery_after_08")),
#     ("MRI", dataframes.get("RESPectDB_Structural_MRI")),
#     ("Outcome", dataframes.get("RESPectDB_Seizure_Outcome"))
# ]
fig = plot_patient_entry_density(dataframes_to_plot, show=show_plot)
fig.savefig(os.path.join(export_dir, "entry_density.png"), dpi=300)

# Write to csv
df_pathology.to_csv(os.path.join(export_dir, "selected_pathology.csv"), index=False, date_format='%d-%m-%Y')
df_surgery.to_csv(os.path.join(export_dir, "selected_surgery.csv"), index=False, date_format='%d-%m-%Y')
df_mri.to_csv(os.path.join(export_dir, "selected_mri.csv"), index=False, date_format='%d-%m-%Y')
df_outcome.to_csv(os.path.join(export_dir, "selected_outcome.csv"), index=False, date_format='%d-%m-%Y')
df_demo.to_csv(os.path.join(export_dir, "selected_demographics.csv"), index=False, date_format='%d-%m-%Y')
df_seizure.to_csv(os.path.join(export_dir, "selected_seizure_type.csv"), index=False, date_format='%d-%m-%Y')
df_merge.to_csv(os.path.join(export_dir, "selected_summary.csv"), index=False, date_format='%d-%m-%Y')
df_dropped.to_csv(os.path.join(export_dir, "dropped_patients.csv"), index=False, date_format='%d-%m-%Y')

# Write patient id lists
selected_ids_path = os.path.join(export_dir, "selected_patient_ids.txt")
excluded_ids_path = os.path.join(export_dir, "excluded_patient_ids.txt")
with open(selected_ids_path, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(dict.fromkeys(patient_ids))) + "\n")
with open(excluded_ids_path, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(dict.fromkeys(manual_remove_ids))) + "\n")

# Final message
print(f"Saved results to {os.path.abspath(export_dir)}\n")