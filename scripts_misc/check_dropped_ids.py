import os
import pandas as pd

# check_dropped_ids.py
# Author: Sjors
# Description: Script to quickly check whether dropped patients are not
#              good candidates for inclusion.

# Highlight keywords for quicker visual check
def highlight_keywords(text, keywords=None):
    if keywords is None:
        return text
    if type(text) is not str:
        return text
    for kw in keywords:
        text = text.replace(kw, f"**{kw.upper()}**")
    return text

# Define path 
csv_path = os.path.join("..", "data", "selection", "dropped_patients.csv")

# Load into dataframe
df = pd.read_csv(csv_path)

# For non-FCD patients, check reports for some keywords
non_fcd = df[df["reason"] == "non-FCD"]
yes_keywords = ["FCD", "focale corticale dysplasie"]
no_keywords = ["(non-FCD)"]
# Match
matching_ids_yes = non_fcd[non_fcd["report"].str.contains('|'.join(yes_keywords), case=False, na=False)]["Participant Id"].tolist()
matching_ids_no = non_fcd[non_fcd["report"].str.contains('|'.join(no_keywords), case=False, na=False)]["Participant Id"].tolist()
matching_ids = [id for id in matching_ids_yes if id not in matching_ids_no]
# Create a dataframe with matching ids and highlighted keywords
check_df = df[df["Participant Id"].isin(matching_ids)]
check_df["report"] = check_df["report"].apply(highlight_keywords, keywords=yes_keywords)

# For TS patients, check reports for some keywords
ts = df[df["reason"] == "tuberous sclerosis"]
ts_keywords = ["TS", "tubereuze sclerose", "tuberous sclerosis", "tuber"]
ts["report"] = ts["report"].apply(highlight_keywords, keywords=ts_keywords)
check_df = pd.concat([check_df, ts], ignore_index=True)

# Add entries with specified reasons to check_df
additional_reasons = [
    "duplicate pathology report",
    "multiple dual pathology reports",
    "dual pathology in single report",
    "manual removal"
]
additional_df = df[df["reason"].isin(additional_reasons)]
additional_df["report"] = additional_df["report"].apply(highlight_keywords, keywords=yes_keywords)
check_df = pd.concat([check_df, additional_df], ignore_index=True)

# Save to CSV
check_df.to_csv(os.path.join("..", "data", "check_dropped_patients.csv"), index=False)