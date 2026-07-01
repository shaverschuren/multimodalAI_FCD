from pathlib import Path
import pandas as pd

# Input/output files
subjects_file = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\multimodal_test_subjects.txt")
baseline_tsv = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\baseline_manual.tsv")
out_pos_file = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\multimodal_test_subjects_mri_pos.txt")
out_neg_file = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\selection\multimodal_test_subjects_mri_neg.txt")

# Read RESP ids from txt
with subjects_file.open("r", encoding="utf-8") as f:
    resp_ids = [line.strip() for line in f if line.strip()]

# Read baseline TSV
df = pd.read_csv(baseline_tsv, sep="\t", dtype=str)

id_col = "participant_id"
mri_col = "mri_presurgical"

# Normalize TSV ids like sub-RESP1234 to RESP1234 so they match the text file.
normalized_id_col = "participant_id_normalized"
df[normalized_id_col] = df[id_col].str.replace(r"^sub-", "", regex=True)

# Filter to requested subjects with MRI-positive status
mri_pos_ids = (
    df.loc[
        df[normalized_id_col].isin(resp_ids)
        & df[mri_col].str.strip().str.lower().eq("positive"),
        normalized_id_col,
    ]
    .drop_duplicates()
    .tolist()
)
mri_neg_ids = (
    df.loc[
        df[normalized_id_col].isin(resp_ids)
        & df[mri_col].str.strip().str.lower().eq("negative"),
        normalized_id_col,
    ]
    .drop_duplicates()
    .tolist()
)

# Preserve original order from multimodal_test_subjects.txt
mri_pos_set = set(mri_pos_ids)
ordered_mri_pos_ids = [resp_id for resp_id in resp_ids if resp_id in mri_pos_set]

mri_neg_set = set(mri_neg_ids)
ordered_mri_neg_ids = [resp_id for resp_id in resp_ids if resp_id in mri_neg_set]

# Write newline-separated output
out_pos_file.write_text("\n".join(ordered_mri_pos_ids) + "\n", encoding="utf-8")
out_neg_file.write_text("\n".join(ordered_mri_neg_ids) + "\n", encoding="utf-8")

print(f"Wrote {len(ordered_mri_pos_ids)} MRI-positive subjects to {out_pos_file}")
print(f"Wrote {len(ordered_mri_neg_ids)} MRI-negative subjects to {out_neg_file}")