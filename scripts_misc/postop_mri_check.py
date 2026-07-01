import os
from pathlib import Path

# Define the base directories
post_operative_dir = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_mri\\post_operative")
masks_dir = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\masks_postop_mri")

# Get all subject subdirectories in post_operative folder
subject_ids = [d.name for d in post_operative_dir.iterdir() if d.is_dir()]

# Check for missing resection masks
missing_masks = []
for subj_id in subject_ids:
    mask_file = masks_dir / f"{subj_id}_resection_mask.nii.gz"
    if not mask_file.exists():
        missing_masks.append(subj_id)

# Report results
print(f"Total subjects in post_operative: {len(subject_ids)}")
print(f"Subjects with resection masks: {len(subject_ids) - len(missing_masks)}")
print(f"Subjects missing resection masks: {len(missing_masks)}")

if missing_masks:
    print("\nMissing resection masks for:")
    for subj_id in missing_masks:
        print(f"  - {subj_id}")
else:
    print("\nAll subjects have corresponding resection masks!")