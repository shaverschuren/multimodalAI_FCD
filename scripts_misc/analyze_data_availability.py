import os
import pandas as pd
import tqdm
import glob
from PIL import Image
import matplotlib.pyplot as plt

# analyze_data_availability.py
# Author: Sjors
# Description: Script to analyze data availability across patients
#              and generate summary statistics to see which data is missing.
#              Also provides dates for all data to check whether timeline makes sense.

# Options
plot_timelines = True  # Whether to plot individual timelines for each patient

# Setup paths
selection_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\selection"
ria_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\mri\\ria_pull"
ieeg_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\ieeg"
eeg_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\eeg\\persyst"
eeg_meta_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\eeg\\block_data"
fs_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_fs"
fastfs_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_fs"
resection_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\masks_postop_mri"
aecog_pics_root = "L:\\Respect-Leijten\\0_Reports\\Pictures_aECoG"
aecog_pics_root_new = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_ECoG_pictures"
output_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\data_availability"

# Read csv's
summary_df = pd.read_csv(os.path.join(selection_root, "selected_summary.csv"))
surgery_df = pd.read_csv(os.path.join(selection_root, "selected_surgery.csv"))
pathology_df = pd.read_csv(os.path.join(selection_root, "selected_pathology.csv"))
mri_pre_df = pd.read_csv(os.path.join(ria_root, "pre_op_scans_manual_select_final.csv"), sep=";")
mri_post_df = pd.read_csv(os.path.join(ria_root, "post_op_scans_manual_select_final.csv"), sep=";")

# Create some output dirs
timeline_dir = os.path.join(output_dir, "timelines")
os.makedirs(timeline_dir, exist_ok=True)

# Get patient id's
patient_ids = summary_df["Participant Id"].unique().tolist()

# Loop over patients and check available data
print(f"Analyzing data availability for {len(patient_ids)} patients...")
availabilities = []
for patient_id in tqdm.tqdm(patient_ids):
    # Get data from csv's
    summary_data = summary_df[summary_df["Participant Id"] == patient_id]
    surgery_data = surgery_df[surgery_df["Participant Id"] == patient_id]
    pathology_data = pathology_df[pathology_df["Participant Id"] == patient_id]
    mri_pre_data = mri_pre_df[mri_pre_df["patient_id"] == patient_id]
    mri_post_data = mri_post_df[mri_post_df["patient_id"] == patient_id]

    # Check MRI data
    has_mri_pre = not mri_pre_data.empty
    has_mri_post = not mri_post_data.empty

    # Check FreeSurfer data
    has_fs = os.path.exists(os.path.join(fs_root, patient_id)) or \
        os.path.exists(os.path.join(fastfs_root, patient_id))
    # Check aECoG electrode coordinates
    has_aecog_coords = os.path.exists(os.path.join(fs_root, patient_id, "Electrode coordinates"))
    # Check aECoG pictures
    has_aecog_pics = os.path.exists(os.path.join(aecog_pics_root, patient_id)) or \
        os.path.exists(os.path.join(aecog_pics_root_new, patient_id))
    # Check iEEG data
    has_ieeg_preop = os.path.exists(os.path.join(ieeg_root, f"sub-{patient_id}", "ses-preop"))
    has_ieeg_intraop = os.path.exists(os.path.join(ieeg_root, f"sub-{patient_id}", "ses-intraop"))
    # Check EEG data
    has_eeg_data = os.path.exists(os.path.join(eeg_root, patient_id, f"{patient_id}.edf"))
    # Check resection mask
    has_resection_mask = os.path.exists(os.path.join(resection_root, f"{patient_id}_resection_mask.nii.gz"))
    # Check if any data available for ground truth derivation
    has_some_gt = has_ieeg_intraop or has_ieeg_preop or has_resection_mask or has_aecog_pics
    has_ieeg_intraop_no_res = has_ieeg_intraop and has_aecog_pics and not has_resection_mask
    has_aecog_pics_no_ieeg_no_res = has_aecog_pics and not (has_ieeg_intraop or has_resection_mask)
    has_only_iemu = has_ieeg_preop and not (has_aecog_pics or has_ieeg_intraop or has_resection_mask)
    has_both_res_and_pics = has_resection_mask and has_aecog_pics
    has_pics_no_res = has_aecog_pics and not has_resection_mask
    has_res_no_pics = has_resection_mask and not has_aecog_pics

    # Store for plotting
    availability = {
        "Pre-op T1w + FLAIR": has_mri_pre,
        "Pre-op EEG (SEIN)": has_eeg_data,
        "Any ground truth": has_some_gt,
        "Post-op MRI:\nResection": has_resection_mask,
        # "aECoG data\n+ pictures": has_ieeg_intraop and has_aecog_pics,
        "Intra-op pictures": has_aecog_pics,
        "IEMU data\n+ coords": has_ieeg_preop,
        "FreeSurfer\n/FastSurfer": has_fs,
        "aECoG coords": has_aecog_coords,
        "iEEG intra-op no res": has_ieeg_intraop_no_res,
        "has_aecog_pics_no_ieeg_no_res": has_aecog_pics_no_ieeg_no_res,
        "has_both_res_and_pics": has_both_res_and_pics,
        "has_pics_no_res": has_pics_no_res,
        "has_res_no_pics": has_res_no_pics,
        # "has_only_iemu": has_only_iemu
    }
    availabilities.append((patient_id, availability))

    # Get EEG metadata if available
    eeg_meta_file = os.path.join(eeg_meta_root, f"{patient_id}_trc2edf_conversion.tsv")
    if os.path.exists(eeg_meta_file):
        eeg_meta_df = pd.read_csv(eeg_meta_file, sep="\t")
    else:
        eeg_meta_df = pd.DataFrame()

    # Collect dates for each data type
    surgery_dates = pd.to_datetime(surgery_data["P4EpSG01"], errors="coerce", format="%d-%m-%Y").tolist() \
        if not surgery_data.empty else []
    pathology_dates = pd.to_datetime(pathology_data["P9Path01"], errors="coerce", format="%d-%m-%Y").tolist() \
        if not pathology_data.empty else []
    mri_pre_dates = pd.to_datetime(mri_pre_data["StudyDate"], errors="coerce", format="%d-%m-%Y").tolist() \
        if not mri_pre_data.empty else []
    mri_post_dates = pd.to_datetime(mri_post_data["StudyDate"], errors="coerce", format="%d-%m-%Y").tolist() \
        if not mri_post_data.empty else []
    eeg_dates = pd.to_datetime(eeg_meta_df["start_absolute_date"], errors="coerce", format="%Y-%m-%d").unique().tolist() \
        if not eeg_meta_df.empty else []

    if plot_timelines:
        # Create timeline
        timeline_events = []
        if surgery_dates:
            for date in surgery_dates:
                timeline_events.append(("Surgery", date))
        if pathology_dates:
            for date in pathology_dates:
                timeline_events.append(("Pathology", date))
        if mri_pre_dates:
            for date in mri_pre_dates:
                timeline_events.append(("MRI Pre-op", date))
        if mri_post_dates:
            for date in mri_post_dates:
                timeline_events.append(("MRI Post-op", date))
        if eeg_dates:
            for date in eeg_dates:
                timeline_events.append(("EEG", date))
        # Sort timeline
        timeline_events = sorted(timeline_events, key=lambda x: x[1])

        # "weird-order" flag
        has_weird_order = None
        if not all([surgery_dates, pathology_dates, mri_pre_dates, eeg_dates]):
            # Missing essential data
            has_weird_order = "Missing dates"
        else:
            if len(mri_pre_dates) > 1:
                # Pre-op MRI's on different dates
                if mri_pre_dates[0] != mri_pre_dates[-1]:
                    has_weird_order = "Multiple-session pre-op MRIs"
            if mri_post_dates:
                if any(date > mri_post_dates[0] for date in surgery_dates + pathology_dates + mri_pre_dates):
                    # Post-op MRI not last
                    has_weird_order = "Post-op MRI not last"
            if any(date > min(pathology_dates) or date > min(surgery_dates) for date in mri_pre_dates):
                # Pre-op MRI after pathology or surgery
                has_weird_order = "Pre-op MRI after pathology or surgery"
            if any(date > min(pathology_dates) or date > min(surgery_dates) for date in eeg_dates):
                # EEG after pathology or surgery
                has_weird_order = "EEG after pathology or surgery"

        # Plot timeline with vertical lines for each event type
        if timeline_events:
            fig, ax = plt.subplots(figsize=(8, 2))
            event_labels = [event[0] for event in timeline_events]
            event_dates = [event[1] for event in timeline_events]
            # Define colors for each event type
            event_colors = {
                "Surgery": "red",
                "Pathology": "orange",
                "MRI Pre-op": "green",
                "MRI Post-op": "blue",
                "EEG": "purple"
            }
            # Plot each event as a vertical line
            for label, date in zip(event_labels, event_dates):
                ax.axvline(date, color=event_colors.get(label, "gray"), linestyle='-', linewidth=3, alpha=0.7)
                ax.text(date, 1.025, label, rotation=90, ha="right", va="center", fontsize=10, color=event_colors.get(label, "gray"))
            # Add grid of vertical lines for each month, thick line for each year
            min_date = min(event_dates)
            max_date = max(event_dates)
            # Generate monthly ticks
            months = pd.date_range(min_date.replace(day=1), max_date, freq='MS')
            for month in months:
                ax.axvline(month, color='lightgray', linestyle='--', linewidth=1, zorder=0)

            ax.set_xlim(min_date - pd.Timedelta(days=15), max_date + pd.Timedelta(days=15))
            ax.set_ylim(0.95, 1.1)
            ax.set_yticks([])
            ax.set_title(f"{patient_id}")
            plt.tight_layout()
            # If weird order, make title and bounding box red/orange
            if has_weird_order:
                # Define colors
                if has_weird_order == "Missing dates":
                    flag_color = "red"
                elif has_weird_order == "Pre-op MRI after pathology or surgery":
                    flag_color = "red"
                elif has_weird_order == "EEG after pathology or surgery":
                    flag_color = "red"
                elif has_weird_order == "Post-op MRI not last":
                    flag_color = "orange"
                elif has_weird_order == "Multiple-session pre-op MRIs":
                    flag_color = "orange"
                else:
                    flag_color = "red"
                # Set colors
                ax.set_title(f"{patient_id} - {has_weird_order}", color=flag_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(flag_color)
                    spine.set_linewidth(2)
            else:
                ax.set_title(f"{patient_id}")
            # Save individual timeline image
            img_path = os.path.join(timeline_dir, f"{patient_id}_timeline.png")
            plt.savefig(img_path)
            plt.close()

# Create availability summary plot
# Prepare data for summary plot
availability_df = pd.DataFrame([{"Patient Id": pid, **avail} for pid, avail in availabilities])
availability_counts = availability_df.drop("Patient Id", axis=1).sum()
total_patients = len(availability_df)

# Plot bar chart
plt.figure(figsize=(10, 5))
bars = availability_counts[0:-6].plot(kind="bar", color="skyblue", edgecolor="black")
# Add gray bar on top of each bar, summing to total number of patients
colors = ["skyblue", "skyblue", "lavender", "mediumaquamarine", "salmon", "lavender", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue"]
for i, count in enumerate(availability_counts[:-6]):
    bars.bar(i, total_patients, color="lightgray", edgecolor="none", zorder=0)
    bars.bar(i, count, color=colors[i], edgecolor="black", zorder=1)
# Plot stacked bar
if len(availability_counts) >= 3:
    bottom = availability_counts.iloc[-2] + availability_counts.iloc[-1]
    plt.bar(2, availability_counts.iloc[-1], color="mediumaquamarine", edgecolor="black")
    plt.bar(2, availability_counts.iloc[-3], bottom=availability_counts.iloc[-1], color="#A1AE93", edgecolor="black")
    plt.bar(2, availability_counts.iloc[-2], bottom=bottom, color="salmon", edgecolor="black")
# Add vertical lines between bars 1-2 and 5-6
ax = plt.gca()
ax.axvline(1.5, color="black", linestyle="--", linewidth=2)
ax.axvline(5.5, color="black", linestyle="--", linewidth=2)
# Add some text
ax.text(0.5, total_patients - 1, "Inputs", ha="center", fontsize=11, fontweight="bold")
ax.text(3.5, total_patients - 1, "Ground truths", ha="center", fontsize=11, fontweight="bold")
ax.text(6.0, total_patients - 1, "Derivatives", ha="center", fontsize=11, fontweight="bold")
# Labels and title
plt.ylabel("Number of patients")
plt.title("Data Availability Overview", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "availability_summary.png"))
plt.close()

# Combine all timeline images into one big image
if plot_timelines:
    timeline_imgs = sorted(glob.glob(os.path.join(timeline_dir, "*_timeline.png")))
    if timeline_imgs:
        # Determine grid size (e.g., 5 columns)
        n_cols = 5
        n_rows = (len(timeline_imgs) + n_cols - 1) // n_cols
        # Open all images and resize to same size
        imgs = [Image.open(img).resize((800, 200)) for img in timeline_imgs]
        width, height = imgs[0].size
        # Create blank canvas
        big_img = Image.new('RGB', (n_cols * width, n_rows * height), (255, 255, 255))
        for idx, img in enumerate(imgs):
            row, col = divmod(idx, n_cols)
            big_img.paste(img, (col * width, row * height))

        big_img.save(os.path.join(output_dir, "timelines_visual.png"))