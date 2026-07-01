import os
from pathlib import Path
import pandas as pd

# --- CONFIG ---
SUBJECTS_CSV = Path(r"L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/selection/selected_summary.csv")
BASE_DIR = Path(r"L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/dataset_fs")
POSTOP_DIR = Path(r"L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/masks_postop_mri")
MANUAL_DIR = Path(r"L:/her_knf_golf/Wetenschap/newtransport/Sjors/data/manual_segs")
OUTPUT_FILE = BASE_DIR / ".." / "tmp" / "gt_summary_results.txt"

# --- FILE CHECK DEFINITIONS ---
PIC2MRI_REQUIRED = [r"pic2mri_output/pic2mri_resection_mask.nii.gz"]
SCENE_REQUIRED = [r"pic2mri_output/scene/scene.mrml"]
FREESURFER_REQUIRED = [
    r"surf/lh.pial",
    r"surf/rh.pial",
    r"mri/brainmask.mgz",
    r"mri/T1.mgz",
]
FREESURFER_SEG_REQUIRED = r"mri/aparc.a2009s+aseg.mgz"
PHOTO_SUFFIX = "_photo_with_masks"
RESECTION_PHOTO_SUFFIX = "_resection_photo"

# --- MAIN LOOP ---

summary_df = pd.read_csv(SUBJECTS_CSV)
subjects = sorted(summary_df["Participant Id"].unique().tolist())

results = []
# subjects = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("RESP")])
check_subjects = []
no_fs_subjects = []
no_fs_seg_subjects = []

for subj_id in subjects:
    # subj_id = subj.name
    pic2mri_dir = BASE_DIR / subj_id / "pic2mri_output"
    scene_dir = pic2mri_dir / "scene"

    # --- Checks ---
    has_pic2mri = pic2mri_dir.exists()
    has_resection_mask = (pic2mri_dir / "pic2mri_resection_mask.nii.gz").exists()
    has_photo_with_masks = any(
        f.name.startswith(subj_id + PHOTO_SUFFIX) for f in pic2mri_dir.glob(f"{subj_id}{PHOTO_SUFFIX}*")
    )
    has_scene = (scene_dir / "scene.mrml").exists()
    has_resection_photo = any(
        f.name.startswith(subj_id + RESECTION_PHOTO_SUFFIX)
        for f in pic2mri_dir.glob(f"{subj_id}{RESECTION_PHOTO_SUFFIX}*")
    )
    freesurfer_complete = all((BASE_DIR / subj_id / f).exists() for f in FREESURFER_REQUIRED)
    freesurfer_seg_complete = (BASE_DIR / subj_id / FREESURFER_SEG_REQUIRED).exists()

    # --- Post-op MRI check ---
    postop_mri_complete = (POSTOP_DIR / f"{subj_id}_resection_mask.nii.gz").exists()

    # --- Derived completeness flags ---
    pic2mri_complete = has_pic2mri and has_resection_mask and has_photo_with_masks and has_scene

    # --- Manual segmentations check ---
    manual_seg_complete = (MANUAL_DIR / f"{subj_id}_manual_resection.nii.gz").exists()

    # --- Combined completeness mark ---
    completeness_score = pic2mri_complete + postop_mri_complete + manual_seg_complete
    if completeness_score == 3:
        completeness_mark = "✅✅✅"
    elif completeness_score == 2:
        completeness_mark = "✅✅"
    elif completeness_score == 1:
        completeness_mark = "✅"
    else:
        completeness_mark = "❌"

    results.append({
        "subject": subj_id,
        "mark": completeness_mark,
        "mark_score": completeness_score,
        "postop_mri": "✅" if postop_mri_complete else "❌",
        "pic2mri": "✅" if pic2mri_complete else "❌",
        "freesurfer": "✅" if freesurfer_complete else "❌",
        "freesurfer_seg": "✅" if freesurfer_seg_complete else "❌",
        "resection_photo": "✅" if has_resection_photo else "❌",
        "manual_seg": "✅" if manual_seg_complete else "❌"
    })

    print(f"Processed {subj_id}: pic2mri={pic2mri_complete}, postop_mri={postop_mri_complete}, freesurfer={freesurfer_complete}, manual_seg={manual_seg_complete}")

# --- SORT: incomplete first ---
results.sort(key=lambda r: (-r["mark_score"], r["subject"]), reverse=True)
results.sort(key=lambda r: r["mark_score"])  # ensures ❌ first, ✅✅ last

# --- TABLE HEADER ---
header_cols = ["✔", "Subject", "Post-op MRI", "pic2mri", "Free/FastSurfer", "Free/FastSurfer Seg", "Resection photo", "Manual Seg"]
col_widths = [6, 12, 14, 12, 16, 20, 18, 14]
header_line = " ".join(col.ljust(w) for col, w in zip(header_cols, col_widths))
sep_line = "-" * len(header_line)

# --- TABLE BODY ---
table_lines = []
for r in results:
    row = (
        f"{r['mark'].ljust(6)} "
        f"{r['subject'].ljust(12)} "
        f"{r['postop_mri'].ljust(14)} "
        f"{r['pic2mri'].ljust(12)} "
        f"{r['freesurfer'].ljust(16)} "
        f"{r['freesurfer_seg'].ljust(20)} "
        f"{r['resection_photo'].ljust(18)} "
        f"{r['manual_seg'].ljust(14)}"
    )
    if r['resection_photo'] == "✅" and r["pic2mri"] == "❌":
        row += "\t<--- ⚠️ Photo present but no pic2mri"
        check_subjects.append(r['subject'])
    if r['freesurfer'] == "❌":
        no_fs_subjects.append(r['subject'])
    if r['freesurfer_seg'] == "❌":
        no_fs_seg_subjects.append(r['subject'])

    table_lines.append(row)

# --- SUMMARY STATS ---
n_total = len(results)
n_pic2mri_ok = sum(r["pic2mri"] == "✅" for r in results)
n_postop_ok = sum(r["postop_mri"] == "✅" for r in results)
n_manual_ok = sum(r["manual_seg"] == "✅" for r in results)
n_both_ok = sum(r["mark_score"] == 2 for r in results)
n_none_ok = sum(r["mark_score"] == 0 for r in results)
n_freesurfer_missing = sum(r["freesurfer"] == "❌" for r in results)
n_resection_photo_missing = sum(r["resection_photo"] == "❌" for r in results)
n_freesurfer_seg_missing = sum(r["freesurfer_seg"] == "❌" for r in results)

summary_text = [
    "",
    "=" * len(header_line),
    f"Summary for {n_total} subjects:",
    f"  ✅✅ both complete: {n_both_ok}",
    f"  ✅ only one complete: {n_total - n_both_ok - n_none_ok}",
    f"  ❌ none complete: {n_none_ok}",
    f"  ✅ pic2mri complete: {n_pic2mri_ok}",
    f"  ✅ post-op MRI complete: {n_postop_ok}",
    f"  ✅ manual segmentations complete: {n_manual_ok}",
    f"  ⚠️ Free/FastSurfer missing: {n_freesurfer_missing}",
    f"  ⚠️ Free/FastSurfer Seg missing: {n_freesurfer_seg_missing}",
    f"  ⚠️ Resection photo missing: {n_resection_photo_missing}",
]

# --- WRITE OUTPUT ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Summary for {n_total} subjects in {BASE_DIR}\n")
    f.write(sep_line + "\n")
    f.write(header_line + "\n")
    f.write(sep_line + "\n")
    f.write("\n".join(table_lines))
    f.write("\n" + "\n".join(summary_text))
    f.write("\n" + sep_line + "\n")
    f.write(f"⚠️ Subjects with photos but no pic2mri: {len(check_subjects)}\n" + ", ".join(sorted(check_subjects)))
    f.write("\n" + sep_line + "\n")
    f.write(f"⚠️ Subjects with Free/FastSurfer data missing: {len(no_fs_subjects)}\n" + ", ".join(sorted(no_fs_subjects)))
    f.write("\n" + sep_line + "\n")
    f.write(f"⚠️ Subjects with Free/FastSurfer Seg data missing: {len(no_fs_seg_subjects)}\n" + ", ".join(sorted(no_fs_seg_subjects)))

print(f"\n✅ Summary written to {OUTPUT_FILE}")
