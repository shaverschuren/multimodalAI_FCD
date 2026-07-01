import json
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from nilearn import plotting
import matplotlib.pyplot as plt

def get_qc_results(qc_file_path):
    """
    Read QC results from a text file and return a dictionary mapping patient IDs to QC status.
    
    Args:
        qc_file_path: Path to the QC results text file.
    """
    qc_results = {}
    with open(qc_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            patient_id = parts[0].strip()
            if len(parts) >= 2:
                status = parts[1].strip().lower()
                qc_results[patient_id] = "Failed: " + status if status != "success" else "Success"
            else:
                qc_results[patient_id] = "Success"
    return qc_results

def plot_coverage_map(coverage_nii_path, output_img_path):
    """
    Plots a coverage map using nilearn's glass brain visualization.
    
    Args:
        coverage_nii_path: Path to the NIfTI file containing the coverage map.
        output_img_path: Path to save the output image.
    """
    # Compute upper limit for color scale based on positive finite values
    coverage_img = nib.load(coverage_nii_path)
    coverage_data = coverage_img.get_fdata()
    positive_values = coverage_data[
        np.isfinite(coverage_data) & (coverage_data > 0)
    ]

    if positive_values.size == 0:
        raise ValueError(
            f"No positive finite coverage values found in: {coverage_nii_path}"
        )

    # Use the actual maximum for integer/count coverage maps.
    vmax = positive_values.max()

    # Set figure
    fig = plt.figure(figsize=(12, 3.2), dpi=300)
    # Plot
    display = plotting.plot_glass_brain(
        stat_map_img=str(coverage_nii_path),
        display_mode="lyrz",
        figure=fig,
        black_bg=False,
        annotate=True,
        colorbar=True,
        cmap="YlGnBu",
        threshold=0,
        vmin=0,
        vmax=vmax,
        plot_abs=False,
        title=None,
    )
    # Save
    display.savefig(
        output_img_path,
        dpi=300,
        bbox_inches="tight",
        # facecolor="white",
    )
    display.close()

    print(f"Saved: {output_img_path}")

def plot_region_level_coverage(region_dict, output_path=None, top_k=12):
    """
    Creates a side-by-side (intertwined) bar chart comparing:
        - TOP region counts (how many subjects contain a region)
        - Soft counts (sum of fractional coverage values)

    Args:
        region_dict: dict of dicts
        output_path: optional; if given, saves the plot to this path
        top_k: optional; keep only the K highest soft-count regions
    """

    print("Computing region-level cumulative coverage...")

    soft_counts = {}
    top_counts = {}  # count of subjects where region appears

    # accumulate
    for subj, regions in region_dict.items():
        for region, value in regions.items():
            soft_counts[region] = soft_counts.get(region, 0) + value

        top_region = max(regions, key=regions.get)
        top_counts[top_region] = top_counts.get(top_region, 0) + 1

    # sort by soft counts
    regions_sorted = sorted(soft_counts, key=soft_counts.get, reverse=True)

    # restrict to top_k if requested
    if top_k is not None:
        regions_sorted = regions_sorted[:top_k]

    soft_values = [soft_counts[r] for r in regions_sorted]
    top_values = [top_counts[r] if r in top_counts else 0 for r in regions_sorted]

    x = np.arange(len(regions_sorted))
    width = 0.4  # width of each bar

    plt.figure(figsize=(14, 7))

    # Vertical intertwined bars
    plt.bar(x - width/2, soft_values, width=width, color="skyblue", edgecolor="black", label="Relative coverage sum")
    plt.bar(x + width/2, top_values,  width=width, color="lightcoral", edgecolor="black", label="Top region coverage sum")

    plt.xticks(x, regions_sorted, rotation=45, ha='right')
    plt.ylabel("Value")
    plt.title("Region-Level Ground-Truth Coverage")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

    print("Region-level intertwined bar plot generated.")

def check_gt_coverage(mask_paths, output_path, qc_results):
    """
    Read ground truth masks and create a coverage map by summing all masks.
    
    Args:
        mask_paths: List of paths to .nii.gz mask files
        output_path: Path for output coverage map
        qc_results: Dictionary mapping patient IDs to QC status
    """
    if not mask_paths:
        raise ValueError("No mask paths provided")
    print(f"Found {len(mask_paths)} ground truth masks. Computing coverage...")

    # Load first mask to get dimensions and affine
    first_mask = nib.load(mask_paths[0])
    coverage_array = np.zeros_like(first_mask.get_fdata())
    
    # Sum all masks
    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        subj_id = os.path.dirname(mask_path).split(os.sep)[-1]
        qc_status = qc_results.get(subj_id, "Unknown")
        if qc_status != "Success":
            tqdm.write(f"\033[38;5;208mSkipping {subj_id} due to QC status: {qc_status}\033[0m")
            continue
        try:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            coverage_array += mask_data
        except Exception as e:
            tqdm.write(f"\033[38;5;208mError processing {mask_path}: {e}\033[0m")
    
    # Create new NIfTI image with coverage data
    coverage_img = nib.Nifti1Image(coverage_array, first_mask.affine, first_mask.header)
    
    # Save coverage map
    nib.save(coverage_img, output_path)
    print(f"Coverage map saved to: {output_path}")
    print(f"Max coverage value: {np.max(coverage_array)}")

if __name__ == "__main__":
    
    # Set paths
    gt_mask_directory = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri"
    gt_label_json = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\gt\\gt_lobe.json"
    output_nii_path = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\data_availability\\gt_coverage_map.nii.gz"
    output_img_map = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\data_availability\\gt_coverage_map.png"
    output_region_img_path = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\data_availability\\gt_region_coverage.png"
    qc_file_path = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri\\QC_images\\success_subjects.txt"

    # Get masks
    mask_paths = list(glob(os.path.join(gt_mask_directory, "RESP*", "RESP*_gt_norm.nii.gz")))
    # Check for QC results
    qc_results = get_qc_results(qc_file_path)
    # Run coverage check (voxel-level)
    check_gt_coverage(mask_paths, output_nii_path, qc_results)
    # Plot coverage with nilearn glass brain
    plot_coverage_map(output_nii_path, output_img_map)
    # Run coverage check (region-level)
    with open(gt_label_json, 'r') as f:
        region_gt_dict = json.load(f)
    plot_region_level_coverage(region_gt_dict, output_region_img_path)