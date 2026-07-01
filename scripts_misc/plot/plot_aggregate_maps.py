"""
Lightweight script to plot aggregate maps using glass brain visualization.
"""

import numpy as np
from pathlib import Path
from nilearn import plotting, datasets
import nibabel as nib
from nibabel.processing import resample_from_to
from matplotlib.colors import LinearSegmentedColormap

ylgnbu_magenta = LinearSegmentedColormap.from_list(
    "YlGnBuMagenta",
    [
        (0.00, "#ecf8b7"),  # pale yellow
        (0.15, "#d4f1ac"),  # pale green
        #(0.20, "#a1d99b"),  # light green
        (0.40, "#41b6c4"),  # turquoise
        (0.55, "#2c7fb8"),  # blue
        (0.75, "#253494"),  # deep blue
        (1.00, "#d01c8b"),  # vivid magenta
    ],
    N=256,
)

# ============================================================================
# Configuration - paths and parameters
# ============================================================================

INPUT_FILE = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\results\aggregate_niftis\eeg\aggregate_prediction_sum.nii.gz")
OUTPUT_FILE = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\results\aggregate_niftis\eeg\aggregate_prediction_sum.png")

# Plotting parameters
DISPLAY_MODE = 'lyrz'
COLORBAR = True
CMAP = ylgnbu_magenta  # Colormap for the overlay
DPI = 150
THRESHOLD = 0.0  # Threshold for displaying the overlay (set to 0 to show all non-zero voxels)
VMIN = 0
VMAX = 16.25
# True voxel-space flips (0=x, 1=y, 2=z). Use (0,) to swap left-right.
FLIP_VOXEL_AXES = (0,)

# Create output directory if it doesn't exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Main plotting
# ============================================================================


def plot_aggregate_maps():
    """Plot aggregate maps from evaluate_maps using glass brain visualization."""
    
    # Load the image
    img = nib.load(INPUT_FILE)
    # Load MNI template for masking (mostly the EEG which isn't masked)
    mni_template = datasets.load_mni152_template()

    # Mask img with mni_template
    img_data = img.get_fdata()
    mni_resampled = resample_from_to(mni_template, img)
    mni_data = mni_resampled.get_fdata()
    masked_data = img_data * (mni_data > 1e-6)

    # Apply a real voxel-array flip after masking to force hemispheric swap.
    for axis in FLIP_VOXEL_AXES:
        masked_data = np.flip(masked_data, axis=int(axis))

    img = nib.Nifti1Image(masked_data, img.affine, img.header)
    
    # Create glass brain plot
    fig = plotting.plot_glass_brain(
        img,
        display_mode=DISPLAY_MODE,
        colorbar=COLORBAR,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        threshold=THRESHOLD,
        title=INPUT_FILE.stem
    )
    
    # Save figure
    fig.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_aggregate_maps()
