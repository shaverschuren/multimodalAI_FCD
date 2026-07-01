import os
import argparse
import nibabel as nib
import numpy as np

def load_aparc_aseg(mgz_file):
    """Load a FreeSurfer MGZ file."""
    img = nib.load(mgz_file)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def load_color_lut(lut_file=None):
    """Load FreeSurferColorLUT.txt and return a dict {label_name: label_value}."""
    if lut_file is None:
        lut_file = os.path.join(os.environ.get('FREESURFER_HOME', ''), 'FreeSurferColorLUT.txt')
    if not os.path.exists(lut_file):
        raise FileNotFoundError(f"FreeSurferColorLUT.txt not found at {lut_file}")

    lut = {}
    with open(lut_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 2:
                value = int(parts[0])
                name = parts[1]
                lut[name] = value
    return lut

def labels_to_mask(data, label_values):
    """Return a binary mask for the given list of label values."""
    mask = np.zeros(data.shape, dtype=np.uint8)
    for val in label_values:
        mask[data == val] = 1
    return mask

def save_mask(mask, affine, out_file):
    """Save a binary mask as a NIfTI file."""
    nii_img = nib.Nifti1Image(mask, affine)
    nib.save(nii_img, out_file)
    print(f"Saved mask to {out_file}")

def main(mgz_file, labels, out_file, lut_file=None):
    data, affine = load_aparc_aseg(mgz_file)
    lut = load_color_lut(lut_file)

    label_values = []
    for label in labels:
        if label not in lut:
            raise ValueError(f"Label '{label}' not found in LUT.")
        label_values.append(lut[label])

    mask = labels_to_mask(data, label_values)
    save_mask(mask, affine, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract label mask(s) from FreeSurfer aparc+aseg.mgz by name")
    parser.add_argument("mgz_file", help="Path to aparc+aseg.mgz")
    parser.add_argument("labels", nargs='+', help="Label name(s) to extract (e.g., 'Left-Hippocampus')")
    parser.add_argument("out_file", help="Output .nii.gz filename")
    parser.add_argument("--lut_file", default=None, help="Optional path to FreeSurferColorLUT.txt")
    args = parser.parse_args()

    main(args.mgz_file, args.labels, args.out_file, args.lut_file)
    