"""Script to normalize some GADO MRI's so the FastSurfer normalizer doesn't panic."""

import sys
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output

def resample_to_1mm(img):
    return resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0))

def clip_intensities(data, hi_percentile=95.0):
    hi = np.percentile(data, hi_percentile)
    lo = hi * 0.005
    return np.clip(data, lo, hi)

def process_nifti(in_path, out_path, hi_percentile=95.0):
    img = nib.load(in_path)
    img_1mm = resample_to_1mm(img)

    data_resampled = img_1mm.get_fdata()
    new_affine = img_1mm.affine
    data_clipped = clip_intensities(data_resampled, hi_percentile=hi_percentile)

    out_img = nib.Nifti1Image(data_clipped.astype(np.float32), new_affine)
    nib.save(out_img, out_path)


if __name__ == "__main__":

    process_nifti(sys.argv[1], sys.argv[2])
