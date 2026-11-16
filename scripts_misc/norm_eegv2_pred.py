from pathlib import Path
import nibabel as nib
import numpy as np

DIR = Path(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\tmp\eeg_spike_v2_29305275_fold1\eeg_v2_fold1_pairwise_depth_20261116-140506\prior_niftis")

for path in DIR.glob("*.nii.gz"):
    if "_visualisation" in path.name:
        continue
    img = nib.load(path)
    data = img.get_fdata()
    data = (data - data.min()) / max(np.ptp(data), 1e-12)
    out = path.with_name(path.name.replace(".nii.gz", "_visualisation.nii.gz"))
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), out)