from normalize_gado_scan import process_nifti
import os
from glob import glob

def normalize_gado_scans_in_directory(input_dir, output_dir, hi_percentile=95.0):
    """Normalize all NIfTI files in the input directory and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    nifti_files = glob(os.path.join(input_dir, '*.nii')) + glob(os.path.join(input_dir, '*.nii.gz'))

    for nifti_file in nifti_files:
        subj_id = os.path.basename(nifti_file).split('-')[0]
        os.makedirs(os.path.join(output_dir, subj_id), exist_ok=True)
        out_file = os.path.join(output_dir, subj_id, subj_id + '_T1w.nii')
        print(f"Normalizing {nifti_file} -> {out_file}")
        process_nifti(nifti_file, out_file, hi_percentile=hi_percentile)

if __name__ == "__main__":

    in_dir = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\gado_scans'

    # out_dir_95 = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_for_fastsurfer\\clip95'
    # normalize_gado_scans_in_directory(in_dir, out_dir_95, hi_percentile=95.0)

    # out_dir_975 = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_for_fastsurfer\\clip975'
    # normalize_gado_scans_in_directory(in_dir, out_dir_975, hi_percentile=97.5)

    out_dir_99 = 'L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\dataset_for_fastsurfer\\clip99'
    normalize_gado_scans_in_directory(in_dir, out_dir_99, hi_percentile=99.0)
