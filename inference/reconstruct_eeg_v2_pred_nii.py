"""Reconstruct MNI EEG v2 priors from predicted electrode weights.

Uses the same 21 precomputed sensitivity maps as gt_channels.json.
Each NIfTI is the probability-weighted sum of electrode sensitivity fields.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def reconstruct(predictions_path, sensitivity_dir, output_dir):
    with open(predictions_path) as f:
        predictions = json.load(f)  # {patient_id: {channel: probability}}
    if not predictions:
        raise ValueError('No predictions found.')

    channels = list(next(iter(predictions.values())))
    images = [nib.load(str(Path(sensitivity_dir) /
                           f'sensitivity_map_MNI_{ch}.nii.gz')) for ch in channels]
    reference = images[0]
    for img in images[1:]:
        if img.shape != reference.shape or not np.allclose(img.affine, reference.affine):
            raise ValueError('Sensitivity maps must have identical shape and affine.')
    fields = np.stack([img.get_fdata(dtype=np.float32) for img in images])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pid, channel_weights in predictions.items():
        weights = np.array([channel_weights[ch] for ch in channels], dtype=np.float32)
        volume = np.einsum('c,cxyz->xyz', weights, fields)
        nii = nib.Nifti1Image(volume, reference.affine, reference.header.copy())
        nii.set_data_dtype(np.float32)
        path = output_dir / f'{pid}_eeg_v2_prior.nii.gz'
        nib.save(nii, path)
        print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True,
                        help='test_predictions.json from the v2 trainer')
    parser.add_argument('--sensitivity_dir', required=True,
                        help='Directory containing sensitivity_map_MNI_<channel>.nii.gz')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    reconstruct(args.predictions, args.sensitivity_dir, args.output_dir)
