"""Reconstruct MNI EEG v2 priors from predicted electrode weights.

Uses the same 21 precomputed sensitivity maps as gt_channels.json.
Each NIfTI is the probability-weighted sum of electrode sensitivity fields.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from datasets.eeg import (CHANNEL_LABEL_TO_INT, DEPTH_BIN_NAMES,
                          DISTANCE_BIN_NAMES, assert_same_grid,
                          compute_depth_map_and_bins,
                          compute_electrode_distance_bins, load_brain_template,
                          resolve_sensitivity_maps)
from inference.reconstruct_eeg_v2_depth_oracle_nii import (
    component_normalizers, load_targets, reconstruct_oracle)

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
EEG_DIR = Path(__file__).resolve().parents[1] / 'preprocessing' / 'eeg'


def reconstruct(predictions_path, sensitivity_dir, output_dir):
    with open(predictions_path, encoding='utf-8') as f:
        raw = json.load(f)
    if not raw:
        raise ValueError('No predictions found.')
    example = next(iter(next(iter(raw.values())).values()))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_paths = resolve_sensitivity_maps(sensitivity_dir, CHANNELS)
    reference = nib.load(str(map_paths[CHANNELS[0]]))

    if not isinstance(example, dict):
        images = [nib.load(str(map_paths[ch])) for ch in CHANNELS]
        for channel, image in zip(CHANNELS, images):
            assert_same_grid(image, reference, channel)
        fields = np.stack([img.get_fdata(dtype=np.float32) for img in images])
        predictions = {
            pid: np.array([weights.get(ch, 0.) for ch in CHANNELS],
                          dtype=np.float32)
            for pid, weights in raw.items()
        }
        for pid, weights in predictions.items():
            volume = np.einsum('c,cxyz->xyz', weights, fields)
            save_volume(volume, reference, output_dir, pid)
        return

    if set(example) == set(DEPTH_BIN_NAMES):
        bins = DEPTH_BIN_NAMES
        mode = 'depth'
    elif set(example) == set(DISTANCE_BIN_NAMES):
        bins = DISTANCE_BIN_NAMES
        mode = 'distance'
    else:
        raise ValueError(f'Unknown prediction bins: {set(example)}')

    predictions = load_targets(predictions_path, bins)
    template, brain_mask = load_brain_template(
        EEG_DIR / 'MNI152_T1_1mm_brain.nii.gz')
    assert_same_grid(reference, template, 'Sensitivity-map reference')

    if mode == 'depth':
        _, bin_maps = compute_depth_map_and_bins(brain_mask, template.affine)
    else:
        with (EEG_DIR / 'standard_1005_mni_coordinates.json').open() as f:
            data = json.load(f)['data']
        coords = {ch: data[ch]['mni'] for ch in CHANNELS}

        def bin_maps(channel):
            return compute_electrode_distance_bins(
                coords[channel], brain_mask, template.affine)[1]

    normalizers = component_normalizers(map_paths, reference, bin_maps)
    for pid, target in predictions.items():
        volume = reconstruct_oracle(target, map_paths, reference, bin_maps,
                                    normalizers, brain_mask)
        save_volume(volume, reference, output_dir, pid)
    print(f'Reconstructed {len(predictions)} {mode} predictions in {output_dir}')


def save_volume(volume, reference, output_dir, patient):
    nii = nib.Nifti1Image(volume.astype(np.float32), reference.affine,
                          reference.header.copy())
    nii.set_data_dtype(np.float32)
    path = output_dir / f'{patient}_eeg_v2_prior.nii.gz'
    nib.save(nii, path)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--sensitivity_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    reconstruct(args.predictions, args.sensitivity_dir, args.output_dir)
