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
    component_normalizers, load_targets, reconstruct_factorized,
    reconstruct_oracle)

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
EEG_DIR = Path(__file__).resolve().parents[1] / 'preprocessing' / 'eeg'


def load_factorized_predictions(raw):
    channel_names = {name.casefold(): index for index, name in enumerate(CHANNELS)}
    depth_names = {name.casefold(): index for index, name in enumerate(DEPTH_BIN_NAMES)}
    predictions = {}
    for patient, entry in raw.items():
        if not isinstance(entry, dict) or set(entry) != {'channels', 'depth'}:
            raise ValueError(f'Expected factorized channels/depth predictions for {patient}.')
        channel = np.zeros(len(CHANNELS), dtype=np.float32)
        for name, value in entry['channels'].items():
            key = name.casefold()
            if key not in channel_names:
                raise ValueError(f'Unknown channel {name!r} for {patient}.')
            channel[channel_names[key]] += float(value)
        depth = np.zeros(len(DEPTH_BIN_NAMES), dtype=np.float32)
        for name, value in entry['depth'].items():
            key = name.casefold()
            if key not in depth_names:
                raise ValueError(f'Unknown depth bin {name!r} for {patient}.')
            depth[depth_names[key]] += float(value)
        predictions[patient] = (channel, depth)
    return predictions


def reconstruct(predictions_path, sensitivity_dir, output_dir, mode='auto'):
    with open(predictions_path, encoding='utf-8') as f:
        raw = json.load(f)
    if not raw:
        raise ValueError('No predictions found.')
    first = next(iter(raw.values()))
    is_factorized = isinstance(first, dict) and set(first) == {'channels', 'depth'}
    if mode == 'factorized_depth' or (mode == 'auto' and is_factorized):
        if not is_factorized:
            raise ValueError('factorized_depth mode requires {channels, depth} predictions.')
        factorized = load_factorized_predictions(raw)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        map_paths = resolve_sensitivity_maps(sensitivity_dir, CHANNELS)
        reference = nib.load(str(map_paths[CHANNELS[0]]))
        template, brain_mask = load_brain_template(
            EEG_DIR / 'MNI152_T1_1mm_brain.nii.gz')
        assert_same_grid(reference, template, 'Sensitivity-map reference')
        _, depth_bins = compute_depth_map_and_bins(brain_mask, template.affine)
        for patient, (channel, depth) in factorized.items():
            volume = reconstruct_factorized(channel, depth, map_paths, reference,
                                            depth_bins, brain_mask)
            save_volume(volume, reference, output_dir, patient)
        print(f'Reconstructed {len(factorized)} factorized-depth predictions in {output_dir}')
        return
    if mode == 'factorized_depth':
        raise ValueError('factorized_depth mode requires factorized predictions.')
    if mode not in {'auto', 'joint'}:
        raise ValueError(f'Unknown reconstruction mode: {mode}')
    if isinstance(first, dict) and first:
        example = next(iter(first.values()))
    else:
        example = first
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_paths = resolve_sensitivity_maps(sensitivity_dir, CHANNELS)
    reference = nib.load(str(map_paths[CHANNELS[0]]))

    if not isinstance(example, dict):
        if mode == 'joint':
            raise ValueError('joint mode requires per-channel bin predictions.')
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
        target_mode = 'depth'
    elif set(example) == set(DISTANCE_BIN_NAMES):
        bins = DISTANCE_BIN_NAMES
        target_mode = 'distance'
    else:
        raise ValueError(f'Unknown prediction bins: {set(example)}')

    predictions = load_targets(predictions_path, bins)
    template, brain_mask = load_brain_template(
        EEG_DIR / 'MNI152_T1_1mm_brain.nii.gz')
    assert_same_grid(reference, template, 'Sensitivity-map reference')

    if target_mode == 'depth':
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
    print(f'Reconstructed {len(predictions)} {target_mode} predictions in {output_dir}')


def save_volume(volume, reference, output_dir, patient):
    volume = volume.astype(np.float32)
    nii = nib.Nifti1Image(volume, reference.affine, reference.header.copy())
    nii.set_data_dtype(np.float32)
    path = output_dir / f'{patient}_eeg_v2_prior.nii.gz'
    nib.save(nii, path)
    print(f'Saved {path}')

    visualisation = (volume - volume.min()) / max(float(np.ptp(volume)), 1e-12)
    visualisation_path = output_dir / f'{patient}_eeg_v2_prior_visualisation.nii.gz'
    visualisation_nii = nib.Nifti1Image(
        visualisation.astype(np.float32), reference.affine, reference.header.copy())
    visualisation_nii.set_data_dtype(np.float32)
    nib.save(visualisation_nii, visualisation_path)
    print(f'Saved {visualisation_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--sensitivity_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mode', choices=['auto', 'joint', 'factorized_depth'],
                        default='auto')
    args = parser.parse_args()
    reconstruct(args.predictions, args.sensitivity_dir, args.output_dir, args.mode)
