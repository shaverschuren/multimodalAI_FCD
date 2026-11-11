"""
training/eeg_spike_mil_training_v2.py

Core EEG spike localization training script for SpikeMILModel_v2.
This is NOT the original EEG MIL model described in our manuscript; it is the updated version 2.
For the original EEG MIL model training, see training/eeg_spike_mil_mh_training.py.

Trains SpikeMILModel_v2 with:
  - Spike encoder:  SpikeEncoder_v2 (temporal 1D CNN + 2D spatial mixing)
  - MIL pooling:    Attention or mean pooling over spikes per patient
  - Head:           Per-channel prediction of underlying irritative zone.

We're re-using most of the EEG dataset sampling + augmentation machinery from the old multi-head EEG trainer.
After training, inference is run automatically on the validation set.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets.eeg import (PatientMILSpikeDataset, mil_collate, load_split,
                          load_labels, find_patient_files, CHANNEL_LABEL_TO_INT,
                          LOBE_LABEL_TO_INT, LOBE_CLASSES,
                          DEPTH_BIN_NAMES, DISTANCE_BIN_NAMES)
from models.eeg_v2 import SpikeMILModel_v2

CHANNELS = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
BIN_SETS = {'depth': DEPTH_BIN_NAMES, 'distance': DISTANCE_BIN_NAMES}


def load_joint_labels(path):
    """Read {patient: {channel: {bin: weight}}} into (21, 3) tensors."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    names = next((set(weights) for patient in raw.values()
                  for weights in patient.values()), None)
    if names is None:
        raise ValueError(f'No joint targets found in {path}.')
    kind = next((key for key, bins in BIN_SETS.items()
                 if names == set(bins)), None)
    if kind is None:
        raise ValueError(f'Unexpected depth/distance bins in {path}: {names}')
    bins = BIN_SETS[kind]

    labels = {}
    for pid, channels in raw.items():
        target = torch.zeros(len(CHANNELS), len(bins), dtype=torch.float32)
        for channel, weights in channels.items():
            if channel not in CHANNEL_LABEL_TO_INT or set(weights) != set(bins):
                raise ValueError(f'Invalid channel/bin names in {pid}: {channel}')
            target[CHANNEL_LABEL_TO_INT[channel]] = torch.tensor(
                [weights[name] for name in bins], dtype=torch.float32)
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError(f'Invalid target values for {pid}.')
        if target.sum() > 0:
            if not torch.isclose(target.sum(), target.new_tensor(1.), atol=1e-5):
                raise ValueError(f'Joint target for {pid} does not sum to one.')
            labels[pid] = target
    if not labels:
        raise ValueError(f'No valid nonzero targets in {path}.')
    return labels, kind, bins


def lobe_sampler(dataset, lobe_json):
    """Optional lobe-harmonised patient sampling."""
    if not lobe_json:
        return None
    labels = load_labels(lobe_json, LOBE_LABEL_TO_INT, len(LOBE_CLASSES))
    present = [labels[p] for p in dataset.patient_ids
               if p in labels and labels[p].sum() > 0]
    if not present:
        return None
    mass = torch.stack(present).sum(dim=0)
    rarity = (mass / mass.sum()).clamp_min(1e-8).reciprocal()
    rarity /= rarity.mean()
    weights = []
    for pid in dataset.patient_ids:
        y = labels.get(pid)
        score = float(y @ rarity) if y is not None and y.sum() > 0 else 1.0
        weights.append(max(1.0 + 0.5 * (score - 1.0), 1e-3))
    return WeightedRandomSampler(weights, len(dataset), replacement=True)


def run_epoch(model, loader, device, optimizer=None):
    """Optimise joint KL; report joint and channel-marginal performance."""
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(('joint_kl', 'joint_overlap', 'kl', 'mae', 'overlap'), 0.)
    count = 0

    for spikes, mask, target in loader:
        spikes, mask = spikes.to(device), mask.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            pred = model(spikes, mask)
            joint_kl = F.kl_div(pred.clamp_min(1e-8).log(), target,
                                reduction='batchmean')
            if training:
                optimizer.zero_grad()
                joint_kl.backward()
                optimizer.step()

        with torch.no_grad():
            p_ch, t_ch = pred.sum(dim=-1), target.sum(dim=-1)
            values = {
                'joint_kl': joint_kl.item(),
                'joint_overlap': torch.minimum(pred, target).sum((1, 2)).mean().item(),
                'kl': F.kl_div(p_ch.clamp_min(1e-8).log(), t_ch,
                               reduction='batchmean').item(),
                'mae': (p_ch - t_ch).abs().mean().item(),
                'overlap': torch.minimum(p_ch, t_ch).sum(dim=1).mean().item(),
            }
        n = target.size(0)
        for metric, value in values.items():
            totals[metric] += value * n
        count += n
    return {metric: value / count for metric, value in totals.items()}


@torch.no_grad()
def save_predictions(model, loader, patient_ids, device, path, bin_names,
                     target_labels=None):
    """Save CSV and JSON-ready nested predictions in canonical channel order."""
    model.eval()
    keys = [(channel, name) for channel in CHANNELS for name in bin_names]
    pred_fields = [f'pred_{ch}_{name}' for ch, name in keys]
    target_fields = [f'target_{ch}_{name}' for ch, name in keys]
    fields = ['patient_id'] + pred_fields + target_fields
    predictions = {}

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        index = 0
        for spikes, mask, targets in loader:
            probs = model(spikes.to(device), mask.to(device)).cpu()
            for p, y in zip(probs, targets):
                pid = patient_ids[index]
                predictions[pid] = {
                    channel: dict(zip(bin_names, p[i].tolist()))
                    for i, channel in enumerate(CHANNELS)
                }
                row = {'patient_id': pid}
                row.update(zip(pred_fields, p.flatten().tolist()))
                target = target_labels.get(pid) if target_labels is not None else y
                if target is not None and target.sum() > 0:
                    row.update(zip(target_fields, target.flatten().tolist()))
                writer.writerow(row)
                index += 1
    return predictions


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ids, val_ids = load_split(args.splits_json, args.fold)
    labels, target_type, bin_names = load_joint_labels(args.label_json)

    def make_dataset(ids, training, max_spikes, unlabelled=False):
        targets = ({pid: torch.zeros(21, len(bin_names)) for pid in ids}
                   if unlabelled else labels)
        ids, files, targets = find_patient_files(
            args.data_dir, ids, targets, test_mode=args.test_mode,
            skip_zero_labels=not unlabelled)
        return PatientMILSpikeDataset(
            ids, files, targets, max_spikes_per_bag=max_spikes,
            min_spikes_per_patient=32, training=training)

    train_ds = make_dataset(train_ids, True, args.max_spikes)
    val_ds = make_dataset(val_ids, False, args.max_spikes_val)
    if not len(train_ds) or not len(val_ds):
        raise ValueError('No valid training or validation subjects after filtering.')

    sampler = lobe_sampler(train_ds, args.lobe_json)
    loader_args = dict(batch_size=args.batch_size, collate_fn=mil_collate,
                       num_workers=args.num_workers,
                       pin_memory=device.type == 'cuda',
                       persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None,
                              sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = SpikeMILModel_v2(spatial_mixing=args.spatial_mixing,
                             num_bins=len(bin_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    run_name = (f'eeg_v2_fold{args.fold}_{args.spatial_mixing}_{target_type}_'
                f'{datetime.now():%Y%m%d-%H%M%S}')
    out_dir = Path(args.log_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)
    best_kl, best_epoch = float('inf'), 0
    print(f'{device} | Target: {target_type} | Train: {len(train_ds)} | '
          f'Val: {len(val_ds)} | {out_dir}')

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        for metric in tr:
            writer.add_scalars(metric, {'train': tr[metric], 'val': va[metric]}, epoch)
        print(f'{epoch:03d}/{args.epochs}  '
              f'train joint KL={tr["joint_kl"]:.4f}  '
              f'val joint KL={va["joint_kl"]:.4f}  '
              f'val channel KL={va["kl"]:.4f}  '
              f'val channel overlap={va["overlap"]:.3f}  '
              f'val joint overlap={va["joint_overlap"]:.3f}')
        if va['joint_kl'] < best_kl:
            best_kl, best_epoch = va['joint_kl'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_joint_kl': best_kl, 'target_type': target_type,
                        'bin_names': list(bin_names), 'channel_order': CHANNELS,
                        'args': vars(args)}, out_dir / 'checkpoint_best.pt')

    writer.close()
    checkpoint = torch.load(out_dir / 'checkpoint_best.pt', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    save_predictions(model, val_loader, val_ds.patient_ids, device,
                     out_dir / 'val_predictions.csv', bin_names)
    print(f'Best epoch: {best_epoch}, val joint KL: {best_kl:.4f}')
    print(f'Saved checkpoint and validation predictions to {out_dir}')

    if args.run_test_inference:
        with open(args.splits_json) as f:
            test_ids = json.load(f)['folds'][f'fold_{args.fold}']['test_ids']
        if not test_ids:
            raise ValueError(f'No test IDs defined for fold {args.fold}.')
        test_ds = make_dataset(test_ids, False, args.max_spikes_val,
                               unlabelled=True)
        if not len(test_ds):
            raise ValueError('No valid test subjects after filtering.')

        test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
        test_dir = Path(args.output_dir) / run_name
        test_dir.mkdir(parents=True, exist_ok=True)
        predictions = save_predictions(
            model, test_loader, test_ds.patient_ids, device,
            test_dir / 'test_predictions.csv', bin_names, target_labels=labels)
        with open(test_dir / 'test_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f'Saved {len(predictions)} test predictions to {test_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits_json', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--label_json', required=True,
                        help='gt_channel_depth.json or gt_channel_distance.json')
    parser.add_argument('--lobe_json', help='Optional gt_lobe.json for balanced sampling')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_spikes', type=int, default=32)
    parser.add_argument('--max_spikes_val', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--spatial_mixing', choices=['pairwise', 'mlp'],
                        default='pairwise')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_root', default='./runs')
    parser.add_argument('--run-test-inference', action='store_true')
    parser.add_argument('--output_dir', default='./output',
                        help='Directory for test predictions, separate from TensorBoard')
    parser.add_argument('--test_mode', action='store_true')
    train(parser.parse_args())
