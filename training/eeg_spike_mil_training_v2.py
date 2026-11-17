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


def load_factorized_labels(channel_path, depth_path):
    with open(channel_path, encoding='utf-8') as f:
        channel_raw = json.load(f)
    with open(depth_path, encoding='utf-8') as f:
        depth_raw = json.load(f)
    channel_names = {name.casefold(): index for index, name in enumerate(CHANNELS)}
    depth_names = {name.casefold(): index for index, name in enumerate(DEPTH_BIN_NAMES)}
    common = sorted(set(channel_raw) & set(depth_raw))
    if not common:
        raise ValueError('Channel and global-depth targets have no shared patients.')
    labels = {}
    skipped = []
    for patient in common:
        channel = torch.zeros(len(CHANNELS), dtype=torch.float32)
        for name, value in channel_raw[patient].items():
            key = name.casefold()
            if key not in channel_names:
                raise ValueError(f'Unknown channel {name!r} for {patient}.')
            channel[channel_names[key]] += float(value)
        depth = torch.zeros(len(DEPTH_BIN_NAMES), dtype=torch.float32)
        for name, value in depth_raw[patient].items():
            key = name.casefold()
            if key not in depth_names:
                raise ValueError(f'Unknown depth bin {name!r} for {patient}.')
            depth[depth_names[key]] += float(value)
        for name, target in (('channel', channel), ('depth', depth)):
            if not torch.isfinite(target).all() or (target < 0).any():
                raise ValueError(f'Invalid {name} target values for {patient}.')
        if channel.sum() <= 0:
            skipped.append(patient)
            continue
        if not torch.isclose(channel.sum(), channel.new_tensor(1.), atol=1e-5):
            raise ValueError(f'Channel target for {patient} does not sum to one.')
        if not torch.isclose(depth.sum(), depth.new_tensor(1.), atol=1e-5):
            raise ValueError(f'Global depth target for {patient} does not sum to one.')
        labels[patient] = torch.cat((channel, depth))
    if skipped:
        print(f'Skipped {len(skipped)} patients with empty channel targets.')
    if not labels:
        raise ValueError('No valid factorized channel/depth targets.')
    unmatched = (len(set(channel_raw) - set(depth_raw)),
                 len(set(depth_raw) - set(channel_raw)))
    if any(unmatched):
        print(f'Unmatched target IDs: {unmatched[0]} channel-only, '
              f'{unmatched[1]} depth-only; using shared patient IDs.')
    return labels


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


def run_epoch(model, loader, device, optimizer=None, mode='joint',
              depth_loss_weight=1.0):
    """Run one epoch and return mode-specific losses and metrics."""
    training = optimizer is not None
    model.train(training)
    metric_names = (('joint_kl', 'joint_overlap', 'kl', 'mae', 'overlap')
                    if mode == 'joint' else
                    ('channel_kl', 'channel_overlap', 'channel_mae', 'depth_kl',
                     'depth_overlap', 'total_loss'))
    totals = dict.fromkeys(metric_names, 0.)
    count = 0

    for spikes, mask, target in loader:
        spikes, mask = spikes.to(device), mask.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            if mode == 'joint':
                pred = model(spikes, mask)
                joint_kl = F.kl_div(pred.clamp_min(1e-8).log(), target,
                                    reduction='batchmean')
                loss = joint_kl
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                p_ch, t_ch = pred.sum(dim=-1), target.sum(dim=-1)
                values = {
                    'joint_kl': joint_kl.item(),
                    'joint_overlap': torch.minimum(pred, target).sum((1, 2)).mean().item(),
                    'kl': F.kl_div(p_ch.clamp_min(1e-8).log(), t_ch,
                                   reduction='batchmean').item(),
                    'mae': (p_ch - t_ch).abs().mean().item(),
                    'overlap': torch.minimum(p_ch, t_ch).sum(dim=1).mean().item(),
                }
            else:
                pred_channel, pred_depth = model(spikes, mask)
                target_channel = target[:, :len(CHANNELS)]
                target_depth = target[:, len(CHANNELS):]
                channel_kl = F.kl_div(pred_channel.clamp_min(1e-8).log(),
                                      target_channel, reduction='batchmean')
                depth_kl = F.kl_div(pred_depth.clamp_min(1e-8).log(),
                                    target_depth, reduction='batchmean')
                loss = channel_kl + depth_loss_weight * depth_kl
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                values = {
                    'channel_kl': channel_kl.item(),
                    'channel_overlap': torch.minimum(pred_channel, target_channel).sum(dim=1).mean().item(),
                    'channel_mae': (pred_channel - target_channel).abs().mean().item(),
                    'depth_kl': depth_kl.item(),
                    'depth_overlap': torch.minimum(pred_depth, target_depth).sum(dim=1).mean().item(),
                    'total_loss': loss.item(),
                }
        n = target.size(0)
        for metric, value in values.items():
            totals[metric] += value * n
        count += n
    return {metric: value / count for metric, value in totals.items()}


@torch.no_grad()
def save_predictions(model, loader, patient_ids, device, path, bin_names,
                     mode='joint', target_labels=None):
    """Save CSV and JSON-ready predictions in canonical channel order."""
    model.eval()
    if mode == 'joint':
        keys = [(channel, name) for channel in CHANNELS for name in bin_names]
        pred_fields = [f'pred_{ch}_{name}' for ch, name in keys]
        target_fields = [f'target_{ch}_{name}' for ch, name in keys]
    else:
        pred_fields = [f'pred_{channel}' for channel in CHANNELS]
        pred_fields += [f'pred_depth_{name}' for name in bin_names]
        target_fields = [f'target_{channel}' for channel in CHANNELS]
        target_fields += [f'target_depth_{name}' for name in bin_names]
    fields = ['patient_id'] + pred_fields + target_fields
    predictions = {}

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        index = 0
        for spikes, mask, targets in loader:
            output = model(spikes.to(device), mask.to(device))
            outputs = output.cpu() if mode == 'joint' else tuple(value.cpu() for value in output)
            for batch_index in range(targets.size(0)):
                pid = patient_ids[index]
                row = {'patient_id': pid}
                target = target_labels.get(pid) if target_labels is not None else targets[batch_index]
                if mode == 'joint':
                    prediction = outputs[batch_index]
                    predictions[pid] = {
                        channel: dict(zip(bin_names, prediction[i].tolist()))
                        for i, channel in enumerate(CHANNELS)
                    }
                    row.update(zip(pred_fields, prediction.flatten().tolist()))
                    valid_target = (
                        target is not None and torch.isfinite(target).all()
                        and (target >= 0).all() and target.sum() > 0
                    )
                    if valid_target:
                        row.update(zip(target_fields, target.flatten().tolist()))
                else:
                    channel_probs, depth_probs = (value[batch_index] for value in outputs)
                    predictions[pid] = {
                        'channels': dict(zip(CHANNELS, channel_probs.tolist())),
                        'depth': dict(zip(bin_names, depth_probs.tolist())),
                    }
                    row.update(zip(pred_fields, channel_probs.tolist() + depth_probs.tolist()))
                    if target is not None:
                        target = target.detach().cpu().view(-1)
                        channel_target = target[:len(CHANNELS)]
                        depth_target = target[len(CHANNELS):]
                        valid_target = (
                            target.numel() == len(CHANNELS) + len(bin_names)
                            and torch.isfinite(target).all() and (target >= 0).all()
                            and channel_target.sum() > 0 and depth_target.sum() > 0
                        )
                        if valid_target:
                            row.update(zip(target_fields, target.tolist()))
                writer.writerow(row)
                index += 1
    return predictions


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ids, val_ids = load_split(args.splits_json, args.fold)
    if args.mode == 'joint':
        labels, target_type, bin_names = load_joint_labels(args.label_json)
        label_paths = {'joint': args.label_json}
    else:
        if not args.depth_label_json:
            raise ValueError('--depth_label_json is required for factorized_depth mode.')
        labels = load_factorized_labels(args.label_json, args.depth_label_json)
        target_type, bin_names = 'factorized_depth', DEPTH_BIN_NAMES
        label_paths = {'channels': args.label_json, 'depth': args.depth_label_json}

    def make_dataset(ids, training, max_spikes, unlabelled=False):
        if unlabelled:
            placeholder = (torch.zeros(21, len(bin_names)) if args.mode == 'joint'
                           else torch.zeros(21 + len(DEPTH_BIN_NAMES)))
            targets = {pid: placeholder.clone() for pid in ids}
        else:
            targets = labels
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
                             num_bins=len(bin_names), mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    run_mode = f'joint_{target_type}' if args.mode == 'joint' else 'factorized_depth'
    run_name = (f'eeg_v2_fold{args.fold}_{args.spatial_mixing}_{run_mode}_'
                f'{datetime.now():%Y%m%d-%H%M%S}')
    out_dir = Path(args.log_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)
    best_loss, best_epoch = float('inf'), 0
    print(f'{device} | Mode: {run_mode} | Train: {len(train_ds)} | '
          f'Val: {len(val_ds)} | {out_dir}')

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer,
                       mode=args.mode, depth_loss_weight=args.depth_loss_weight)
        va = run_epoch(model, val_loader, device, mode=args.mode,
                       depth_loss_weight=args.depth_loss_weight)
        for metric in tr:
            writer.add_scalars(metric, {'train': tr[metric], 'val': va[metric]}, epoch)
        if args.mode == 'joint':
            print(f'{epoch:03d}/{args.epochs}  '
                  f'train joint KL={tr["joint_kl"]:.4f}  '
                  f'val joint KL={va["joint_kl"]:.4f}  '
                  f'val channel KL={va["kl"]:.4f}  '
                  f'val channel overlap={va["overlap"]:.3f}  '
                  f'val joint overlap={va["joint_overlap"]:.3f}')
            validation_loss = va['joint_kl']
        else:
            print(f'{epoch:03d}/{args.epochs}  '
                  f'train total={tr["total_loss"]:.4f}  '
                  f'val total={va["total_loss"]:.4f}  '
                  f'val channel KL={va["channel_kl"]:.4f}  '
                  f'channel overlap={va["channel_overlap"]:.3f}  '
                  f'val depth KL={va["depth_kl"]:.4f}  '
                  f'depth overlap={va["depth_overlap"]:.3f}')
            validation_loss = va['total_loss']
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            checkpoint = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'mode': args.mode, 'validation_loss': best_loss,
                'target_type': target_type, 'target_paths': label_paths,
                'bin_names': list(bin_names), 'channel_order': CHANNELS,
                'depth_loss_weight': args.depth_loss_weight,
                'args': vars(args),
            }
            if args.mode == 'joint':
                checkpoint['val_joint_kl'] = best_loss
            else:
                checkpoint['val_total_loss'] = best_loss
            torch.save(checkpoint, out_dir / 'checkpoint_best.pt')

    writer.close()
    checkpoint = torch.load(out_dir / 'checkpoint_best.pt', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    save_predictions(model, val_loader, val_ds.patient_ids, device,
                     out_dir / 'val_predictions.csv', bin_names, mode=args.mode,
                     target_labels=labels)
    print(f'Best epoch: {best_epoch}, validation loss: {best_loss:.4f}')
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
            test_dir / 'test_predictions.csv', bin_names, mode=args.mode,
            target_labels=labels)
        with open(test_dir / 'test_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f'Saved {len(predictions)} test predictions to {test_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits_json', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--label_json', required=True,
                        help='Joint labels, or gt_channels.json for factorized_depth')
    parser.add_argument('--mode', choices=['joint', 'factorized_depth'],
                        default='joint')
    parser.add_argument('--depth_label_json',
                        help='Independent global-depth targets for factorized_depth mode')
    parser.add_argument('--depth_loss_weight', type=float, default=0.1)
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
