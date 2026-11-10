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
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets.eeg import (PatientMILSpikeDataset, mil_collate, load_split,
                          load_labels, find_patient_files, CHANNEL_LABEL_TO_INT,
                          LOBE_LABEL_TO_INT, LOBE_CLASSES)
from models.eeg_v2 import SpikeMILModel_v2

COORD_PATH = (Path(__file__).resolve().parent.parent / 'preprocessing' /
              'eeg' / 'standard_1005_mni_coordinates.json')
ANNEAL_FRACTION = 0.6


def channel_affinity(sigma, device):
    """Row-normalised Gaussian affinity between electrodes (coordinates in mm)."""
    with open(COORD_PATH) as f:
        data = json.load(f)['data']
    channels = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
    coords = torch.tensor([[data[ch]['mni'][axis] for axis in 'xyz']
                           for ch in channels], dtype=torch.float32, device=device)
    dist2 = torch.cdist(coords, coords).square()
    return torch.softmax(-dist2 / (2 * sigma**2), dim=1)


def spatial_kl(pred, target, affinity, alpha):
    """KL against targets with locally redistributed probability mass."""
    soft_target = (1 - alpha) * target + alpha * (target @ affinity)
    return F.kl_div(pred.clamp_min(1e-8).log(), soft_target,
                    reduction='batchmean')


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


def run_epoch(model, loader, device, optimizer=None, affinity=None, alpha=0.0):
    training = optimizer is not None
    model.train(training)
    total_kl = total_mae = total_overlap = total_loss = count = 0
    for spikes, mask, target in loader:
        spikes, mask = spikes.to(device), mask.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            pred = model(spikes, mask)
            kl = F.kl_div(pred.clamp_min(1e-8).log(), target,
                          reduction='batchmean')
            loss = spatial_kl(pred, target, affinity, alpha) if training else kl
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        n = target.size(0)
        total_loss += loss.item() * n
        total_kl += kl.item() * n
        total_mae += (pred - target).abs().mean().item() * n
        total_overlap += torch.minimum(pred, target).sum(dim=1).mean().item() * n
        count += n
    return (total_kl / count, total_mae / count,
            total_overlap / count, total_loss / count)


@torch.no_grad()
def save_predictions(model, loader, patient_ids, device, path):
    """Save validation predictions in canonical 21-channel order."""
    model.eval()
    channels = sorted(CHANNEL_LABEL_TO_INT, key=CHANNEL_LABEL_TO_INT.get)
    fields = ['patient_id'] + [f'pred_{c}' for c in channels]
    fields += [f'target_{c}' for c in channels]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        index = 0
        for spikes, mask, targets in loader:
            probs = model(spikes.to(device), mask.to(device)).cpu()
            for p, y in zip(probs, targets):
                row = dict(zip(fields, [patient_ids[index]] + p.tolist() + y.tolist()))
                writer.writerow(row)
                index += 1


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ids, val_ids = load_split(args.splits_json, args.fold)
    labels = load_labels(args.label_json, CHANNEL_LABEL_TO_INT, 21)

    def make_dataset(ids, training, max_spikes):
        ids, files, targets = find_patient_files(
            args.data_dir, ids, labels, test_mode=args.test_mode,
            skip_zero_labels=True)
        return PatientMILSpikeDataset(
            ids, files, targets, max_spikes_per_bag=max_spikes,
            min_spikes_per_patient=32, training=training)

    train_ds = make_dataset(train_ids, True, args.max_spikes)
    val_ds = make_dataset(val_ids, False, args.max_spikes_val)
    if not len(train_ds) or not len(val_ds):
        raise ValueError('No valid training or validation subjects after filtering.')

    sampler = lobe_sampler(train_ds, args.lobe_json)
    loader_args = dict(batch_size=args.batch_size, collate_fn=mil_collate,
                       num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                       persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None,
                              sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = SpikeMILModel_v2(spatial_mixing=args.spatial_mixing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    affinity = channel_affinity(args.smooth_sigma, device)
    out_dir = Path(args.log_root) / (
        f'eeg_v2_fold{args.fold}_{args.spatial_mixing}_{datetime.now():%Y%m%d-%H%M%S}')
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir)
    best_kl, best_epoch = float('inf'), 0
    print(f'{device} | Train: {len(train_ds)} | Val: {len(val_ds)} | {out_dir}')

    for epoch in range(1, args.epochs + 1):
        progress = min((epoch - 1) / (ANNEAL_FRACTION * args.epochs), 1.0)
        alpha = args.smooth_alpha_end + 0.5 * (
            args.smooth_alpha_start - args.smooth_alpha_end) * (
            1 + math.cos(math.pi * progress))
        tr = run_epoch(model, train_loader, device, optimizer, affinity, alpha)
        va = run_epoch(model, val_loader, device)
        for metric, train_val, val_val in zip(('kl', 'mae', 'overlap'), tr, va):
            writer.add_scalars(metric, {'train': train_val, 'val': val_val}, epoch)
        writer.add_scalar('loss/train_spatial_kl', tr[3], epoch)
        writer.add_scalar('loss/alpha', alpha, epoch)
        print(f'{epoch:03d}/{args.epochs}  alpha={alpha:.3f}  '
              f'train KL={tr[0]:.4f}  train soft KL={tr[3]:.4f}  '
              f'val KL={va[0]:.4f}  val MAE={va[1]:.4f}  '
              f'val overlap={va[2]:.3f}')
        if va[0] < best_kl:
            best_kl, best_epoch = va[0], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_kl': best_kl, 'channel_order': list(CHANNEL_LABEL_TO_INT),
                        'args': vars(args)}, out_dir / 'checkpoint_best.pt')

    writer.close()
    checkpoint = torch.load(out_dir / 'checkpoint_best.pt', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    save_predictions(model, val_loader, val_ds.patient_ids, device,
                     out_dir / 'val_predictions.csv')
    print(f'Best epoch: {best_epoch}, val KL: {best_kl:.4f}')
    print(f'Saved checkpoint and validation predictions to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits_json', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--label_json', required=True, help='gt_channels.json')
    parser.add_argument('--lobe_json', help='Optional gt_lobe.json for balanced sampling')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_spikes', type=int, default=32)
    parser.add_argument('--max_spikes_val', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--spatial_mixing', choices=['pairwise', 'mlp'], default='pairwise')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smooth_alpha_start', type=float, default=0.5)
    parser.add_argument('--smooth_alpha_end', type=float, default=0.)
    parser.add_argument('--smooth_sigma', type=float, default=30.0, help='Electrode distance scale in mm')
    parser.add_argument('--log_root', default='./runs')
    parser.add_argument('--test_mode', action='store_true')
    train(parser.parse_args())
