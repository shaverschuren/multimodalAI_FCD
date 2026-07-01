"""
training/eeg_spike_mil_channel_training.py

[EXPERIMENTAL — FAILED]
Training script for channel-aligned per-electrode spike localization.

This experiment attempted to predict per-channel spike density using a
ChannelAlignedMILClassifier. It did not yield meaningful results and is retained
for reference only.

For production use, see eeg_spike_mil_mh_training.py.
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from models.eeg import ChannelAlignedMILClassifier
from datasets.eeg import (
    PatientMILSpikeDataset,
    mil_collate,
    load_split,
    load_labels,
    find_patient_files,
    CHANNEL_LABEL_TO_INT,
)
import random


# ---------------------------------------------------------------------
# Metrics & regularizers
# ---------------------------------------------------------------------

def mass_normalized_mse(y_hat, y, eps=1e-6):
    y_hat_norm = y_hat / (y_hat.sum(dim=1, keepdim=True) + eps)
    return F.mse_loss(y_hat_norm, y)


def norm_attention_entropy(attn, mask=None, eps=1e-8):
    """
    attn: (K, N)
    mask: (K, N)
    """
    if mask is not None:
        attn = attn * mask
        Z = attn.sum(dim=1, keepdim=True).clamp_min(eps)
        attn = attn / Z
        K_eff = mask.sum(dim=1).float().clamp_min(1.0)
    else:
        K_eff = torch.full((attn.size(0),), attn.size(1), device=attn.device)

    p = attn.clamp_min(eps)
    H = -(p * p.log()).sum(dim=1)
    Hmax = K_eff.log()
    Hnorm = H / (Hmax + eps)
    Hnorm = torch.where(K_eff <= 1, torch.ones_like(Hnorm), Hnorm)
    return Hnorm


def channel_mae(pred, target):
    return torch.abs(pred - target).mean().item()


def mean_spearman(pred, target):
    corrs = []
    for i in range(pred.size(0)):
        p = pred[i].argsort().float()
        t = target[i].argsort().float()
        corr = torch.corrcoef(torch.stack([p, t]))[0, 1]
        if not torch.isnan(corr):
            corrs.append(corr)
    return torch.stack(corrs).mean().item() if corrs else 0.0


def mean_pearson(y_hat, y, eps=1e-6):
    corrs = []
    for i in range(y.size(0)):
        yh = y_hat[i]
        yt = y[i]
        if yh.std() < eps or yt.std() < eps:
            continue
        corr = torch.corrcoef(torch.stack([yh, yt]))[0, 1]
        if not torch.isnan(corr):
            corrs.append(corr)
    return torch.stack(corrs).mean().item() if corrs else 0.0


# ---------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler,
    epoch, writer=None, attn_entropy_lambda=0.005
):
    model.train()
    total_loss, total_mae, total = 0.0, 0.0, 0

    last_batch_cache = None

    for step, (X, mask, y) in enumerate(loader):
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            y_hat, attn = model(X, mask)
            loss = criterion(y_hat, y)

            # Attention entropy regularization (per channel)
            B, C, N = attn.shape
            attn_flat = attn.view(B * C, N)
            mask_flat = mask.repeat_interleave(C, dim=0)
            loss -= attn_entropy_lambda * norm_attention_entropy(
                attn_flat, mask_flat
            ).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
        total_mae += channel_mae(y_hat, y) * X.size(0)
        total += X.size(0)

        last_batch_cache = (y_hat.detach(), y.detach(), attn.detach(), mask.detach())

        if writer:
            global_step = epoch * len(loader) + step
            writer.add_scalar("Train/Loss_step", loss.item(), global_step)

    # ---------------- Epoch-level logging ----------------
    if writer and epoch % 10 == 0 and last_batch_cache is not None:
        y_hat, y, attn, mask = last_batch_cache
        B, C, N = attn.shape

        # Channel mass
        writer.add_scalars(
            "ChannelMass/Predicted",
            {f"Ch_{i}": y_hat[:, i].mean().item() for i in range(C)},
            epoch
        )
        writer.add_scalars(
            "ChannelMass/Target",
            {f"Ch_{i}": y[:, i].mean().item() for i in range(C)},
            epoch
        )

        # Channel MAE
        writer.add_scalars(
            "ChannelError/MAE",
            {f"Ch_{i}": torch.abs(y_hat[:, i] - y[:, i]).mean().item() for i in range(C)},
            epoch
        )

        # Rank correlation
        writer.add_scalar(
            "ChannelRank/Spearman",
            mean_spearman(y_hat, y),
            epoch
        )
        writer.add_scalar(
            "Pearson/Train",
            mean_pearson(y_hat.detach(), y.detach()),
            epoch
        )

        # Heatmaps (patient 0) - reshape to 4x6 rectangle
        pred_heatmap = y_hat[0].clone()
        target_heatmap = y[0].clone()
        
        # Pad to 24 elements (4x6) if needed
        if pred_heatmap.numel() < 24:
            pad_size = 24 - pred_heatmap.numel()
            pred_heatmap = F.pad(pred_heatmap, (0, pad_size))
            target_heatmap = F.pad(target_heatmap, (0, pad_size))
        
        writer.add_image(
            "ChannelHeatmap/TrainPredicted",
            pred_heatmap[:24].view(4, 6).unsqueeze(0),
            epoch,
            dataformats="CHW"
        )
        writer.add_image(
            "ChannelHeatmap/TrainTarget",
            target_heatmap[:24].view(4, 6).unsqueeze(0),
            epoch,
            dataformats="CHW"
        )

        # Attention entropy per channel
        ent = norm_attention_entropy(
            attn.view(B * C, N),
            mask.repeat_interleave(C, dim=0)
        ).view(B, C).mean(dim=0)

        writer.add_scalars(
            "Attention/Entropy_per_channel",
            {f"Ch_{i}": ent[i].item() for i in range(C)},
            epoch
        )

        # Attention histogram + image
        writer.add_histogram(
            "Attention/TrainWeights_hist",
            attn[0].flatten().cpu(),
            epoch
        )

        attn_img = attn[0].mean(dim=0)
        attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)

        total_elements = attn_img.numel()
        height = int(torch.sqrt(torch.tensor(total_elements / 2)).ceil())
        width = height * 2
        pad = height * width - total_elements
        if pad > 0:
            attn_img = F.pad(attn_img, (0, pad))

        writer.add_image(
            "Attention/TrainWeights_image",
            attn_img.view(1, height, width),
            epoch
        )

    return total_loss / total, total_mae / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    total_loss, total_mae, total = 0.0, 0.0, 0

    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            y_hat, _ = model(X, mask)
            loss = criterion(y_hat, y)

        total_loss += loss.item() * X.size(0)
        total_mae += channel_mae(y_hat, y) * X.size(0)
        total += X.size(0)

    writer.add_scalar("Val/Loss", total_loss / total, epoch)
    writer.add_scalar("Val/MAE", total_mae / total, epoch)

    # Log validation heatmaps similar to training - use last batch
    if writer and epoch % 10 == 0:
        # Log Pearson correlation for validation
        writer.add_scalar(
            "Pearson/Val",
            mean_pearson(y_hat, y),
            epoch
        )

        # Heatmaps (random patient) - reshape to 4x6 rectangle
        patient_idx = random.randint(0, y.size(0) - 1)
        pred_heatmap = y_hat[patient_idx].clone()
        target_heatmap = y[patient_idx].clone()
        
        # Pad to 24 elements (4x6) if needed
        if pred_heatmap.numel() < 24:
            pad_size = 24 - pred_heatmap.numel()
            pred_heatmap = F.pad(pred_heatmap, (0, pad_size))
            target_heatmap = F.pad(target_heatmap, (0, pad_size))
        
        writer.add_image(
            "ChannelHeatmap/ValPredicted",
            pred_heatmap[:24].view(4, 6).unsqueeze(0),
            epoch,
            dataformats="CHW"
        )
        writer.add_image(
            "ChannelHeatmap/ValTarget",
            target_heatmap[:24].view(4, 6).unsqueeze(0),
            epoch,
            dataformats="CHW"
        )

    return total_loss / total, total_mae / total


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def train(
    splits_json, fold, data_dir, label_json,
    in_channels=21, max_spikes=32, batch_size=4,
    emb_dim=None, hidden=None, dropout=None,
    lr=1e-4, weight_decay=1e-4, epochs=50,
    log_root="./runs", num_workers=0, test_mode=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ids, val_ids = load_split(splits_json, fold)
    label_dict = load_labels(
        label_json,
        label_to_int=CHANNEL_LABEL_TO_INT,
        num_classes=in_channels,
        multi_label=True,
    )

    train_ids, train_files, train_labels = find_patient_files(
        data_dir, train_ids, label_dict, test_mode, skip_zero_labels=True
    )
    val_ids, val_files, val_labels = find_patient_files(
        data_dir, val_ids, label_dict, test_mode, skip_zero_labels=True
    )

    train_ds = PatientMILSpikeDataset(
        train_ids, train_files, train_labels,
        max_spikes_per_bag=max_spikes, training=True
    )
    val_ds = PatientMILSpikeDataset(
        val_ids, val_files, val_labels,
        max_spikes_per_bag=max_spikes, training=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=mil_collate, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=mil_collate, num_workers=num_workers
    )

    model_kwargs = {k: v for k, v in dict(emb_dim=emb_dim, hidden=hidden, dropout=dropout).items() if v is not None}
    print(f"Model kwargs (overrides): {model_kwargs}")
    model = ChannelAlignedMILClassifier(in_channels=in_channels, **model_kwargs).to(device)

    criterion = mass_normalized_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    run_name = f"channel_mil_fold{fold}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(os.path.join(log_root, run_name))
    print("Logging to:", writer.log_dir)

    best_val_mae = float("inf")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, epoch, writer
        )
        val_loss, val_mae = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        print(f"Epoch duration: {epoch_duration:.1f}s")
        
        writer.add_scalars(
            "Epoch/Loss",
            {"Train": train_loss, "Val": val_loss},
            epoch
        )
        writer.add_scalars(
            "Epoch/MAE",
            {"Train": train_mae, "Val": val_mae},
            epoch
        )
        
        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(writer.log_dir, "checkpoint_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                best_checkpoint_path,
            )
            print(f"✓ New best validation MAE: {val_mae:.4f} (loss: {val_loss:.4f})")

    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best validation - Loss: {best_val_loss:.4f}, MAE: {best_val_mae:.4f}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_json", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_spikes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_root", default="./runs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    train(
        splits_json=args.splits_json,
        fold=args.fold,
        data_dir=args.data_dir,
        label_json=args.label_json,
        max_spikes=args.max_spikes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_root=args.log_root,
        num_workers=args.num_workers,
        test_mode=args.test_mode
    )
