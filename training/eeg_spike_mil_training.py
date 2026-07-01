"""
training/eeg_spike_mil_training.py

[EXPERIMENTAL / DEPRECATED]
Training script for EEG spike localization using configurable MIL pooling.

This script trains a simplified SpikeMILClassifier (lobe/hemisphere classification only,
no spatial prior). It was a developmental step toward the full multi-head model.

For production use, see eeg_spike_mil_mh_training.py.
"""

import os
import argparse
import time
from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from models.eeg import SpikeMILClassifier
from datasets.eeg import (
    PatientMILSpikeDataset,
    mil_collate,
    load_split,
    load_labels,
    find_patient_files,
    LOBE_LABEL_TO_INT,
    LOBE_CLASSES,
)


def norm_attention_entropy(weights, mask=None, eps=1e-8):
    # weights: (B, N), already softmaxed
    if mask is not None:
        weights = weights * mask
        weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = weights / weights_sum

        # effective K = number of valid (mask=1) positions per sample
        K = mask.sum(dim=1).float().clamp_min(1.0)
    else:
        B, N = weights.shape
        K = torch.full((B,), float(N), device=weights.device)

    # Compute normalized entropy
    prob = weights.clamp_min(eps)
    H = -(prob * prob.log()).sum(dim=1)  # (B,)
    H_max = K.log()
    norm = H / (H_max + eps)
    # Hard-code entropy = 1 for only one spike (K=1), else value blows up
    norm = torch.where(K <= 1, torch.ones_like(norm), norm)

    return norm             # in [0,1]

def top1_accuracy_from_distributions(pred_probs, target_probs):
    """
    Top-1 accuracy: compares argmax of predicted vs argmax of target.
    """
    pred_classes = pred_probs.argmax(dim=1)
    target_classes = target_probs.argmax(dim=1)
    return (pred_classes == target_classes).float().mean().item()

def l1_distribution_error(pred_probs, target_probs):
    """
    Mean L1 distance between predicted and target distributions.
    Taking mean over all samples, so this is basically MAE. Range [0-1].
    """
    return torch.abs(pred_probs - target_probs).mean().item()

def kl_divergence(pred_probs, target_probs, eps=1e-8):
    """
    KL( target || pred ) computed per sample then averaged.
    Both inputs are probability distributions.
    """
    p = target_probs.clamp(min=eps)
    q = pred_probs.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=1).mean().item()

class SoftLabelCrossEntropy(nn.Module):
    """
    Cross-entropy between a predicted distribution (from logits) and a soft
    target distribution over classes (each row of targets sums to 1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B, C), each row of targets ~ distribution
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, writer=None, 
                   attn_entropy_lambda=0.005, attn_log_interval=10, multi_label=True):
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_l1 = 0.0
    total_probs = 0.0
    # total_kl = 0.0
    total = 0

    global_step = epoch * len(dataloader)

    for i, (X, mask, y) in enumerate(dataloader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, attn = model(X, mask)
            loss = criterion(logits, y)
            if attn is not None:
                loss -= attn_entropy_lambda * norm_attention_entropy(attn.float(), mask).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)

        # Metrics
        if multi_label:
            probs = torch.softmax(logits, dim=1)
            # Top-1 accuracy (dominant lobe)
            batch_top1 = top1_accuracy_from_distributions(probs, y)
            # L1 distribution error
            batch_l1 = l1_distribution_error(probs, y)
            # KL divergence
            # batch_kl = kl_divergence(probs, y)
            # Accumulate
            total_top1 += batch_top1 * X.size(0)
            total_l1 += batch_l1 * X.size(0)
            total_probs += probs.sum(dim=0).detach().cpu()
        else:
            # Single label case
            _, preds = logits.max(dim=1)
            batch_exact = (preds == y).float().mean().item()
            total_top1 += batch_exact * X.size(0)

        total += y.size(0)

        # Logging
        if writer is not None:
            step = global_step + i
            writer.add_scalar("Train/Loss_step", loss.item(), step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Train/LR", current_lr, step)

            # Log attention pooling: entropy, histogram, small image
            if attn is not None and epoch % attn_log_interval == 0 and i == 0:
                # Attention weights entropy
                writer.add_scalar("Attention/TrainWeights_entropy", norm_attention_entropy(attn, mask).mean(), epoch)
                # Attention weights histogram
                writer.add_histogram("Attention/TrainWeights_hist", attn[0].detach().cpu().numpy(), epoch)
                # Image of attention vector (1 x N)
                attn_img = attn[0].detach().cpu()
                # Rescale for better visualization
                attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)
                # Make it a 4:2 rectangle with padding if needed
                total_elements = attn_img.numel()
                height = int(torch.sqrt(torch.tensor(total_elements / 2, dtype=torch.float)).ceil())
                width = height * 2
                target_elements = width * height
                padding_needed = target_elements - total_elements
                if padding_needed > 0:
                    attn_img = torch.nn.functional.pad(attn_img, (0, padding_needed))
                attn_img = attn_img.view(1, height, width)
                writer.add_image("Attention/TrainWeights_image", attn_img, epoch)

    # Epoch metrics
    epoch_loss = total_loss / total
    epoch_top1 = total_top1 / total
    epoch_l1 = total_l1 / total
    epoch_probs = total_probs / total

    return epoch_loss, epoch_top1, epoch_l1, epoch_probs

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer=None, attn_log_interval=10, multi_label=True):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_l1 = 0.0
    total_probs = 0.0
    total = 0

    for i, (X, mask, y) in enumerate(dataloader):
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, attn = model(X, mask)
            loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        
        # Metrics
        if multi_label:
            probs = torch.softmax(logits, dim=1)
            # Top-1 accuracy (dominant lobe)
            batch_top1 = top1_accuracy_from_distributions(probs, y)
            # L1 distribution error
            batch_l1 = l1_distribution_error(probs, y)
            # KL divergence
            # batch_kl = kl_divergence(probs, y)
            # Accumulate
            total_top1 += batch_top1 * X.size(0)
            total_l1 += batch_l1 * X.size(0)
            total_probs += probs.sum(dim=0).detach().cpu()
        else:
            # Single label case
            _, preds = logits.max(dim=1)
            batch_exact = (preds == y).float().mean().item()
            total_top1 += batch_exact * X.size(0)
        
        total += y.size(0)

        # Per-step logging
        if writer is not None and attn is not None and epoch % attn_log_interval == 0 and i == 0:
            # Attention weights entropy
            writer.add_scalar("Attention/ValWeights_entropy", norm_attention_entropy(attn, mask).mean(), epoch)
            # Attention weights histogram
            writer.add_histogram("Attention/ValWeights_hist", attn[0].detach().cpu().numpy(), epoch)
            # Image of attention vector (1 x N)
            attn_img = attn[0].detach().cpu()
            # Rescale for better visualization
            attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)
            # Make it a 4:2 rectangle with padding if needed
            total_elements = attn_img.numel()
            height = int(torch.sqrt(torch.tensor(total_elements / 2, dtype=torch.float)).ceil())
            width = height * 2
            target_elements = width * height
            padding_needed = target_elements - total_elements
            if padding_needed > 0:
                attn_img = torch.nn.functional.pad(attn_img, (0, padding_needed))
            attn_img = attn_img.view(1, height, width)
            writer.add_image("Attention/ValWeights_image", attn_img, epoch)

    avg_loss = total_loss / total
    avg_top1 = total_top1 / total
    avg_l1 = total_l1 / total
    avg_probs = total_probs / total

    return avg_loss, avg_top1, avg_l1, avg_probs

def train(json_split_path, fold_index, data_dir, json_label_path,
    in_channels=21, max_spikes=32, batch_size=8,
    num_classes=2,
    emb_dim=None, hidden=None, dropout=None,
    lr=1e-3, weight_decay=5e-2, epochs=50,
    log_root="./runs", num_workers=0, test_mode=False, multi_label=True,
    amp_enabled=True, pooling=None
):
    """
    Main training entrypoint for EEG spike localization model.

    Args:
        json_split_path (str): Path to JSON with k-fold subject splits.
        fold_index (int): Which fold to train.
        data_dir (str): Directory containing *_spikes.npy files.
        json_label_path (str): Path to JSON mapping patient_id -> class labels.
        in_channels (int): Number of EEG channels.
        max_spikes (int): Max spikes per patient to sample.
        batch_size (int): Patients per batch.
        num_classes (int): Output classes.
        lr (float): Learning rate.
        weight_decay (float): AdamW weight decay.
        epochs (int): Number of epochs.
        log_root (str): Directory to store TensorBoard logs.
        num_workers (int): Number of DataLoader worker processes. 0 means main process.
        test_mode (bool): If True, runs in test mode with limited data.
        multi_label (bool): If True, use multi-label training with BCE loss.
        amp_enabled (bool): If True, use automatic mixed precision (AMP) for training.
    """

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load training/validation id's
    train_ids, val_ids = load_split(json_split_path, fold_index)
    print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")

    # TODO: Watch it, set up as hemi now.
    label_dict = load_labels(json_label_path, label_to_int={"left": 0, "right": 1}, num_classes=num_classes, multi_label=multi_label)

    # Build file paths and labels while checking validity
    train_ids, train_files, train_labels = find_patient_files(data_dir, train_ids, label_dict, test_mode=test_mode)
    val_ids, val_files, val_labels = find_patient_files(data_dir, val_ids, label_dict, test_mode=test_mode)

    # Build datasets and dataloaders
    train_dataset = PatientMILSpikeDataset(
        train_ids, train_files, train_labels,
        max_spikes_per_bag=max_spikes, training=True
    )
    val_dataset = PatientMILSpikeDataset(
        val_ids, val_files, val_labels,
        max_spikes_per_bag=max_spikes, training=False
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=mil_collate, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=mil_collate, shuffle=False, pin_memory=True)

    # Build model
    model_kwargs = {k: v for k, v in dict(emb_dim=emb_dim, hidden=hidden, dropout=dropout, pooling=pooling).items() if v is not None}
    print(f"Model kwargs (overrides): {model_kwargs}")
    model = SpikeMILClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)
    
    # Loss function
    # multi_label=True: soft distribution over lobes → soft-label cross-entropy
    # multi_label=False: single hard class index → standard CrossEntropy
    if multi_label:
        criterion = SoftLabelCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # TensorBoard logging
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_str = "multilabel" if multi_label else "singlelabel"
    log_dir = os.path.join(log_root, f"eeg_mil_{mode_str}_fold{fold_index}_{datestr}")
    writer = SummaryWriter(log_dir=log_dir)
    print("Logging to:", log_dir)

    # ----- Main Training Loop -----
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_l1 = float("inf")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_top1, train_l1, train_probs = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, writer, multi_label=multi_label
        )
        val_loss, val_top1, val_l1, val_probs = validate(
            model, val_loader, criterion, device, epoch, writer, multi_label=multi_label
        )
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_top1:.3f}, L1: {train_l1:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_top1:.3f}, L1: {val_l1:.3f}")
        print(f"Epoch duration: {epoch_duration:.1f}s")

        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Accuracy_top1", {"Train": train_top1, "Val": val_top1}, epoch)
        writer.add_scalars("L1", {"Train": train_l1, "Val": val_l1}, epoch)
        
        # Save best checkpoint
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_val_loss = val_loss
            best_val_l1 = val_l1
            best_checkpoint_path = os.path.join(log_dir, "checkpoint_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_top1,
                    "val_l1": val_l1,
                },
                best_checkpoint_path,
            )
            print(f"✓ New best validation Acc: {val_top1:.3f} (loss: {val_loss:.4f}, L1: {val_l1:.3f})")

        for class_idx in range(num_classes):
            writer.add_scalars(
                f"ClassProb/Train",
                {f"Class_{class_idx}": train_probs[class_idx].item()},
                epoch
            )
            writer.add_scalars(
                f"ClassProb/Val",
                {f"Class_{class_idx}": val_probs[class_idx].item()},
                epoch
            )

    # Close logger
    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best validation - Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.3f}, L1: {best_val_l1:.3f}")
    print(f"{'=' * 60}")


# Main entrypoint for EEG spike localization model
if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Train a MIL classifier on EEG spike data with configurable pooling."
    )

    # Paths
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to JSON file containing k-fold subject splits.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_spikes.npy files.")
    parser.add_argument("--log_root", type=str, default="./runs",
                        help="Base directory for TensorBoard logs.")
    parser.add_argument("--label_json", type=str, required=True,
                        help="JSON mapping patient_id → class labels.")
    # Fold selection
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold from the JSON splits to use.")
    # Model & training hyperparameters
    parser.add_argument("--in_channels", type=int, default=21,
                        help="Number of EEG channels in input.")
    parser.add_argument("--max_spikes", type=int, default=32,
                        help="Max spikes sampled per patient per epoch.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Patients per batch.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker processes. 0 means main process.")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["attention", "mean", "mean-max-topk"],
                        help="MIL pooling mode for the classifier model.")
    parser.add_argument("--test_mode", action="store_true",
                        help="If set, runs in test mode with limited data.")
    parser.add_argument("--single_label", action="store_true",
                        help="If set, uses single-label mode (original behavior).")
    # Parse args
    args = parser.parse_args()

    # Run training
    train(
        json_split_path=args.splits_json,
        fold_index=args.fold,
        data_dir=args.data_dir,
        json_label_path=args.label_json,
        in_channels=args.in_channels,
        max_spikes=args.max_spikes,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_root=args.log_root,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
        multi_label=not args.single_label,
        pooling=args.pooling,
    )
