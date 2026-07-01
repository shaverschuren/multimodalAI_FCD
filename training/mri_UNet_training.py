"""
training.mri_UNet_training.py
Training script for a standard 3D U-Net model for MRI-only segmentation.

The model uses:
- MRI backbone: 3D Residual Encoder U-Net (nnUNet-style ResEncUNet_3D)
- Input: T1 + FLAIR (2 channels)
- Output: segmentation logits (num_classes channels)

The script assumes:
1. A k-fold dataset split JSON is available
2. Preprocessed MRI data (*.npz files) are available

Author: Sjors Verschuren
Date: January 2026
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from datasets.mri import MRIDataset
from models.mri import ResEncUNet_3D


def load_fold_split(json_path, fold_index):
    """Load subject IDs for a given fold from k_fold_splits.json."""
    with open(json_path, "r") as f:
        payload = json.load(f)
    
    fold_payload = payload["folds"]
    fold_key = f"fold_{fold_index}"
    
    if fold_key not in fold_payload:
        raise ValueError(f"{fold_key} not found in fold split JSON.")
    
    train_ids = fold_payload[fold_key]["train_ids"]
    val_ids = fold_payload[fold_key]["val_ids"]
    
    return train_ids, val_ids

def build_cases(subject_ids, mri_data_dir):
    """
    Build case list from subject IDs and MRI data directory.
    Each case should have a .npz file: <id>_preproc.npz
    """
    cases = []
    for sid in subject_ids:
        npy_path = os.path.join(mri_data_dir, sid, f"{sid}_preproc.npz")
        if os.path.exists(npy_path):
            cases.append({"id": sid, "npy": npy_path})
        else:
            print(f"Warning: Missing preprocessed file for {sid} at {npy_path}")
    
    if not cases:
        raise ValueError(f"No valid cases found in {mri_data_dir}")
    
    print(f"Built {len(cases)} valid cases from {len(subject_ids)} subjects.")
    return cases

def dice_loss(logits, targets, smooth=1.0):
    """
    Dice loss for segmentation.
    logits: [B, C, D, H, W]
    targets: [B, D, H, W] (binary/class indices)
    """
    probs = torch.softmax(logits, dim=1)  # [B, C, D, H, W]
    
    # For binary segmentation (C=2), take foreground probability
    if logits.shape[1] == 2:
        probs = probs[:, 1]  # [B, D, H, W]
    else:
        # For multi-class, use max probability
        probs = probs.max(dim=1)[0]  # [B, D, H, W]
    
    targets_float = targets.float()
    
    intersection = (probs * targets_float).sum()
    union = probs.sum() + targets_float.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, writer=None):
    """Train for one epoch. Returns (epoch_loss, epoch_dice)."""
    model.train()
    
    total_loss = 0.0
    total = 0
    total_dice = 0.0
    
    global_step = epoch * len(dataloader)
    
    for step, batch in enumerate(dataloader):
        x = batch["x"].to(device, non_blocking=True)  # [B, 2, D, H, W]
        y = batch["y"].to(device, non_blocking=True)  # [B, D, H, W]
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)  # [B, num_classes, D, H, W]
            ce_loss = criterion(logits, y.long())
            dice_loss_val = dice_loss(logits, y)
            dice_score = 1.0 - float(dice_loss_val.detach())
            loss = ce_loss + dice_loss_val
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score * batch_size
        total += batch_size
        
        # Per-step logging
        if writer is not None and step % 10 == 0:
            step_idx = global_step + step
            writer.add_scalar("Train/Loss_step", loss.item(), step_idx)
            writer.add_scalar("Train/Dice_step", dice_score, step_idx)
    
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_dice = total_dice / total if total > 0 else 0.0
    return epoch_loss, epoch_dice

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate for one epoch. Returns (epoch_loss, epoch_dice)."""
    model.eval()
    
    total_loss = 0.0
    total = 0
    total_dice = 0.0
    
    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)  # [B, 2, D, H, W]
        y = batch["y"].to(device, non_blocking=True)  # [B, D, H, W]
        
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)  # [B, num_classes, D, H, W]
            ce_loss = criterion(logits, y.long())
            dice_loss_val = dice_loss(logits, y)
            dice_score = 1.0 - float(dice_loss_val)
            loss = ce_loss + dice_loss_val
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score * batch_size
        total += batch_size
    
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_dice = total_dice / total if total > 0 else 0.0
    # Log validation metrics
    if writer is not None:
        writer.add_scalar("Val/Dice_epoch", epoch_dice, epoch)
    return epoch_loss, epoch_dice


def train(
    json_fold_path,
    fold_index,
    mri_data_dir,
    mri_checkpoint_path=None,
    num_classes=2,
    batch_size=2,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=50,
    log_root="./runs_mri",
    num_workers=0,
    test_mode=False,
    amp_enabled=True,
    patch_size=128,
    enable_patch_sampling=False,
):
    """
    Main training entrypoint for MRI-only segmentation.
    
    Args:
        json_fold_path (str): Path to JSON with k-fold subject splits.
        fold_index (int): Which fold to train (0-4).
        mri_data_dir (str): Directory containing preprocessed MRI data (*.npz files).
        mri_checkpoint_path (str, optional): Path to pre-trained MRI checkpoint for fine-tuning.
        num_classes (int): Number of output classes (default 2 for binary segmentation).
        batch_size (int): Batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for AdamW optimizer.
        epochs (int): Number of training epochs.
        log_root (str): Root directory for TensorBoard logs.
        num_workers (int): Number of DataLoader workers.
        test_mode (bool): If True, use limited data for quick testing.
        amp_enabled (bool): If True, use automatic mixed precision.
        patch_size (int): Size of patches to extract (only used if enable_patch_sampling=True).
        enable_patch_sampling (bool): If True, extract patches during training.
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load fold splits
    train_ids, val_ids = load_fold_split(json_fold_path, fold_index)
    print(f"Fold {fold_index}: {len(train_ids)} train subjects, {len(val_ids)} val subjects")
    
    # Limit to test data if requested
    if test_mode:
        train_ids = train_ids[:2]
        val_ids = val_ids[:1]
        print(f"Test mode: limited to {len(train_ids)} train and {len(val_ids)} val subjects")
    
    # Build case lists
    train_cases = build_cases(train_ids, mri_data_dir)
    val_cases = build_cases(val_ids, mri_data_dir)
    
    # Create datasets
    train_dataset = MRIDataset(
        cases=train_cases,
        image_dtype=torch.float16,
        gt_dtype=torch.uint8,
        return_float32=True,
        patch_size=patch_size,
        enable_patch_sampling=enable_patch_sampling,
        patch_center_mode="random",
        enable_augmentation=True,
    )
    
    val_dataset = MRIDataset(
        cases=val_cases,
        image_dtype=torch.float16,
        gt_dtype=torch.uint8,
        return_float32=True,
        patch_size=patch_size,
        enable_patch_sampling=enable_patch_sampling,
        patch_center_mode="gt_com",
        enable_augmentation=False,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 4,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    
    # Build model
    model = ResEncUNet_3D(input_channels=2, num_classes=num_classes).to(device)
    
    # Load pre-trained weights if provided
    if mri_checkpoint_path is not None:
        print(f"Loading pre-trained MRI checkpoint from {mri_checkpoint_path}")
        model.load_from_pth(
            path=mri_checkpoint_path,
            device=device,
            pretrained_in_channels=2,
            verbose=True,
        )
    
    # Loss function: combination of CE + Dice for segmentation
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_enabled))
    
    # TensorBoard logging
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"mri_fold{fold_index}_{datestr}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")
    
    # Save hyperparameters
    hparams = {
        "fold": fold_index,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "patch_size": patch_size,
        "enable_patch_sampling": enable_patch_sampling,
    }
    writer.add_text("hyperparameters", json.dumps(hparams, indent=2))
    
    # ----- Main Training Loop -----
    best_val_loss = float("inf")
    best_val_dice = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 60}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, writer
        )
        
        # Log epoch metrics
        writer.add_scalar("Train/Loss_epoch", train_loss, epoch)
        writer.add_scalar("Train/Dice_epoch", train_dice, epoch)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch, writer)
        
        writer.add_scalar("Val/Loss_epoch", val_loss, epoch)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, DSC: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, DSC: {val_dice:.4f}")
        print(f"Epoch duration: {epoch_duration:.1f}s")
        
        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_checkpoint_path = os.path.join(log_dir, "checkpoint_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                },
                best_checkpoint_path,
            )
            print(f"✓ New best validation DSC: {val_dice:.4f} (loss: {val_loss:.4f})")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(log_dir, "checkpoint_final.pth")
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_checkpoint_path,
    )
    
    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best validation - Loss: {best_val_loss:.4f}, DSC: {best_val_dice:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MRI-only U-Net segmentation model."
    )
    
    # Required paths
    parser.add_argument(
        "--fold_json", type=str, required=True,
        help="Path to JSON with k-fold subject splits."
    )
    parser.add_argument(
        "--mri_data_dir", type=str, required=True,
        help="Directory containing preprocessed MRI data (*.npz files)."
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        help="Which fold to train."
    )
    
    # Optional
    parser.add_argument(
        "--mri_checkpoint", type=str, default=None,
        help="Path to pre-trained MRI checkpoint for fine-tuning (optional)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5,
        help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--log_root", type=str, default="./data/tmp/runs/mri",
        help="Root directory for TensorBoard logs."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of DataLoader worker processes."
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of output classes (2 for binary segmentation)."
    )
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Size of patches to extract (only used if --enable_patch_sampling is set)."
    )
    parser.add_argument(
        "--enable_patch_sampling", action="store_true",
        help="If set, extract patches during training instead of using full volumes."
    )
    parser.add_argument(
        "--test_mode", action="store_true",
        help="If set, runs in test mode with limited data."
    )
    
    args = parser.parse_args()
    
    train(
        json_fold_path=args.fold_json,
        fold_index=args.fold,
        mri_data_dir=args.mri_data_dir,
        mri_checkpoint_path=args.mri_checkpoint,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_root=args.log_root,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
        patch_size=args.patch_size,
        enable_patch_sampling=args.enable_patch_sampling,
    )
