"""
training/eeg_spike_mil_regression_training.py

[EXPERIMENTAL / DEPRECATED]
Training script for direct coordinate regression of the epileptogenic zone using MIL.

Uses SpikeMILRegressor with heteroscedastic Gaussian NLL loss to predict normalized
MNI coordinates. This was a developmental step before the full multi-head model.

For production use, see eeg_spike_mil_mh_training.py.
"""

import os
import argparse
import time
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from models.eeg import SpikeMILRegressor
from datasets.eeg import (
    PatientMILSpikeDataset,
    mil_collate,
    load_split,
    load_regression_targets,
    find_patient_files,
)
from util import emit_run_fingerprint, EarlyStopping


def norm_attention_entropy(weights, mask=None, eps=1e-8):
    """Compute normalized attention entropy for regularization."""
    # Apply mask if provided
    if mask is not None:
        weights = weights * mask
        weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = weights / weights_sum
        K = mask.sum(dim=1).float().clamp_min(1.0)
    else:
        B, N = weights.shape
        K = torch.full((B,), float(N), device=weights.device)

    # Calculate normalized entropy
    prob = weights.clamp_min(eps)
    H = -(prob * prob.log()).sum(dim=1)
    H_max = K.log()
    norm = H / (H_max + eps)
    norm = torch.where(K <= 1, torch.ones_like(norm), norm)

    return norm


def gaussian_nll_loss(
    mu_hat,
    log_sigma_hat,
    mu_gt,
    log_sigma_min=-5.0,
    log_sigma_max=3.0,
):
    """Compute diagonal Gaussian NLL with clamped log-sigma for stability."""
    log_sigma = log_sigma_hat.clamp(min=log_sigma_min, max=log_sigma_max)
    inv_var = torch.exp(-2.0 * log_sigma)
    loss = 0.5 * ((mu_gt - mu_hat) ** 2 * inv_var + 2.0 * log_sigma)
    return loss.mean(), log_sigma


def train_one_epoch(model, dataloader, optimizer, device, scaler, epoch,
    writer=None, attn_entropy_lambda=0.005, sigma_reg_lambda=5e-4):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_euclidean_norm = 0.0
    total_euclidean_mm = 0.0
    total_sigma = 0.0
    total_sigma_components = [0.0, 0.0, 0.0]
    total = 0

    global_step = epoch * len(dataloader)

    for i, (X, mask, y) in enumerate(dataloader):
        # Move data to device
        X = X.to(device)
        mask = mask.to(device)
        mu_gt = y.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            mu_hat, log_sigma_hat, attn = model(X, mask)

            # Compute losses
            nll, log_sigma = gaussian_nll_loss(mu_hat, log_sigma_hat, mu_gt)
            sigma_reg = log_sigma.mean()

            # Combine losses with regularization
            loss = nll
            loss += sigma_reg_lambda * sigma_reg
            if attn is not None:
                loss -= attn_entropy_lambda * norm_attention_entropy(
                        attn.float(), mask
                ).mean()

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item() * X.size(0)
        # Calculate error in millimeters using per-axis normalization factors
        MNI_EXTENT_MM = torch.tensor([90, 126, 72], device=mu_hat.device, dtype=mu_hat.dtype)
        diff_normalized = mu_hat.detach() - mu_gt  # [batch_size, 3]
        diff_mm = diff_normalized * MNI_EXTENT_MM  # Scale each axis to millimeters
        euclidean_mm_per_sample = torch.norm(diff_mm, dim=1)
        euclidean_mm = euclidean_mm_per_sample.mean().item()  # Mean Cartesian distance in mm
        # Also calculate Euclidean distance in normalized space for logging
        euclidean_norm = torch.norm(mu_hat.detach() - mu_gt, dim=1).mean().item()

        # Total logs
        total_euclidean_norm += euclidean_norm * X.size(0)
        total_euclidean_mm += euclidean_mm * X.size(0)
        total_sigma += torch.exp(log_sigma.detach()).mean().item() * X.size(0)
        for dim in range(3):
            total_sigma_components[dim] += torch.exp(
                log_sigma.detach()[:, dim]
            ).mean().item() * X.size(0)
        total += X.size(0)

        # Log step metrics
        if writer is not None:
            step = global_step + i
            writer.add_scalar("Train/Loss_step", loss.item(), step)
            writer.add_scalar("Train/EuclideanNorm_step", euclidean_norm, step)
            writer.add_scalar("Train/Sigma_step", torch.exp(log_sigma).mean(), step)

            # Log attention entropy for first batch
            if writer is not None and epoch % 10 == 0 and i == 0:
                if attn is not None:
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

                # Also log histogram of predictions
                mu = mu_hat.detach().float().cpu()
                writer.add_histogram(f"Predictions/TrainX", mu[:, 0], epoch)
                writer.add_histogram(f"Predictions/TrainY", mu[:, 1], epoch)
                writer.add_histogram(f"Predictions/TrainZ", mu[:, 2], epoch)

                # Add sigma histogram
                sigma = torch.exp(log_sigma.detach()).float().cpu()
                writer.add_histogram(f"Sigma/TrainX", sigma[:, 0], epoch)
                writer.add_histogram(f"Sigma/TrainY", sigma[:, 1], epoch)
                writer.add_histogram(f"Sigma/TrainZ", sigma[:, 2], epoch)

                # Error spread in mm
                writer.add_histogram(
                    "Euclidean_mm/Train_hist",
                    euclidean_mm_per_sample.detach().float().cpu(),
                    epoch,
                )

    return (
            total_loss / total,
            total_euclidean_norm / total,
            total_euclidean_mm / total,
            total_sigma / total,
            [comp / total for comp in total_sigma_components],
    )


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer=None):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_euclidean_norm = 0.0
    total_euclidean_mm = 0.0
    total_sigma = 0.0
    total_sigma_components = [0.0, 0.0, 0.0]
    total = 0

    for i, (X, mask, y) in enumerate(dataloader):
        # Move data to device
        X = X.to(device)
        mask = mask.to(device)
        mu_gt = y.to(device)

        # Forward pass
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                mu_hat, log_sigma_hat, attn = model(X, mask)
                nll, log_sigma = gaussian_nll_loss(mu_hat, log_sigma_hat, mu_gt)
                loss = nll

        # Update metrics
        total_loss += loss.item() * X.size(0)
        # Calculate error in millimeters using per-axis normalization factors
        MNI_EXTENT_MM = torch.tensor([90, 126, 72], device=mu_hat.device, dtype=mu_hat.dtype)
        diff_normalized = mu_hat.detach() - mu_gt  # [batch_size, 3]
        diff_mm = diff_normalized * MNI_EXTENT_MM  # Scale each axis to millimeters
        euclidean_mm_per_sample = torch.norm(diff_mm, dim=1)
        euclidean_mm = euclidean_mm_per_sample.mean().item()  # Mean Cartesian distance in mm
        # Also calculate Euclidean distance in normalized space for logging
        euclidean_norm = torch.norm(mu_hat.detach() - mu_gt, dim=1).mean().item()
        # Total logs
        total_euclidean_norm += euclidean_norm * X.size(0)
        total_euclidean_mm += euclidean_mm * X.size(0)
        total_sigma += torch.exp(log_sigma.detach()).mean().item() * X.size(0)
        for dim in range(3):
            total_sigma_components[dim] += torch.exp(
                    log_sigma.detach()[:, dim]
            ).mean().item() * X.size(0)
        total += X.size(0)

        # Log attention entropy for first batch
        if writer is not None and epoch % 10 == 0 and i == 0:
            if attn is not None:
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

            # Also log histogram of predictions
            mu = mu_hat.detach().float().cpu()
            writer.add_histogram(f"Predictions/ValX", mu[:, 0], epoch)
            writer.add_histogram(f"Predictions/ValY", mu[:, 1], epoch)
            writer.add_histogram(f"Predictions/ValZ", mu[:, 2], epoch)

            # Add sigma histogram
            sigma = torch.exp(log_sigma.detach()).float().cpu()
            writer.add_histogram(f"Sigma/ValX", sigma[:, 0], epoch)
            writer.add_histogram(f"Sigma/ValY", sigma[:, 1], epoch)
            writer.add_histogram(f"Sigma/ValZ", sigma[:, 2], epoch)

            # Error spread in mm
            writer.add_histogram(
                "Euclidean_mm/Val_hist",
                euclidean_mm_per_sample.detach().float().cpu(),
                epoch,
            )

    return (
        total_loss / total,
        total_euclidean_norm / total,
        total_euclidean_mm / total,
        total_sigma / total,
        [comp / total for comp in total_sigma_components],
    )


def load_pretrained_encoder(model, checkpoint_path, freeze_mode="none"):
    """
    Load pretrained encoder weights from an ``encoder_best.pt`` checkpoint
    (saved by ``eeg_spike_encoder_pretraining.py``) into *model*.

    Parameters
    ----------
    model : nn.Module
        MIL model whose ``model.encoder`` sub-module will receive the weights.
    checkpoint_path : str
        Path to ``encoder_best.pt``.
    freeze_mode : {"none", "full", "temporal"}
        - ``"none"``     : weights are loaded; all parameters remain trainable.
        - ``"full"``     : the entire encoder is frozen.
        - ``"temporal"`` : only sub-modules of the encoder whose name contains
                           ``"temporal"`` are frozen (e.g. ``temporal_cnn`` /
                           ``temporal_projection`` in ``SpikeEncoder_T_S``).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    enc_state = ckpt.get("encoder_state_dict")
    if enc_state is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path!r} has no 'encoder_state_dict' key. "
            "Make sure it was saved by eeg_spike_encoder_pretraining.py."
        )
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=True)
    if missing:
        print(f"  [pretrained encoder] Missing keys: {missing}")
    if unexpected:
        print(f"  [pretrained encoder] Unexpected keys: {unexpected}")
    print(
        f"  Loaded pretrained encoder weights from {checkpoint_path!r} "
        f"(freeze_mode={freeze_mode!r})"
    )

    if freeze_mode == "full":
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder fully frozen.")

    elif freeze_mode == "temporal":
        frozen = []
        for name, module in model.encoder.named_children():
            if "temporal" in name:
                for param in module.parameters():
                    param.requires_grad = False
                frozen.append(name)
        if frozen:
            print(f"  Encoder temporal sub-modules frozen: {frozen}")
        else:
            print(
                f"  Warning: freeze_mode='temporal' but no sub-module named "
                f"'temporal*' found in {type(model.encoder).__name__}. "
                "Nothing was frozen."
            )


def train(json_split_path, fold_index, data_dir, json_targets_path,
                in_channels=21, max_spikes=32, min_spikes_per_patient=64, batch_size=4, lr=1e-4,
                weight_decay=1e-4, epochs=50, log_root="./runs", pooling=None,
                emb_dim=None, hidden=None, dropout=None, encoder_type=None,
                num_workers=0, test_mode=False,
                early_stopping=True, early_stopping_patience=150,
                early_stopping_min_delta=0.0, early_stopping_warmup=150,
                early_stopping_smoothing_window=10, restore_best_checkpoint=True,
                pretrained_encoder_path=None, freeze_encoder="none"):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load data splits and targets
    train_ids, val_ids = load_split(json_split_path, fold_index)
    print(f"Train subjects: {len(train_ids)}, Val subjects: {len(val_ids)}")
    target_dict = load_regression_targets(json_targets_path)

    # Prepare datasets
    train_ids, train_files, train_targets = find_patient_files(
            data_dir, train_ids, target_dict, test_mode=test_mode
    )
    val_ids, val_files, val_targets = find_patient_files(
            data_dir, val_ids, target_dict, test_mode=test_mode
    )
    
    # Create datasets and dataloaders
    train_dataset = PatientMILSpikeDataset(
            train_ids, train_files, train_targets,
            max_spikes_per_bag=max_spikes, training=True, min_spikes_per_patient=min_spikes_per_patient
    )
    val_dataset = PatientMILSpikeDataset(
            val_ids, val_files, val_targets,
            max_spikes_per_bag=max_spikes, training=False, min_spikes_per_patient=min_spikes_per_patient
    )
    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=mil_collate, pin_memory=True,
    )
    val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=mil_collate, pin_memory=True,
    )

    # Initialize model, optimizer, and scaler
    model_kwargs = {k: v for k, v in dict(emb_dim=emb_dim, hidden=hidden, dropout=dropout, pooling=pooling, encoder_type=encoder_type).items() if v is not None}
    print(f"Model kwargs (overrides): {model_kwargs}")
    model = SpikeMILRegressor(in_channels=in_channels, **model_kwargs).to(device)

    if pretrained_encoder_path is not None:
        load_pretrained_encoder(model, pretrained_encoder_path, freeze_mode=freeze_encoder)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
            log_root, f"eeg_mil_regression_fold{fold_index}_{datestr}"
    )
    os.makedirs(log_dir, exist_ok=True)

    run_fingerprint_payload = emit_run_fingerprint(
        script_name="eeg_spike_mil_regression_training",
        train_config={
            "json_split_path": json_split_path,
            "fold_index": fold_index,
            "data_dir": data_dir,
            "json_targets_path": json_targets_path,
            "in_channels": in_channels,
            "max_spikes": max_spikes,
            "min_spikes_per_patient": min_spikes_per_patient,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "log_root": log_root,
            "num_workers": num_workers,
            "test_mode": test_mode,
            "pooling": pooling,
            "encoder_type": encoder_type,
            "emb_dim": emb_dim,
            "hidden": hidden,
            "dropout": dropout,
            "pretrained_encoder_path": pretrained_encoder_path,
            "freeze_encoder": freeze_encoder,
        },
        model_kwargs=model_kwargs,
        effective_model_config={
            "model_class": model.__class__.__name__,
            "emb_dim": model.emb_dim,
            "hidden": model.hidden,
            "dropout": model.dropout,
            "pooling": model.pooling,
            "encoder_type": model.encoder_type,
        },
        extra={"device": str(device)},
    )

    # Persist run settings/fingerprint alongside TensorBoard and inference outputs.
    run_fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(run_fingerprint_path, "w") as f:
        json.dump(run_fingerprint_payload, f, indent=2)
    print(f"Saved run fingerprint to: {run_fingerprint_path}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Setup logging
    writer = SummaryWriter(log_dir=log_dir)
    print("Logging to:", log_dir)

    # Checkpoint setup
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping setup
    es = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        warmup=early_stopping_warmup,
        smoothing_window=early_stopping_smoothing_window,
        enabled=early_stopping,
    )
    best_val_euclidean_mm = float("inf")

    # Checkpoint paths
    ckpt_last = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    ckpt_best_raw = os.path.join(checkpoint_dir, "checkpoint_best_raw_val_loss.pt")
    ckpt_best_smoothed = os.path.join(checkpoint_dir, "checkpoint_best_smoothed_val_loss.pt")

    def _make_checkpoint(epoch_1based, val_loss_val, smoothed_val_loss_val, es_info):
        scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
        return {
            "epoch": epoch_1based,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": scaler_state,
            "val_loss": val_loss_val,
            "smoothed_val_loss": smoothed_val_loss_val,
            "best_raw_val_loss": es_info["best_raw_val_loss"],
            "best_smoothed_val_loss": es_info["best_smoothed_val_loss"],
            "args": {
                "in_channels": in_channels,
                "emb_dim": model.emb_dim,
                "hidden": model.hidden,
                "dropout": model.dropout,
                "pooling": model.pooling,
                "encoder_type": model.encoder_type,
                "early_stopping": early_stopping,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_warmup": early_stopping_warmup,
                "early_stopping_smoothing_window": early_stopping_smoothing_window,
                "restore_best_checkpoint": restore_best_checkpoint,
            },
        }

    # Training loop
    stopped_early = False
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_euclidean_norm, train_euclidean_mm, train_sigma, train_sigma_components = train_one_epoch(
                model, train_loader, optimizer, device, scaler, epoch, writer
        )
        val_loss, val_euclidean_norm, val_euclidean_mm, val_sigma, val_sigma_components = validate(
                model, val_loader, device, epoch, writer
        )

        # Update early stopping state
        es_info = es.update(epoch, val_loss)
        smoothed_val_loss = es_info["smoothed_val_loss"]

        # Track best euclidean for print summary
        if es_info["raw_improved"]:
            best_val_euclidean_mm = val_euclidean_mm
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Print epoch metrics
        best_epoch_1based = (es_info["best_epoch"] + 1) if es_info["best_epoch"] is not None else None
        lr_current = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Train - Loss: {train_loss:.4f}, Euclidean (mm): {train_euclidean_mm:.4f}, Sigma: {train_sigma:.4f}"
        )
        print(
            f"Val   - Loss: {val_loss:.4f} (smoothed: {smoothed_val_loss:.4f}), "
            f"Euclidean (mm): {val_euclidean_mm:.4f}, Sigma: {val_sigma:.4f}"
        )
        print(
            f"ES    - best_smoothed: {es_info['best_smoothed_val_loss']:.6f} @ epoch {best_epoch_1based}, "
            f"best_raw: {es_info['best_raw_val_loss']:.6f}, "
            f"no_improve: {es_info['epochs_without_improvement']}/{early_stopping_patience}, "
            f"lr: {lr_current:.2e}"
        )
        print(f"Epoch duration: {epoch_duration:.1f}s")

        # Log epoch metrics
        writer.add_scalars(
                "Loss", {"Train": train_loss, "Val": val_loss}, epoch
        )
        writer.add_scalars(
            "Euclidean_norm", {"Train": train_euclidean_norm, "Val": val_euclidean_norm}, epoch
        )
        writer.add_scalars(
            "Euclidean_mm", {"Train": train_euclidean_mm, "Val": val_euclidean_mm}, epoch
        )
        writer.add_scalars(
                "Sigma", {"Train": train_sigma, "Val": val_sigma}, epoch
        )
        writer.add_scalars(
                "Sigma_comp/Train",
                {"X": train_sigma_components[0], "Y": train_sigma_components[1], "Z": train_sigma_components[2]},
                epoch,
        )
        writer.add_scalars(
            "Sigma_comp/Val",
            {"X": val_sigma_components[0], "Y": val_sigma_components[1], "Z": val_sigma_components[2]},
            epoch,
        )
        writer.add_scalar("val/loss_raw", val_loss, epoch)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss, epoch)
        writer.add_scalar("early_stopping/best_smoothed_val_loss", es_info["best_smoothed_val_loss"], epoch)
        writer.add_scalar("early_stopping/epochs_without_improvement", es_info["epochs_without_improvement"], epoch)
        if best_epoch_1based is not None:
            writer.add_scalar("early_stopping/best_epoch", best_epoch_1based, epoch)

        ckpt_data = _make_checkpoint(epoch + 1, val_loss, smoothed_val_loss, es_info)

        # Save last checkpoint every epoch
        torch.save(ckpt_data, ckpt_last)

        # Save best raw-loss checkpoint
        if es_info["raw_improved"]:
            torch.save(ckpt_data, ckpt_best_raw)
            print(f"✓ New best raw val loss: {val_loss:.4f} (Euclidean: {val_euclidean_mm:.4f} mm)")

        # Save best smoothed-loss checkpoint
        if es_info["improved"]:
            torch.save(ckpt_data, ckpt_best_smoothed)
            print(f"✓ New best smoothed val loss: {smoothed_val_loss:.6f}")

        # Early stopping check
        if es_info["should_stop"]:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1}. "
                f"Best smoothed validation loss was {es_info['best_smoothed_val_loss']:.6f} "
                f"at epoch {best_epoch_1based}."
            )
            stopped_early = True
            break

    # Restore best smoothed checkpoint before inference/export
    if restore_best_checkpoint and os.path.exists(ckpt_best_smoothed):
        restored = torch.load(ckpt_best_smoothed, map_location=device)
        model.load_state_dict(restored["model_state_dict"])
        restored_epoch = restored["epoch"]
        print(f"\nRestored best smoothed-loss checkpoint (epoch {restored_epoch}).")

    # Save final (post-restore) checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else None
    torch.save(
            {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler_state,
            "best_epoch": (es.best_epoch + 1) if es.best_epoch is not None else None,
            "best_raw_val_loss": es.best_raw_val_loss,
            "best_smoothed_val_loss": es.best_smoothed_val_loss,
            "stopped_early": stopped_early,
            },
            final_ckpt_path,
    )
    print(f"Saved final checkpoint to: {final_ckpt_path}")
    writer.close()
    print(f"\n{'=' * 60}")
    print(f"Training complete!" + (" (early stop)" if stopped_early else ""))
    print(f"Best raw val loss:      {es.best_raw_val_loss:.4f} @ epoch {(es.best_raw_epoch + 1) if es.best_raw_epoch is not None else 'N/A'}")
    print(f"Best smoothed val loss: {es.best_smoothed_val_loss:.6f} @ epoch {(es.best_epoch + 1) if es.best_epoch is not None else 'N/A'}")
    print(f"Best Euclidean (mm):    {best_val_euclidean_mm:.4f}")
    print(f"{'=' * 60}")

    # Return paths for inference
    return {
            "best_checkpoint": ckpt_best_smoothed,
            "best_raw_checkpoint": ckpt_best_raw,
            "last_checkpoint": ckpt_last,
            "final_checkpoint": final_ckpt_path,
            "log_dir": log_dir,
    }


@torch.no_grad()
def infer_predictions(
    model,
    dataloader,
    case_ids,
    device,
):
    """
    Run inference on a dataset and collect predictions.
    
    Returns a dict: {case_id: {"mu": [x, y, z], "sigma": [sx, sy, sz]}}
    """
    model.eval()
    predictions = {}
    case_idx = 0
    
    for batch_idx, (X, mask, y) in enumerate(dataloader):
        X = X.to(device)
        mask = mask.to(device)
        
        mu_hat, log_sigma_hat, _ = model(X, mask)

        # Use a running pointer to preserve ID alignment for variable batch sizes.
        batch_size = X.size(0)
        for i in range(batch_size):
            if case_idx >= len(case_ids):
                break

            case_id = case_ids[case_idx]
            mu = mu_hat[i].cpu().numpy().tolist()
            sigma = torch.exp(log_sigma_hat[i]).cpu().numpy().tolist()

            predictions[case_id] = {
                "mu": mu,
                "sigma": sigma,
            }
            case_idx += 1
    
    return predictions


def generate_prior_niftis(predictions, mri_npy_dir, output_dir, clamp_min_sigma_vox=1.0):
    """
    Generate Gaussian prior NIfTI files from EEG coordinate predictions.

    For each patient ID in ``predictions``, loads the corresponding MRI npz file
    (expected at ``{mri_npy_dir}/{patient_id}.npz``) to obtain the affine and volume
    shape, then builds a 3-D Gaussian blob in voxel space and saves it as
    ``{output_dir}/{patient_id}_prior.nii.gz``.

    Parameters
    ----------
    predictions : dict
        Maps patient_id -> {"mu": [x, y, z] (normalized MNI), "sigma": float or [sx, sy, sz]}
    mri_npy_dir : str
        Directory containing ``{patient_id}.npz`` files with ``"image"`` (C,D,H,W)
        and ``"affine"`` (4,4) arrays.
    output_dir : str
        Directory in which the generated NIfTI files are saved.
    clamp_min_sigma_vox : float
        Minimum Gaussian sigma in voxels (passed to gaussian_prior_ijk).
    """
    import nibabel as nib
    from datasets.multimodal import norm_to_mm, mm_to_vox, sigma_mm_to_vox, gaussian_prior_ijk

    os.makedirs(output_dir, exist_ok=True)
    saved, skipped = 0, 0

    for patient_id, pred in predictions.items():
        npy_path = os.path.join(mri_npy_dir, f"{patient_id}_preproc.npz")
        if not os.path.exists(npy_path):
            print(f"  [skip] No MRI npz found for {patient_id} at {npy_path}")
            skipped += 1
            continue

        try:
            npz = np.load(npy_path, allow_pickle=True)
            img = npz["image"]          # (2, D, H, W)
            affine = npz["affine"].astype(np.float32)
            npz.close()
        except Exception as e:
            print(f"  [skip] Error loading {npy_path}: {e}")
            skipped += 1
            continue

        _, D, H, W = img.shape

        # Convert normalized MNI prediction to voxel-space Gaussian parameters
        mu_norm = np.asarray(pred["mu"], dtype=np.float32)
        sigma = pred["sigma"]
        sigma_norm = (
            np.array([float(sigma)] * 3, dtype=np.float32)
            if np.isscalar(sigma)
            else np.asarray(sigma, dtype=np.float32)
        )

        mu_mm, sig_mm = norm_to_mm(mu_norm, sigma_norm)
        mu_ijk = mm_to_vox(mu_mm, affine)
        sig_ijk = sigma_mm_to_vox(sig_mm, affine)

        prior = gaussian_prior_ijk(
            (D, H, W), mu_ijk, sig_ijk, clamp_min_vox=clamp_min_sigma_vox
        )

        # Mask prior to brain region
        img_mask = (img > 1e-5).all(axis=0).astype(np.float32)
        prior = prior * img_mask

        nii_path = os.path.join(output_dir, f"{patient_id}_prior.nii.gz")
        nib.save(nib.Nifti1Image(prior, affine), nii_path)
        saved += 1

    print(f"  Saved {saved} prior NIfTIs to {output_dir} ({skipped} skipped)")


def run_inference(
    checkpoint_path,
    json_split_path,
    fold_index,
    data_dir,
    json_targets_path,
    output_dir,
    in_channels=21,
    max_spikes=32,
    min_spikes_per_patient=64,
    batch_size=4,
    num_workers=0,
    pooling=None,
    encoder_type=None,
    generate_niftis=False,
    mri_npy_dir=None,
    test_mode=False,
    infer_test_set=False
):
    """
    Load trained model and run inference on train/val splits.
    Optionally run inference on test set if infer_test_set=True.
    Outputs JSON files: train_predictions.json, val_predictions.json, and optionally test_predictions.json.

    Optionally generates per-patient Gaussian prior NIfTI files from the
    predicted (mu, sigma) pairs when ``generate_niftis=True``.  In that case
    ``mri_npy_dir`` must point to the directory that contains the preprocessed
    ``{patient_id}.npz`` files (same format as expected by UNetWithPriorDataset).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model — restore arch from checkpoint args, explicit params override.
    saved_args = checkpoint.get("args", {})
    model_kwargs = {k: v for k, v in dict(
        emb_dim=saved_args.get("emb_dim", None),
        hidden=saved_args.get("hidden", None),
        dropout=saved_args.get("dropout", None),
        pooling=saved_args.get("pooling", pooling),
        encoder_type=saved_args.get("encoder_type", encoder_type),
    ).items() if v is not None}
    for k, v in dict(pooling=pooling, encoder_type=encoder_type).items():
        if v is not None:
            model_kwargs[k] = v
    print(f"Model kwargs (from checkpoint + overrides): {model_kwargs}")
    model = SpikeMILRegressor(in_channels=in_channels, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    
    # Load data splits and targets
    train_ids, val_ids = load_split(json_split_path, fold_index)
    target_dict = load_regression_targets(json_targets_path)
    
    # Load test IDs if inference on test set is enabled
    test_ids = []
    if infer_test_set:
        with open(json_split_path, "r") as f:
            payload = json.load(f)
        fold_payload = payload["folds"]
        fold_key = f"fold_{fold_index}"
        test_ids = fold_payload[fold_key].get("test_ids", [])
    
    # Prepare datasets
    train_ids_found, train_files, train_targets = find_patient_files(
        data_dir, train_ids, target_dict, test_mode=test_mode
    )
    val_ids_found, val_files, val_targets = find_patient_files(
        data_dir, val_ids, target_dict, test_mode=test_mode
    )
    test_ids_found, test_files, test_targets = ([], [], []) if not infer_test_set else find_patient_files(
        data_dir, test_ids, target_dict, test_mode=test_mode
    )
    
    # Create dataloaders
    train_dataset = PatientMILSpikeDataset(
        train_ids_found, train_files, train_targets,
        max_spikes_per_bag=max_spikes, training=False, min_spikes_per_patient=min_spikes_per_patient
    )
    val_dataset = PatientMILSpikeDataset(
        val_ids_found, val_files, val_targets,
        max_spikes_per_bag=max_spikes, training=False, min_spikes_per_patient=min_spikes_per_patient
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=mil_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=mil_collate, pin_memory=True,
    )
    
    # Run inference (use dataset.patient_ids which reflect any min_spikes filtering)
    print("\nRunning inference on training set...")
    train_preds = infer_predictions(model, train_loader, train_dataset.patient_ids, device)
    print(f"Got predictions for {len(train_preds)} training cases")
    
    print("Running inference on validation set...")
    val_preds = infer_predictions(model, val_loader, val_dataset.patient_ids, device)
    print(f"Got predictions for {len(val_preds)} validation cases")
    
    # Run inference on test set if enabled
    test_preds = {}
    if infer_test_set and len(test_ids_found) > 0:
        test_dataset = PatientMILSpikeDataset(
            test_ids_found, test_files, test_targets,
            min_spikes_per_patient=min_spikes_per_patient,
            training=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=mil_collate, pin_memory=True,
        )
        print("Running inference on test set...")
        test_preds = infer_predictions(model, test_loader, test_dataset.patient_ids, device)
        print(f"Got predictions for {len(test_preds)} test cases")

    # Calculate validation metrics
    val_metrics = {}
    target_dict_val = {pid: label for pid, label in zip(val_dataset.patient_ids, val_dataset.patient_labels)}

    for case_id in val_preds:
        pred_mu = torch.tensor(val_preds[case_id]["mu"])
        pred_sigma = torch.tensor(val_preds[case_id]["sigma"])
        gt_mu = target_dict_val[case_id]["mu"]
        
        # Euclidean distance in normalized space
        euclidean_norm = torch.norm(pred_mu - gt_mu).item()
        
        # Euclidean distance in millimeters
        MNI_EXTENT_MM = torch.tensor([90, 126, 72], dtype=torch.float32)
        diff_mm = (pred_mu - gt_mu) * MNI_EXTENT_MM
        euclidean_mm = torch.norm(diff_mm).item()
        
        val_metrics[case_id] = {
            "euclidean_norm": euclidean_norm,
            "euclidean_mm": euclidean_mm,
            "pred_mu": val_preds[case_id]["mu"],
            "gt_mu": gt_mu.tolist(),
            "sigma": val_preds[case_id]["sigma"],
        }

    # Save results and metrics to JSON files
    os.makedirs(output_dir, exist_ok=True)

    inference_settings = {
        "checkpoint_path": checkpoint_path,
        "json_split_path": json_split_path,
        "fold_index": fold_index,
        "data_dir": data_dir,
        "json_targets_path": json_targets_path,
        "output_dir": output_dir,
        "in_channels": in_channels,
        "max_spikes": max_spikes,
        "min_spikes_per_patient": min_spikes_per_patient,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pooling": pooling,
        "encoder_type": encoder_type,
        "generate_niftis": generate_niftis,
        "mri_npy_dir": mri_npy_dir,
        "infer_test_set": bool(infer_test_set),
        "test_mode": test_mode,
        "resolved_model_kwargs": model_kwargs,
        "saved_train_args": saved_args,
    }
    inference_settings_path = os.path.join(output_dir, "inference_settings.json")
    with open(inference_settings_path, "w") as f:
        json.dump(inference_settings, f, indent=2)
    print(f"Saved inference settings to: {inference_settings_path}")

    # Save outputs
    output_path = os.path.join(output_dir, "predictions.json")
    combined_preds = {
        "train": train_preds,
        "val": val_preds,
    }
    if infer_test_set and test_preds:
        combined_preds["test"] = test_preds
    with open(output_path, "w") as f:
        json.dump(combined_preds, f, indent=2)
    print(f"Saved predictions to: {output_path}")
    # Save validation metrics
    metrics_path = os.path.join(output_dir, "validation.json")
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved validation metrics to: {metrics_path}")

    # Optionally generate Gaussian prior NIfTI files
    if generate_niftis:
        if mri_npy_dir is None:
            print(
                "\033[38;5;208mWarning: generate_niftis=True but mri_npy_dir is None. "
                "Skipping NIfTI generation.\033[0m"
            )
        else:
            print("\nGenerating prior NIfTIs for training set...")
            generate_prior_niftis(
                train_preds,
                mri_npy_dir=mri_npy_dir,
                output_dir=os.path.join(output_dir, "prior_niftis", "train"),
            )
            print("Generating prior NIfTIs for validation set...")
            generate_prior_niftis(
                val_preds,
                mri_npy_dir=mri_npy_dir,
                output_dir=os.path.join(output_dir, "prior_niftis", "val"),
            )

    return train_preds, val_preds


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a MIL regressor on EEG spike data with configurable pooling."
    )

    # Required arguments
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to JSON file containing k-fold subject splits.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_spikes.npy files.")
    parser.add_argument("--targets_json", type=str, required=True,
                        help="JSON mapping patient_id → regression targets.")
    
    # Optional arguments
    parser.add_argument("--log_root", type=str, default="./runs",
                        help="Base directory for TensorBoard logs.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold from the JSON splits to use.")
    parser.add_argument("--in_channels", type=int, default=21,
                        help="Number of EEG channels in input.")
    parser.add_argument("--max_spikes", type=int, default=32,
                        help="Max spikes sampled per patient per epoch.")
    parser.add_argument("--min_spikes_per_patient", type=int, default=64,
                        help="Minimum number of spikes required for a patient to be included in training/validation.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Patients per batch.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["attention", "mean", "mean-max-topk"],
                        help="MIL pooling mode. Defaults to model class default if unset.")
    parser.add_argument("--encoder_type", type=str, default=None,
                        choices=["eegnet", "cnn1d", "t_s_cnn"],
                        help="Spike encoder type. Defaults to model class default if unset.")
    parser.add_argument("--test_mode", action="store_true",
                        help="If set, runs in test mode with limited data.")
    parser.add_argument("--skip_inference", action="store_true",
                        help="If set, skips inference after training.")
    parser.add_argument("--infer_test_set", action="store_true",
                        help="If set, run inference on test set in addition to train/val.")
    parser.add_argument("--generate_niftis", action="store_true",
                        help="If set, generate Gaussian prior NIfTI files from inference predictions.")
    parser.add_argument("--mri_npy_dir", type=str, default=None,
                        help="Directory containing {patient_id}.npz MRI files (with 'image' and 'affine'). "
                             "Required when --generate_niftis is set.")

    # Early stopping arguments
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable early stopping (default: True). Use --no-early-stopping to disable.")
    parser.add_argument("--early_stopping_patience", type=int, default=150,
                        help="Epochs without improvement before stopping (default: 150).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum absolute improvement in smoothed val loss (default: 0.0).")
    parser.add_argument("--early_stopping_warmup", type=int, default=150,
                        help="Epochs before early stopping may trigger (default: 150).")
    parser.add_argument("--early_stopping_smoothing_window", type=int, default=10,
                        help="Rolling window size for smoothed val loss (default: 10).")
    parser.add_argument("--restore_best_checkpoint", action=argparse.BooleanOptionalAction, default=True,
                        help="Restore best smoothed-loss checkpoint before inference (default: True).")

    # Pretrained encoder arguments
    parser.add_argument("--pretrained_encoder_path", type=str, default=None,
                        help="Path to encoder_best.pt from spike-encoder pretraining. "
                             "If set, the encoder weights are loaded before training.")
    parser.add_argument("--freeze_encoder", type=str, default="none",
                        choices=["none", "full", "temporal"],
                        help="Freeze pretrained encoder weights: 'none' (no freeze), "
                             "'full' (entire encoder), or 'temporal' (temporal sub-modules only).")

    args = parser.parse_args()

    # Run training
    train_results = train(
        json_split_path=args.splits_json,
        fold_index=args.fold,
        data_dir=args.data_dir,
        json_targets_path=args.targets_json,
        in_channels=args.in_channels,
        max_spikes=args.max_spikes,
        min_spikes_per_patient=args.min_spikes_per_patient,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_root=args.log_root,
        pooling=args.pooling,
        encoder_type=args.encoder_type,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_warmup=args.early_stopping_warmup,
        early_stopping_smoothing_window=args.early_stopping_smoothing_window,
        restore_best_checkpoint=args.restore_best_checkpoint,
        pretrained_encoder_path=args.pretrained_encoder_path,
        freeze_encoder=args.freeze_encoder,
    )

    # Run inference with best checkpoint if not skipped
    if not args.skip_inference and train_results is not None:
        best_ckpt = train_results["best_checkpoint"]
        log_dir = train_results["log_dir"]
        
        if os.path.exists(best_ckpt):
            print("\n" + "=" * 80)
            print("RUNNING INFERENCE WITH BEST CHECKPOINT")
            print("=" * 80)
            run_inference(
                checkpoint_path=best_ckpt,
                json_split_path=args.splits_json,
                fold_index=args.fold,
                data_dir=args.data_dir,
                json_targets_path=args.targets_json,
                output_dir=log_dir,
                in_channels=args.in_channels,
                max_spikes=args.max_spikes,
                min_spikes_per_patient=args.min_spikes_per_patient,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pooling=args.pooling,
                encoder_type=args.encoder_type,
                generate_niftis=args.generate_niftis,
                mri_npy_dir=args.mri_npy_dir,
                test_mode=args.test_mode,
                infer_test_set=args.infer_test_set,
            )
        else:
            print(f"Warning: Best checkpoint not found at {best_ckpt}")
    elif args.skip_inference:
        print("\nInference skipped (--skip_inference flag set).")
