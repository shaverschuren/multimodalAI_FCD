"""
training/eeg_spike_encoder_pretraining.py

[AUXILIARY]
Self-supervised spike encoder pretraining script.

Pre-trains the SpikeEncoder_T_S backbone on the full spike corpus using a supervised
auxiliary task (Persyst score regression + detected channel classification) before
downstream MIL training. Pretrained weights can be loaded into SpikeMILModel
via --pretrained_encoder_path in eeg_spike_mil_mh_training.py.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.eeg import (
    CHANNEL_LABEL_TO_INT,
    FlatSpikeEncoderPretrainDataset,
    build_flat_pretrain_datasets,
    flat_spike_pretrain_collate,
)
from models.eeg import (
    SpikeEncoderPretrainModel,
)
from util import EarlyStopping, emit_run_fingerprint


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


def extract_patient_ids(index_table) -> list[str]:
    """Collect unique patient IDs from the pretraining index table."""
    if index_table is None:
        return []
    if hasattr(index_table, "columns") and "patient_id" in index_table.columns:
        values = index_table["patient_id"].tolist()
    else:
        values = [row.get("patient_id") for row in index_table if isinstance(row, dict)]
    return sorted({str(pid) for pid in values if pid is not None})

def make_channel_multilabel_target(
    channel_target: torch.Tensor,
    n_channels: int,
) -> torch.Tensor:
    """
    Convert integer channel indices (or -1 sentinel) to a one-hot float
    matrix suitable for BCEWithLogitsLoss.

    channel_target : (B,) LongTensor; valid class in [0, n_channels), else -1
    Returns        : (B, n_channels) float32 tensor with 0/1 entries
    """
    y = torch.zeros(
        channel_target.shape[0], n_channels, device=channel_target.device
    )
    valid = (channel_target >= 0) & (channel_target < n_channels)
    if valid.any():
        y[valid, channel_target[valid]] = 1.0
    return y


def channel_top1_accuracy(
    channel_logits: torch.Tensor,
    channel_target: torch.Tensor,
) -> torch.Tensor:
    """Top-1 accuracy on rows that have a valid channel label."""
    valid = (channel_target >= 0) & (channel_target < channel_logits.shape[-1])
    if valid.sum() == 0:
        return torch.tensor(0.0, device=channel_logits.device)
    pred = channel_logits[valid].argmax(dim=-1)
    return (pred == channel_target[valid]).float().mean()


def channel_zero_accuracy(
    channel_logits: torch.Tensor,
    channel_target: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Fraction of non-spike rows (channel_target == -1) where all channel
    logits are correctly below *threshold* (no spurious positive prediction).
    """
    non_spike = channel_target < 0
    if non_spike.sum() == 0:
        return torch.tensor(1.0, device=channel_logits.device)
    no_pos_pred = (channel_logits[non_spike].max(dim=-1).values <= threshold)
    return no_pos_pred.float().mean()


# ---------------------------------------------------------------------------
# Subject-adversarial lambda schedule (DANN-style ramp)
# ---------------------------------------------------------------------------

def compute_subject_adv_lambda(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    lambda_max: float,
) -> float:
    """Return the gradient-reversal scaling factor for the current epoch.

    During *warmup_epochs* the lambda is 0 (encoder trains normally).
    After that it ramps from 0 to *lambda_max* over *ramp_epochs* using a
    smooth DANN-style sigmoid curve.
    """
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return lambda_max
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: SpikeEncoderPretrainModel,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion_perc: nn.Module,
    criterion_ch: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    lambda_perception: float,
    lambda_channel: float,
    training: bool,
    writer: Optional[SummaryWriter] = None,
    n_channels: int = 21,
    log_step_interval: int = 1,
    grad_clip_norm: float = 1.0,
    step_offset: int = 0,
    amp_enabled: bool = True,
    # Subject-adversarial options (ignored when use_subject_adversary=False)
    use_subject_adversary: bool = False,
    subject_adv_lambda: float = 0.0,
    subject_to_idx: Optional[dict] = None,
    subject_adv_loss_weight: float = 1.0,
):
    """Single train or validation epoch. Returns dict of aggregated metrics."""
    if training:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if training else torch.no_grad()

    agg = dict(
        loss=0.0, loss_perc=0.0, loss_ch=0.0,
        perc_mae=0.0, perc_rmse_sq=0.0,
        ch_top1=0.0, ch_zero=0.0,
        perc_pred_det=0.0, perc_pred_nonspike=0.0,
        perc_tgt_det=0.0, perc_tgt_nonspike=0.0,
        subject_loss=0.0, subject_acc=0.0,
        n=0, n_det=0, n_nonspike=0,
    )

    ch_pred_sum = torch.zeros(n_channels, device=device, dtype=torch.float32)
    ch_tgt_sum = torch.zeros(n_channels, device=device, dtype=torch.float32)

    all_perc_preds: list[torch.Tensor] = []
    all_ch_logits: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []

    with context:
        for step, (X, targets) in enumerate(dataloader):
            X = X.to(device)                                   # (B, C, L)
            perc_tgt    = targets["perception"].to(device)     # (B,)
            ch_tgt_int  = targets["channel_target"].to(device) # (B,) long
            spike_tgt   = targets["spike_target"].to(device)   # (B,)

            # Convert patient_id strings to subject indices when adversary is on
            subject_idx = None
            if use_subject_adversary and subject_to_idx is not None:
                subject_idx = torch.tensor(
                    [subject_to_idx.get(pid, -1) for pid in targets["patient_id"]],
                    dtype=torch.long, device=device,
                )

            if training:
                optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
                perc_pred, ch_logits, emb, subject_logits = model(
                    X, subject_adv_lambda=subject_adv_lambda
                )

                ch_target_ml = make_channel_multilabel_target(ch_tgt_int, n_channels)

                loss_perc = criterion_perc(perc_pred, perc_tgt)
                # Per-element BCE → mean over channel dim → (B,); scale by
                # perception target so low-confidence / non-spike rows
                # contribute proportionally less to the channel loss.
                # Floor at 0.1 ensures non-spike rows still push spurious
                # logits towards zero (avoids zero gradient there).
                # nan_to_num guards against inf*0=NaN when logits overflow fp16.
                bce_per_sample = criterion_ch(ch_logits, ch_target_ml).mean(dim=-1)
                spike_weight = perc_tgt.clamp(min=0.1)
                loss_ch   = (torch.nan_to_num(bce_per_sample, nan=0.0, posinf=0.0)
                             * spike_weight).mean()
                loss      = lambda_perception * loss_perc + lambda_channel * loss_ch

                if use_subject_adversary and subject_logits is not None and subject_idx is not None:
                    valid_subject = subject_idx >= 0
                    if valid_subject.any():
                        subject_loss = F.cross_entropy(subject_logits[valid_subject], subject_idx[valid_subject])
                        loss = loss + subject_adv_loss_weight * subject_loss
                    else:
                        subject_loss = torch.tensor(0.0, device=device)
                else:
                    subject_loss = torch.tensor(0.0, device=device)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            B = X.size(0)

            # Accumulate
            agg["loss"]      += loss.item()      * B
            agg["loss_perc"] += loss_perc.item() * B
            agg["loss_ch"]   += loss_ch.item()   * B
            agg["n"]         += B

            with torch.no_grad():
                perc_mae     = (perc_pred - perc_tgt).abs().mean().item()
                perc_rmse_sq = ((perc_pred - perc_tgt) ** 2).mean().item()
                ch_top1      = channel_top1_accuracy(ch_logits, ch_tgt_int).item()
                ch_zero      = channel_zero_accuracy(ch_logits, ch_tgt_int).item()

                agg["perc_mae"]      += perc_mae      * B
                agg["perc_rmse_sq"]  += perc_rmse_sq  * B
                agg["ch_top1"]       += ch_top1        * B
                agg["ch_zero"]       += ch_zero        * B

                if use_subject_adversary and subject_logits is not None and subject_idx is not None:
                    valid_subject = subject_idx >= 0
                    if valid_subject.any():
                        valid_B = valid_subject.sum().item()
                        agg["subject_loss"] += subject_loss.item() * valid_B
                        s_pred = subject_logits[valid_subject].argmax(dim=1)
                        agg["subject_acc"]  += (s_pred == subject_idx[valid_subject]).float().mean().item() * valid_B

                # Channel-wise epoch means for diagnostics.
                ch_pred_sum += torch.sigmoid(ch_logits.detach()).sum(dim=0).float()
                ch_tgt_sum += ch_target_ml.detach().sum(dim=0).float()

                # Detection vs non-spike perception tracking
                det_mask      = spike_tgt > 0.5
                nonspike_mask = ~det_mask
                if det_mask.any():
                    agg["perc_pred_det"]  += perc_pred[det_mask].mean().item()  * det_mask.sum().item()
                    agg["perc_tgt_det"]   += perc_tgt[det_mask].mean().item()   * det_mask.sum().item()
                    agg["n_det"]          += det_mask.sum().item()
                if nonspike_mask.any():
                    agg["perc_pred_nonspike"] += perc_pred[nonspike_mask].mean().item() * nonspike_mask.sum().item()
                    agg["perc_tgt_nonspike"]  += perc_tgt[nonspike_mask].mean().item()  * nonspike_mask.sum().item()
                    agg["n_nonspike"]         += nonspike_mask.sum().item()

                # Collect for histograms
                all_perc_preds.append(perc_pred.detach().cpu())
                all_ch_logits.append(ch_logits.detach().cpu())
                if emb is not None:
                    all_embeddings.append(emb.detach().cpu())

            # Per-step TensorBoard logging (training only)
            if (
                training
                and writer is not None
                and log_step_interval > 0
                and (step % log_step_interval == 0)
            ):
                gs = step_offset + step + 1

                # Skip pathological non-finite values to keep TB traces clean.
                if np.isfinite(loss.item()):
                    writer.add_scalar("loss/total_step", loss.item(), gs)
                if np.isfinite(loss_perc.item()):
                    writer.add_scalar("loss/perception_step", loss_perc.item(), gs)
                if np.isfinite(loss_ch.item()):
                    writer.add_scalar("loss/channel_step", loss_ch.item(), gs)
                if np.isfinite(perc_mae):
                    writer.add_scalar("metrics/perception_mae_step", perc_mae, gs)
                if np.isfinite(ch_top1):
                    writer.add_scalar("metrics/channel_top1_acc_step", ch_top1, gs)
                if use_subject_adversary and np.isfinite(subject_loss.item()):
                    writer.add_scalar("train/subject_loss_step", subject_loss.item(), gs)

    N = max(agg["n"], 1)
    results = {
        "loss":           agg["loss"]      / N,
        "loss_perc":      agg["loss_perc"] / N,
        "loss_ch":        agg["loss_ch"]   / N,
        "perc_mae":       agg["perc_mae"]  / N,
        "perc_rmse":      (agg["perc_rmse_sq"] / N) ** 0.5,
        "ch_top1":        agg["ch_top1"]   / N,
        "ch_zero":        agg["ch_zero"]   / N,
        "subject_loss":   agg["subject_loss"] / N,
        "subject_acc":    agg["subject_acc"]  / N,
        "perc_pred_det":      agg["perc_pred_det"]     / max(agg["n_det"], 1),
        "perc_pred_nonspike": agg["perc_pred_nonspike"]/ max(agg["n_nonspike"], 1),
        "perc_tgt_det":       agg["perc_tgt_det"]      / max(agg["n_det"], 1),
        "perc_tgt_nonspike":  agg["perc_tgt_nonspike"] / max(agg["n_nonspike"], 1),
        "_all_perc_preds":    torch.cat(all_perc_preds) if all_perc_preds else None,
        "_all_ch_logits":     torch.cat(all_ch_logits)  if all_ch_logits  else None,
        "_all_embeddings":    torch.cat(all_embeddings) if all_embeddings  else None,
        "_mean_pred_per_channel": torch.nan_to_num(ch_pred_sum / N).detach().cpu(),
        "_mean_tgt_per_channel":  torch.nan_to_num(ch_tgt_sum / N).detach().cpu(),
        "_num_steps":         len(dataloader),
    }
    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    json_split_path: str,
    fold_index: int,
    data_dir: str,
    in_channels: int = 21,
    batch_size: int = 64,
    emb_dim: int = 64,
    hidden: int = 64,
    dropout: float = 0.3,
    encoder_type: str = "t_s_cnn",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    window_size: int = 128,
    max_offset: int = 16,
    max_segments_per_patient: Optional[int] = 3000,
    lambda_perception: float = 1.0,
    lambda_channel: float = 1.0,
    grad_clip_norm: float = 1.0,
    log_root: str = "./runs",
    num_workers: int = 0,
    use_memmap: bool = False,
    early_stopping_patience: int = 30,
    early_stopping_warmup: int = 10,
    early_stopping_enabled: bool = True,
    early_stopping_min_delta: float = 0.0,
    early_stopping_smoothing_window: int = 10,
    restore_best_checkpoint: bool = True,
    amp_enabled: bool = True,
    dry_run: bool = False,
    # Subject-adversarial options
    use_subject_adversary: bool = False,
    subject_adv_lambda_max: float = 0.02,
    subject_adv_warmup_epochs: int = 20,
    subject_adv_ramp_epochs: int = 80,
    subject_adv_hidden_dim: int = 64,
    subject_adv_dropout: float = 0.2,
    subject_adv_loss_weight: float = 1.0,
):
    n_channels = len(CHANNEL_LABEL_TO_INT)   # 21

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets ---
    train_dataset, val_dataset = build_flat_pretrain_datasets(
        flat_data_dir=data_dir,
        splits_json_path=json_split_path,
        fold=fold_index,
        window_size=window_size,
        max_offset=max_offset,
        max_segments_per_patient=max_segments_per_patient,
        use_memmap=use_memmap,
    )

    if dry_run:
        # Limit to a handful of samples for quick smoke-test
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(min(128, len(train_dataset)))))
        val_dataset   = Subset(val_dataset,   list(range(min(64,  len(val_dataset)))))
        print(">>> DRY RUN: limited to 128 / 64 train / val samples")

    # --- Subject-to-index mapping ---
    # Built from the full training set (including dry_run subset wrapping).
    subject_to_idx: Optional[dict] = None
    num_subjects: Optional[int] = None
    if use_subject_adversary:
        # Unwrap Subset if needed to access the underlying dataset's index_table
        base_train = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
        all_pids = extract_patient_ids(getattr(base_train, "index_table", None))
        subject_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}
        num_subjects   = len(subject_to_idx)
        print(f"Subject adversary enabled: {num_subjects} subjects found in training set.")
        print(f"  subject_to_idx (first 5): {dict(list(subject_to_idx.items())[:5])}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # TODO (subject-adversarial): for best domain-confusion signal, replace
        # shuffle=True with a subject-balanced sampler that draws N subjects per
        # batch and K segments per subject (e.g. using a custom BatchSampler that
        # iterates over subject_to_idx groups).  The current random sampling
        # works but may yield many same-subject batches.
        num_workers=num_workers,
        collate_fn=flat_spike_pretrain_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=flat_spike_pretrain_collate,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model ---
    model = SpikeEncoderPretrainModel(
        in_channels=in_channels,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        n_channels=n_channels,
        encoder_type=encoder_type,
        window_size=window_size,
        use_subject_adversary=use_subject_adversary,
        num_subjects=num_subjects,
        subject_adv_hidden_dim=subject_adv_hidden_dim,
        subject_adv_dropout=subject_adv_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Losses ---
    criterion_perc = nn.MSELoss()
    criterion_ch   = nn.BCEWithLogitsLoss(reduction="none")

    # --- Optimizer & scaler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_enabled))

    # --- Logging ---
    datestr  = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir  = os.path.join(
        log_root, f"spike_encoder_pretrain_fold{fold_index}_{datestr}"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    # --- Run fingerprint ---
    train_config = dict(
        json_split_path=json_split_path,
        fold_index=fold_index,
        data_dir=data_dir,
        in_channels=in_channels,
        batch_size=batch_size,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        encoder_type=encoder_type,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        window_size=window_size,
        max_offset=max_offset,
        max_segments_per_patient=max_segments_per_patient,
        lambda_perception=lambda_perception,
        lambda_channel=lambda_channel,
        use_memmap=use_memmap,
        use_subject_adversary=use_subject_adversary,
        subject_adv_lambda_max=subject_adv_lambda_max,
        subject_adv_warmup_epochs=subject_adv_warmup_epochs,
        subject_adv_ramp_epochs=subject_adv_ramp_epochs,
        subject_adv_hidden_dim=subject_adv_hidden_dim,
        subject_adv_dropout=subject_adv_dropout,
        subject_adv_loss_weight=subject_adv_loss_weight,
        num_subjects=num_subjects,
    )
    emit_run_fingerprint(
        script_name="eeg_spike_encoder_pretraining.py",
        train_config=train_config,
        model_kwargs=dict(
            in_channels=in_channels,
            emb_dim=emb_dim,
            hidden=hidden,
            dropout=dropout,
            encoder_type=encoder_type,
        ),
        effective_model_config=dict(
            total_params=total_params,
            n_channels=n_channels,
            lambda_perception=lambda_perception,
            lambda_channel=lambda_channel,
        ),
    )

    # Save run fingerprint JSON to disk for easy reproducibility inspection
    fingerprint_path = os.path.join(log_dir, "run_fingerprint.json")
    with open(fingerprint_path, "w") as f:
        json.dump(
            {
                "train_config": train_config,
                "subject_to_idx": subject_to_idx,
                "channel_label_to_int": CHANNEL_LABEL_TO_INT,
            },
            f,
            indent=2,
        )
    print(f"Run fingerprint saved: {fingerprint_path}")

    # --- Early stopping ---
    early_stop = EarlyStopping(
        patience=early_stopping_patience,
        warmup=early_stopping_warmup,
        smoothing_window=early_stopping_smoothing_window,
        min_delta=early_stopping_min_delta,
        enabled=early_stopping_enabled and not dry_run,
    )

    best_raw_val_loss      = float("inf")
    best_smoothed_val_loss = float("inf")
    global_step = 0

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(epochs):
        t0 = time.time()

        subject_adv_lambda = compute_subject_adv_lambda(
            epoch,
            warmup_epochs=subject_adv_warmup_epochs,
            ramp_epochs=subject_adv_ramp_epochs,
            lambda_max=subject_adv_lambda_max,
        ) if use_subject_adversary else 0.0

        train_m = _run_epoch(
            model, train_loader, optimizer,
            criterion_perc, criterion_ch,
            device, scaler, epoch,
            lambda_perception, lambda_channel,
            training=True, writer=writer, n_channels=n_channels,
            grad_clip_norm=grad_clip_norm,
            step_offset=global_step,
            amp_enabled=amp_enabled,
            use_subject_adversary=use_subject_adversary,
            subject_adv_lambda=subject_adv_lambda,
            subject_to_idx=subject_to_idx,
            subject_adv_loss_weight=subject_adv_loss_weight,
        )

        global_step += train_m["_num_steps"]

        val_m = _run_epoch(
            model, val_loader, None,
            criterion_perc, criterion_ch,
            device, scaler, epoch,
            lambda_perception, lambda_channel,
            training=False, writer=None, n_channels=n_channels,
            amp_enabled=amp_enabled,
            use_subject_adversary=use_subject_adversary,
            subject_adv_lambda=subject_adv_lambda,
            subject_to_idx=subject_to_idx,
            subject_adv_loss_weight=subject_adv_loss_weight,
        )

        epoch_time = time.time() - t0

        # Early stopping update
        es_state = early_stop.update(epoch, val_m["loss"])
        smoothed_val_loss = es_state["smoothed_val_loss"]
        best_raw_val_loss      = es_state["best_raw_val_loss"]
        best_smoothed_val_loss = es_state["best_smoothed_val_loss"]

        # --- Console ---
        subj_info = (
            f"  SubjLoss: {train_m['subject_loss']:.4f}, SubjAcc: {train_m['subject_acc']:.3f} | "
            f"Val SubjLoss: {val_m['subject_loss']:.4f}, Val SubjAcc: {val_m['subject_acc']:.3f} | "
            f"lambda={subject_adv_lambda:.5f}\n"
        ) if use_subject_adversary else ""
        print(
            f"\nEpoch {epoch + 1}/{epochs}  ({epoch_time:.1f}s)\n"
            f"  Train - Loss: {train_m['loss']:.4f}, "
            f"Perception: {train_m['loss_perc']:.4f}, "
            f"Channel: {train_m['loss_ch']:.4f}, "
            f"PercMAE: {train_m['perc_mae']:.4f}, "
            f"ChAcc: {train_m['ch_top1']:.3f}, "
            f"ChZero: {train_m['ch_zero']:.3f}\n"
            f"  Val   - Loss: {val_m['loss']:.4f}, "
            f"Perception: {val_m['loss_perc']:.4f}, "
            f"Channel: {val_m['loss_ch']:.4f}, "
            f"PercMAE: {val_m['perc_mae']:.4f}, "
            f"ChAcc: {val_m['ch_top1']:.3f}, "
            f"ChZero: {val_m['ch_zero']:.3f}\n"
            f"{subj_info}"
            f"  Smoothed val loss: {smoothed_val_loss:.4f}  "
            f"(best raw: {best_raw_val_loss:.4f}, "
            f"best smooth: {best_smoothed_val_loss:.4f})"
        )

        # --- TensorBoard epoch scalars ---
        # Paired train/val metrics land on the same card via add_scalars.
        writer.add_scalars("loss/total",              {"Train": train_m["loss"],      "Val": val_m["loss"]},      global_step)
        writer.add_scalars("loss/perception",         {"Train": train_m["loss_perc"], "Val": val_m["loss_perc"]}, global_step)
        writer.add_scalars("loss/channel",            {"Train": train_m["loss_ch"],   "Val": val_m["loss_ch"]},   global_step)
        writer.add_scalars("metrics/perception_mae",  {"Train": train_m["perc_mae"],  "Val": val_m["perc_mae"]},  global_step)
        writer.add_scalars("metrics/perception_rmse", {"Train": train_m["perc_rmse"], "Val": val_m["perc_rmse"]}, global_step)
        writer.add_scalars("metrics/channel_top1_acc",{"Train": train_m["ch_top1"],   "Val": val_m["ch_top1"]},   global_step)
        writer.add_scalars("metrics/channel_zero_acc",{"Train": train_m["ch_zero"],   "Val": val_m["ch_zero"]},   global_step)
        writer.add_scalars("perception/pred_detection_mean",   {"Train": train_m["perc_pred_det"],      "Val": val_m["perc_pred_det"]},      global_step)
        writer.add_scalars("perception/pred_non_spike_mean",   {"Train": train_m["perc_pred_nonspike"], "Val": val_m["perc_pred_nonspike"]}, global_step)
        writer.add_scalars("perception/target_detection_mean", {"Train": train_m["perc_tgt_det"],       "Val": val_m["perc_tgt_det"]},       global_step)
        writer.add_scalars("perception/target_non_spike_mean", {"Train": train_m["perc_tgt_nonspike"],  "Val": val_m["perc_tgt_nonspike"]},  global_step)

        if use_subject_adversary:
            writer.add_scalar("train/subject_adv_lambda", subject_adv_lambda,         global_step)
            writer.add_scalar("train/subject_loss",       train_m["subject_loss"],     global_step)
            writer.add_scalar("train/subject_acc",        train_m["subject_acc"],      global_step)
            writer.add_scalar("val/subject_loss",         val_m["subject_loss"],       global_step)
            writer.add_scalar("val/subject_acc",          val_m["subject_acc"],        global_step)

        # Val-only tracking scalars (raw + smoothed, early stopping)
        writer.add_scalar("val/loss_raw",      val_m["loss"],      global_step)
        writer.add_scalar("val/loss_smoothed", smoothed_val_loss,   global_step)
        writer.add_scalar("early_stopping/best_smoothed_val_loss",        es_state["best_smoothed_val_loss"],    global_step)
        writer.add_scalar("early_stopping/epochs_without_improvement",    early_stop.epochs_without_improvement, global_step)
        if early_stop.best_epoch is not None:
            writer.add_scalar("early_stopping/best_epoch", early_stop.best_epoch + 1, global_step)

        # --- Histograms every 2 epochs ---
        if (epoch + 1) % 2 == 0:
            # Channel-wise mean score/target histograms (21 bins = channel indices).
            writer.add_histogram("channel/train_mean_pred_per_channel_hist", train_m["_mean_pred_per_channel"], global_step)
            writer.add_histogram("channel/train_mean_tgt_per_channel_hist",  train_m["_mean_tgt_per_channel"],  global_step)
            writer.add_histogram("channel/val_mean_pred_per_channel_hist",   val_m["_mean_pred_per_channel"],   global_step)
            writer.add_histogram("channel/val_mean_tgt_per_channel_hist",    val_m["_mean_tgt_per_channel"],    global_step)
            # Overall prediction histograms (unaggregated per-sample predictions/logits).
            if train_m["_all_perc_preds"] is not None:
                writer.add_histogram("perception/train_pred_hist",   train_m["_all_perc_preds"], global_step)
            if val_m["_all_perc_preds"] is not None:
                writer.add_histogram("perception/val_pred_hist",     val_m["_all_perc_preds"],   global_step)
            if train_m["_all_ch_logits"] is not None:
                writer.add_histogram("channel/train_logits_hist",    train_m["_all_ch_logits"],  global_step)
            if val_m["_all_ch_logits"] is not None:
                writer.add_histogram("channel/val_logits_hist",      val_m["_all_ch_logits"],    global_step)
            if (
                train_m["_all_embeddings"] is not None
                and train_m["_all_embeddings"].numel() < 10_000_000
            ):
                writer.add_histogram("embedding/train_embedding_hist", train_m["_all_embeddings"], global_step)

        # --- Save best checkpoint ---
        if es_state["raw_improved"]:
            ckpt_path = os.path.join(log_dir, "checkpoint_best.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "val_loss":             val_m["loss"],
                    "smoothed_val_loss":    smoothed_val_loss,
                    "best_raw_val_loss":    best_raw_val_loss,
                    "best_smoothed_val_loss": best_smoothed_val_loss,
                    "args": train_config,
                    "channel_label_to_int": CHANNEL_LABEL_TO_INT,
                    # Subject-adversary metadata
                    "use_subject_adversary":    use_subject_adversary,
                    "num_subjects":             num_subjects,
                    "subject_to_idx":           subject_to_idx,
                    "subject_adv_lambda_max":   subject_adv_lambda_max,
                    "subject_adv_warmup_epochs": subject_adv_warmup_epochs,
                    "subject_adv_ramp_epochs":  subject_adv_ramp_epochs,
                    "subject_adv_hidden_dim":   subject_adv_hidden_dim,
                    "subject_adv_dropout":      subject_adv_dropout,
                    "subject_adv_loss_weight":  subject_adv_loss_weight,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint: {ckpt_path}")

            # Encoder-only checkpoint (no subject head — backward compatible)
            enc_path = os.path.join(log_dir, "encoder_best.pt")
            torch.save(
                {
                    "encoder_state_dict":  model.encoder.state_dict(),
                    "emb_dim":             model.emb_dim,
                    "encoder_type":        model.encoder_type,
                    "in_channels":         in_channels,
                    "window_size":         window_size,
                    "channel_label_to_int": CHANNEL_LABEL_TO_INT,
                },
                enc_path,
            )
            print(f"  ✓ Saved encoder-only checkpoint: {enc_path}")

        if es_state["should_stop"]:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1} "
                f"({early_stop.epochs_without_improvement} epochs without improvement)."
            )
            break

        if dry_run:
            print(">>> DRY RUN: stopping after first epoch.")
            break

    # Final checkpoint (last epoch state)
    last_ckpt_path = os.path.join(log_dir, "checkpoint_last.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "args": train_config,
            "channel_label_to_int": CHANNEL_LABEL_TO_INT,
            "use_subject_adversary":    use_subject_adversary,
            "num_subjects":             num_subjects,
            "subject_to_idx":           subject_to_idx,
        },
        last_ckpt_path,
    )
    print(f"\nSaved last checkpoint: {last_ckpt_path}")

    # Restore best weights before printing final summary
    best_ckpt_path = os.path.join(log_dir, "checkpoint_best.pth")
    if restore_best_checkpoint and os.path.isfile(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Restored best checkpoint from epoch {best_ckpt['epoch']} (val loss {best_ckpt['val_loss']:.4f})")

    writer.close()
    print(
        f"\n{'=' * 60}\n"
        f"Pretraining complete!\n"
        f"Best raw val loss:      {best_raw_val_loss:.4f}\n"
        f"Best smoothed val loss: {best_smoothed_val_loss:.4f}\n"
        f"{'=' * 60}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain the EEG spike encoder on flat 1-70 Hz segments."
    )

    # Required paths
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to JSON with k-fold subject splits.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_flat_1-70Hz_segments.npy files.")
    parser.add_argument("--log_root", type=str, default="./runs",
                        help="Base directory for TensorBoard logs and checkpoints.")

    # Fold
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index from the JSON splits file.")

    # Model
    parser.add_argument("--in_channels", type=int, default=21,
                        help="Number of EEG channels.")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="Encoder output embedding dimension.")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden units in the prediction heads.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability.")
    parser.add_argument("--encoder_type", type=str, default="t_s_cnn",
                        choices=["eegnet", "cnn1d", "t_s_cnn"],
                        help="Encoder architecture.")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)

    # Data / augmentation
    parser.add_argument("--window_size", type=int, default=128,
                        help="Temporal crop size (samples).")
    parser.add_argument("--max_offset", type=int, default=0,
                        help="Max jitter around center crop during training.")
    parser.add_argument("--max_segments_per_patient", type=int, default=3000,
                        help="Cap on segments per patient after harmonisation (default: 3000).")
    parser.add_argument(
        "--use_memmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load .npy files with mmap_mode='r' to reduce RAM usage.",
    )

    # Loss weights
    parser.add_argument("--lambda_perception", type=float, default=1.0,
                        help="Weight on the perception MSE loss term.")
    parser.add_argument("--lambda_channel", type=float, default=1.0,
                        help="Weight on the channel BCE loss term.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled).")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--early_stopping_warmup", type=int, default=10)
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable early stopping (default: enabled).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement in val loss to reset early stopping counter.",
    )
    parser.add_argument(
        "--early_stopping_smoothing_window",
        type=int,
        default=10,
        help="Window size for smoothing val loss in early stopping.",
    )
    parser.add_argument(
        "--restore_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore best checkpoint weights after training (default: enabled).",
    )

    # Subject-adversarial learning
    parser.add_argument(
        "--use_subject_adversary",
        action="store_true",
        help="Enable subject-adversarial training with gradient reversal.",
    )
    parser.add_argument("--subject_adv_lambda_max",    type=float, default=0.02)
    parser.add_argument("--subject_adv_warmup_epochs", type=int,   default=0)
    parser.add_argument("--subject_adv_ramp_epochs",   type=int,   default=10)
    parser.add_argument("--subject_adv_hidden_dim",    type=int,   default=64)
    parser.add_argument("--subject_adv_dropout",       type=float, default=0.2)
    parser.add_argument("--subject_adv_loss_weight",   type=float, default=1.0)

    # Misc
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a single forward/backward pass then exit (for smoke-testing).",
    )

    args = parser.parse_args()

    print(">>> eeg_spike_encoder_pretraining.py  —  run configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    train(
        json_split_path=args.splits_json,
        fold_index=args.fold,
        data_dir=args.data_dir,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        window_size=args.window_size,
        max_offset=args.max_offset,
        max_segments_per_patient=args.max_segments_per_patient,
        lambda_perception=args.lambda_perception,
        lambda_channel=args.lambda_channel,
        grad_clip_norm=args.grad_clip_norm,
        log_root=args.log_root,
        num_workers=args.num_workers,
        use_memmap=args.use_memmap,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_warmup=args.early_stopping_warmup,
        early_stopping_enabled=args.early_stopping,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_smoothing_window=args.early_stopping_smoothing_window,
        restore_best_checkpoint=args.restore_best_checkpoint,
        amp_enabled=not args.no_amp,
        dry_run=args.dry_run,
        use_subject_adversary=args.use_subject_adversary,
        subject_adv_lambda_max=args.subject_adv_lambda_max,
        subject_adv_warmup_epochs=args.subject_adv_warmup_epochs,
        subject_adv_ramp_epochs=args.subject_adv_ramp_epochs,
        subject_adv_hidden_dim=args.subject_adv_hidden_dim,
        subject_adv_dropout=args.subject_adv_dropout,
        subject_adv_loss_weight=args.subject_adv_loss_weight,
    )
