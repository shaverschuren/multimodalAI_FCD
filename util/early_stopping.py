"""
early_stopping.py

Shared EarlyStopping utility for training scripts.

Tracks raw and smoothed validation loss, manages patience counters,
and signals when training should stop.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping with smoothed validation loss and warmup period.

    Parameters
    ----------
    patience : int
        Number of validation epochs without improvement before stopping.
    min_delta : float
        Minimum absolute decrease in smoothed val loss to count as improvement.
    warmup : int
        Number of epochs (0-based) before early stopping may trigger.
        Stopping is only considered when ``epoch >= warmup``.
    smoothing_window : int
        Rolling window size for computing the smoothed validation loss.
    enabled : bool
        When False the class still tracks state but ``should_stop`` is always False.
    """

    def __init__(
        self,
        patience: int = 200,
        min_delta: float = 0.0,
        warmup: int = 150,
        smoothing_window: int = 10,
        enabled: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.smoothing_window = smoothing_window
        self.enabled = enabled

        self.val_loss_history: list[float] = []
        self.best_raw_val_loss: float = float("inf")
        self.best_raw_epoch: int | None = None
        self.best_smoothed_val_loss: float = float("inf")
        self.best_epoch: int | None = None           # epoch (0-based) of best smoothed loss
        self.epochs_without_improvement: int = 0

    # ------------------------------------------------------------------
    def update(self, epoch: int, val_loss: float) -> dict:
        """
        Process a new validation loss value.

        Parameters
        ----------
        epoch : int
            Current epoch (0-based, as used inside the training loop).
        val_loss : float
            Raw validation loss for this epoch.

        Returns
        -------
        dict with keys:
            smoothed_val_loss     : float
            improved              : bool   – smoothed loss improved this epoch
            raw_improved          : bool   – raw loss improved this epoch
            should_stop           : bool   – trigger early stopping
            best_epoch            : int | None   (0-based)
            best_smoothed_val_loss: float
            best_raw_val_loss     : float
            best_raw_epoch        : int | None   (0-based)
            epochs_without_improvement : int
        """
        # --- raw loss tracking ----------------------------------------
        self.val_loss_history.append(val_loss)
        raw_improved = val_loss < self.best_raw_val_loss
        if raw_improved:
            self.best_raw_val_loss = val_loss
            self.best_raw_epoch = epoch

        # --- smoothed loss --------------------------------------------
        recent = self.val_loss_history[-self.smoothing_window:]
        smoothed_val_loss = float(np.mean(recent))

        # --- improvement check ----------------------------------------
        improved = smoothed_val_loss < self.best_smoothed_val_loss - self.min_delta
        if improved:
            self.best_smoothed_val_loss = smoothed_val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # --- stopping decision ----------------------------------------
        should_stop = (
            self.enabled
            and epoch >= self.warmup
            and self.epochs_without_improvement >= self.patience
        )

        return {
            "smoothed_val_loss": smoothed_val_loss,
            "improved": improved,
            "raw_improved": raw_improved,
            "should_stop": should_stop,
            "best_epoch": self.best_epoch,
            "best_smoothed_val_loss": self.best_smoothed_val_loss,
            "best_raw_val_loss": self.best_raw_val_loss,
            "best_raw_epoch": self.best_raw_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
