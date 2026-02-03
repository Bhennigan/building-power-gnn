"""Training pipeline for building power GNN models.

Optimized for NVIDIA DGX Spark with Grace Hopper architecture.
"""

from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model hyperparameters
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    temporal_dim: int = 64
    dropout: float = 0.1

    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    warmup_epochs: int = 5

    # Optimization
    use_amp: bool = True  # Automatic mixed precision
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Scheduler
    scheduler_type: str = "cosine"  # 'cosine' or 'onecycle'
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DGX Spark specific
    use_cudnn_benchmark: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricTracker:
    """Tracks and computes training metrics."""

    def __init__(self):
        """Initialize metric tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self._values = {}
        self._counts = {}

    def update(self, metrics: dict[str, float], n: int = 1):
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric names to values.
            n: Number of samples.
        """
        for name, value in metrics.items():
            if name not in self._values:
                self._values[name] = 0.0
                self._counts[name] = 0
            self._values[name] += value * n
            self._counts[name] += n

    def compute(self) -> dict[str, float]:
        """Compute averaged metrics."""
        return {
            name: self._values[name] / self._counts[name]
            for name in self._values
        }


class Trainer:
    """Training loop manager for building power GNN."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            loss_fn: Loss function (defaults to MSELoss).
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.MSELoss()

        # Setup optimization
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.min_lr,
            )
        else:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.num_epochs,
                steps_per_epoch=len(train_loader),
            )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            config.patience,
            config.min_delta,
        )

        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # DGX Spark optimizations
        if config.use_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.config.device)

            # Forward pass with optional AMP
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics
            self.train_metrics.update(
                {"loss": loss.item() * self.config.accumulation_steps},
                n=batch.num_nodes if hasattr(batch, "num_nodes") else 1,
            )

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        predictions = []
        targets = []

        for batch in self.val_loader:
            batch = batch.to(self.config.device)

            if self.config.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)

            self.val_metrics.update(
                {"loss": loss.item()},
                n=batch.num_nodes if hasattr(batch, "num_nodes") else 1,
            )

            # Collect predictions for additional metrics
            for node_type, out in outputs.items():
                if "power_pred" in out:
                    predictions.append(out["power_pred"].cpu())
                    if hasattr(batch[node_type], "y"):
                        targets.append(batch[node_type].y.cpu())

        metrics = self.val_metrics.compute()

        # Compute additional metrics
        if predictions and targets:
            preds = torch.cat(predictions)
            targs = torch.cat(targets)
            metrics["mape"] = self._compute_mape(preds, targs)
            metrics["rmse"] = torch.sqrt(nn.MSELoss()(preds, targs)).item()

        return metrics

    def _compute_loss(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        batch: HeteroData,
    ) -> torch.Tensor:
        """Compute total loss."""
        total_loss = 0.0
        count = 0

        for node_type, out in outputs.items():
            if "power_pred" in out and hasattr(batch[node_type], "y"):
                pred = out["power_pred"].squeeze()
                target = batch[node_type].y.squeeze()
                total_loss += self.loss_fn(pred, target)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.config.device, requires_grad=True)

        return total_loss / count

    @staticmethod
    def _compute_mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
        """Compute Mean Absolute Percentage Error."""
        mask = target.abs() > eps
        if mask.sum() == 0:
            return 0.0
        return (((pred[mask] - target[mask]).abs() / target[mask].abs()).mean() * 100).item()

    def train(self, callbacks: Optional[list[Callable]] = None) -> dict:
        """Run full training loop.

        Args:
            callbacks: Optional list of callback functions called after each epoch.

        Returns:
            Training history.
        """
        history = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"AMP enabled: {self.config.use_amp}")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            if "loss" in val_metrics:
                history["val_loss"].append(val_metrics["loss"])

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)
            if self.config.scheduler_type == "cosine":
                self.scheduler.step()

            # Logging
            epoch_time = time.time() - epoch_start
            log_msg = (
                f"Epoch {epoch + 1}/{self.config.num_epochs} "
                f"[{epoch_time:.1f}s] - "
                f"train_loss: {train_metrics['loss']:.4f}"
            )
            if val_metrics:
                log_msg += f" - val_loss: {val_metrics['loss']:.4f}"
                if "mape" in val_metrics:
                    log_msg += f" - MAPE: {val_metrics['mape']:.2f}%"
            log_msg += f" - lr: {current_lr:.2e}"
            logger.info(log_msg)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Best model
            if val_metrics and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")

            # Early stopping
            if val_metrics and self.early_stopping.step(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


def create_data_loaders(
    train_data: list[HeteroData],
    val_data: Optional[list[HeteroData]] = None,
    config: Optional[TrainingConfig] = None,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training.

    Args:
        train_data: List of training graphs.
        val_data: Optional list of validation graphs.
        config: Training configuration.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    config = config or TrainingConfig()

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = None
    if val_data:
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return train_loader, val_loader
