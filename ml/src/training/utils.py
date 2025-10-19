"""
Training utilities for diffusion model finetuning.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a shadow copy of model parameters that is updated with EMA.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate
            min_decay: Minimum decay rate
            update_after_step: Start EMA after this many steps
            use_ema_warmup: Use EMA warmup schedule
            inv_gamma: Inverse gamma for warmup
            power: Power for warmup schedule
        """
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power

        # Store shadow parameters
        self.shadow_params = OrderedDict()
        self.collected_params = OrderedDict()

        # Copy model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()

        self.step = 0

    def get_decay(self, step: int) -> float:
        """Get decay rate for current step."""
        if step < self.update_after_step:
            return 0.0

        if self.use_ema_warmup:
            step = step - self.update_after_step
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
            return max(self.min_decay, min(value, self.decay))
        else:
            return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        self.step += 1
        decay = self.get_decay(self.step)

        # Update shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(decay).add_(
                    param.data, alpha=1 - decay
                )

    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def store(self, model: nn.Module):
        """Store current model parameters before loading EMA."""
        self.collected_params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.collected_params[name] = param.data.clone()

    def restore(self, model: nn.Module):
        """Restore original parameters after EMA evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.collected_params:
                param.data.copy_(self.collected_params[name])

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "step": self.step,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.decay = state_dict["decay"]
        self.min_decay = state_dict["min_decay"]
        self.step = state_dict["step"]
        self.shadow_params = state_dict["shadow_params"]


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles:
    - Periodic checkpoint saving
    - Best checkpoint tracking
    - Checkpoint cleanup (keep last N)
    - Resume from checkpoint
    """

    def __init__(
        self,
        output_dir: str,
        save_steps: int = 500,
        keep_last_n: int = 5,
        save_best: bool = True,
        best_metric: str = "val_loss",
        higher_is_better: bool = False,
    ):
        """
        Args:
            output_dir: Directory to save checkpoints
            save_steps: Save checkpoint every N steps
            keep_last_n: Keep only last N checkpoints
            save_best: Track and save best checkpoint
            best_metric: Metric name for best checkpoint
            higher_is_better: Whether higher metric is better
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_steps = save_steps
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better

        self.best_metric_value = float("-inf") if higher_is_better else float("inf")
        self.checkpoints = []

    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        return step % self.save_steps == 0

    def is_better(self, metric_value: float) -> bool:
        """Check if metric is better than current best."""
        if self.higher_is_better:
            return metric_value > self.best_metric_value
        else:
            return metric_value < self.best_metric_value

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler = None,
        ema_model: Optional[EMAModel] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Save a checkpoint.

        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer state
            lr_scheduler: Learning rate scheduler state
            ema_model: EMA model if used
            metrics: Current metrics
            extra_state: Any additional state to save
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Save model
        if hasattr(model, "save_pretrained"):
            # HuggingFace model
            model.save_pretrained(checkpoint_dir / "model")
        else:
            # PyTorch model
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save scheduler
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # Save EMA
        if ema_model is not None:
            torch.save(ema_model.state_dict(), checkpoint_dir / "ema.pt")

        # Save training state
        state = {
            "step": step,
            "metrics": metrics or {},
        }
        if extra_state:
            state.update(extra_state)

        torch.save(state, checkpoint_dir / "training_state.pt")

        # Track checkpoint
        self.checkpoints.append((step, checkpoint_dir))

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        # Check if best checkpoint
        if self.save_best and metrics and self.best_metric in metrics:
            metric_value = metrics[self.best_metric]
            if self.is_better(metric_value):
                logger.info(
                    f"New best {self.best_metric}: {metric_value:.4f} "
                    f"(previous: {self.best_metric_value:.4f})"
                )
                self.best_metric_value = metric_value
                self._save_best_checkpoint(checkpoint_dir)

        return checkpoint_dir

    def _save_best_checkpoint(self, checkpoint_dir: Path):
        """Copy checkpoint to best checkpoint directory."""
        best_dir = self.output_dir / "checkpoint-best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Saved best checkpoint to {best_dir}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.checkpoints) <= self.keep_last_n:
            return

        # Sort by step
        self.checkpoints.sort(key=lambda x: x[0])

        # Remove oldest
        while len(self.checkpoints) > self.keep_last_n:
            _, old_dir = self.checkpoints.pop(0)
            if old_dir.exists() and old_dir.name != "checkpoint-best":
                shutil.rmtree(old_dir)
                logger.info(f"Removed old checkpoint: {old_dir}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler = None,
        ema_model: Optional[EMAModel] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into
            optimizer: Optimizer to restore state
            lr_scheduler: Scheduler to restore state
            ema_model: EMA model to restore

        Returns:
            Training state dict
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Load model
        if hasattr(model, "from_pretrained"):
            # HuggingFace model
            model_dir = checkpoint_dir / "model"
            if model_dir.exists():
                model_state = model.__class__.from_pretrained(model_dir)
                model.load_state_dict(model_state.state_dict())
        else:
            # PyTorch model
            model_path = checkpoint_dir / "model.pt"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))

        # Load optimizer
        if optimizer is not None:
            opt_path = checkpoint_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path))

        # Load scheduler
        if lr_scheduler is not None:
            sched_path = checkpoint_dir / "scheduler.pt"
            if sched_path.exists():
                lr_scheduler.load_state_dict(torch.load(sched_path))

        # Load EMA
        if ema_model is not None:
            ema_path = checkpoint_dir / "ema.pt"
            if ema_path.exists():
                ema_model.load_state_dict(torch.load(ema_path))

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
        else:
            state = {}

        logger.info(f"Resumed from step {state.get('step', 0)}")
        return state


class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    """Tracks multiple metrics during training."""

    def __init__(self):
        self.metrics = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = AverageMeter(name)
            self.metrics[name].update(value, n)

    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        """Get average values of all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __str__(self):
        return " | ".join(str(meter) for meter in self.metrics.values())


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count number of parameters in model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module):
    """Freeze all parameters in model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all parameters in model."""
    for param in model.parameters():
        param.requires_grad = True
