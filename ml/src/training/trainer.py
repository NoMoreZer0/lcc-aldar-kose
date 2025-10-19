"""
Main training loop for diffusion model finetuning.
"""
import logging
import os
import math
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

from .utils import (
    EMAModel,
    CheckpointManager,
    MetricTracker,
    get_lr,
    count_parameters,
)

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """
    Trainer for finetuning diffusion models.

    Supports:
    - Full finetuning
    - LoRA/QLoRA
    - Mixed precision training
    - Gradient accumulation
    - EMA
    - Distributed training
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[UNet2DConditionModel] = None,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        noise_scheduler: Optional[DDPMScheduler] = None,
    ):
        """
        Args:
            config: Training configuration dict
            model: UNet model (if None, will be loaded from config)
            vae: VAE model
            text_encoder: Text encoder
            tokenizer: Tokenizer
            noise_scheduler: Noise scheduler for training
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models if not provided
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler

        if self.model is None:
            self._load_models()

        # Move models to device
        self.model.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        if not config.get("text_encoder", {}).get("train_text_encoder", False):
            self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing if specified
        if config["model"].get("gradient_checkpointing", False):
            self.model.enable_gradient_checkpointing()
            if config.get("text_encoder", {}).get("train_text_encoder", False):
                self.text_encoder.gradient_checkpointing_enable()

        # Enable xformers if specified
        if config["model"].get("enable_xformers", False):
            try:
                self.model.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")

        # Print parameter counts
        logger.info(f"Total parameters: {count_parameters(self.model, only_trainable=False):,}")
        logger.info(f"Trainable parameters: {count_parameters(self.model, only_trainable=True):,}")

        # Initialize training components
        self.optimizer = None
        self.lr_scheduler = None
        self.ema_model = None
        self.checkpoint_manager = None
        self.scaler = None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

    def _load_models(self):
        """Load pretrained models from config."""
        from diffusers import StableDiffusionXLPipeline

        model_id = self.config["model"]["base_model_id"]
        logger.info(f"Loading models from {model_id}")

        # Load full pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config["training"]["mixed_precision"] == "fp16" else torch.float32,
            variant="fp16" if self.config["training"]["mixed_precision"] == "fp16" else None,
        )

        self.model = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer

        # Create noise scheduler for training
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )

    def setup_training(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """
        Setup optimizer, scheduler, and other training components.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        config = self.config["training"]

        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / config["gradient_accumulation_steps"]
        )

        if config.get("max_train_steps") is None:
            max_train_steps = config["num_epochs"] * num_update_steps_per_epoch
        else:
            max_train_steps = config["max_train_steps"]

        self.max_train_steps = max_train_steps

        # Setup optimizer
        if config.get("use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.AdamW

        # Get trainable parameters
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        # Add text encoder parameters if training
        if config.get("text_encoder", {}).get("train_text_encoder", False):
            text_encoder_lr = config.get("text_encoder", {}).get("learning_rate", config["learning_rate"])
            params_to_optimize = [
                {"params": params_to_optimize, "lr": config["learning_rate"]},
                {"params": self.text_encoder.parameters(), "lr": text_encoder_lr},
            ]
        else:
            params_to_optimize = [{"params": params_to_optimize}]

        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=config["learning_rate"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            weight_decay=config["adam_weight_decay"],
            eps=config["adam_epsilon"],
        )

        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            config["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=config["lr_warmup_steps"] * config["gradient_accumulation_steps"],
            num_training_steps=max_train_steps * config["gradient_accumulation_steps"],
        )

        # Setup EMA
        if self.config["ema"].get("use_ema", False):
            self.ema_model = EMAModel(
                self.model,
                decay=self.config["ema"]["decay"],
                update_after_step=self.config["ema"]["start_step"],
            )
            logger.info(f"Using EMA with decay={self.config['ema']['decay']}")

        # Setup gradient scaler for mixed precision
        if config["mixed_precision"] != "no":
            self.scaler = GradScaler()

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=config["output_dir"],
            save_steps=self.config["checkpointing"]["save_steps"],
            keep_last_n=self.config["checkpointing"]["keep_last_n_checkpoints"],
            save_best=self.config["checkpointing"]["save_best"],
            best_metric=self.config["checkpointing"]["best_metric"],
            higher_is_better=self.config["checkpointing"]["higher_is_better"],
        )

        logger.info(f"Total training steps: {max_train_steps}")
        logger.info(f"Training for {config['num_epochs']} epochs")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dict of loss values
        """
        self.model.train()

        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch.get("input_ids")

        # Encode images to latents
        if "latents" in batch:
            # Use cached latents
            latents = batch["latents"].to(self.device)
        else:
            # Encode on the fly
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise offset if configured
        noise_offset = self.config["training"].get("noise_offset", 0.0)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device
            )

        batch_size = latents.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
        ).long()

        # Add noise to latents according to noise scheduler
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        if input_ids is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(input_ids.to(self.device))[0]
        else:
            # Use unconditional embedding
            encoder_hidden_states = None

        # Predict noise
        with autocast(enabled=self.config["training"]["mixed_precision"] != "no"):
            model_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]

            # Get target
            if self.config["training"].get("prediction_type", "epsilon") == "epsilon":
                target = noise
            elif self.config["training"]["prediction_type"] == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.config['training']['prediction_type']}")

            # Calculate loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Apply SNR weighting if configured
            snr_gamma = self.config["training"].get("snr_gamma")
            if snr_gamma is not None:
                snr = self._compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                loss = loss * mse_loss_weights

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item(),
        }

    def _compute_snr(self, timesteps):
        """
        Compute signal-to-noise ratio for timesteps.
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        snr = (alpha / sigma) ** 2
        return snr

    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_dataloader: Training data loader

        Returns:
            Average metrics for the epoch
        """
        self.model.train()
        self.train_metrics.reset()

        config = self.config["training"]
        gradient_accumulation_steps = config["gradient_accumulation_steps"]

        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {self.epoch}",
            disable=False,
        )

        for step, batch in enumerate(train_dataloader):
            # Training step
            metrics = self.train_step(batch)
            self.train_metrics.update(metrics)

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                if config.get("max_grad_norm") is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config["max_grad_norm"],
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema_model is not None:
                    self.ema_model.update(self.model)

                self.global_step += 1

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{get_lr(self.optimizer):.2e}",
                })

                # Save checkpoint
                if self.checkpoint_manager.should_save(self.global_step):
                    self._save_checkpoint()

                # Check if training is complete
                if self.global_step >= self.max_train_steps:
                    break

        progress_bar.close()
        return self.train_metrics.get_averages()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        metrics = {
            **self.train_metrics.get_averages(),
            **self.val_metrics.get_averages(),
        }

        self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ema_model=self.ema_model,
            metrics=metrics,
            extra_state={
                "epoch": self.epoch,
            },
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        # Setup training
        self.setup_training(train_dataloader, val_dataloader)

        # Resume from checkpoint if specified
        resume_path = self.config["checkpointing"].get("resume_from_checkpoint")
        if resume_path:
            state = self.checkpoint_manager.load_checkpoint(
                resume_path,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.ema_model,
            )
            self.global_step = state.get("step", 0)
            self.epoch = state.get("epoch", 0)

        logger.info("Starting training...")

        # Training loop
        num_epochs = self.config["training"]["num_epochs"]
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            logger.info(f"Epoch {epoch} train metrics: {train_metrics}")

            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch} val metrics: {val_metrics}")

            # Check if max steps reached
            if self.global_step >= self.max_train_steps:
                logger.info("Max training steps reached")
                break

        logger.info("Training complete!")

        # Save final checkpoint
        self._save_checkpoint()

    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            val_dataloader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        for batch in tqdm(val_dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch.get("input_ids")

            # Encode to latents
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device,
            ).long()

            # Add noise
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            if input_ids is not None:
                encoder_hidden_states = self.text_encoder(input_ids.to(self.device))[0]
            else:
                encoder_hidden_states = None

            # Predict noise
            model_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]

            # Calculate loss
            if self.config["training"].get("prediction_type", "epsilon") == "epsilon":
                target = noise
            else:
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            self.val_metrics.update({"val_loss": loss.item()})

        return self.val_metrics.get_averages()
