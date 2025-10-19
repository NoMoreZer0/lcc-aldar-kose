"""
CLI for training diffusion models.
"""
import argparse
import logging
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.src.training.dataset import StoryboardDataset, collate_fn
from ml.src.training.trainer import DiffusionTrainer
from ml.src.training.utils import count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_dataloaders(config: dict, tokenizer):
    """Setup training and validation data loaders."""
    dataset_config = config["dataset"]
    training_config = config["training"]

    # Training dataset
    train_dataset = StoryboardDataset(
        data_dir=dataset_config["train_data_dir"],
        metadata_file=dataset_config["metadata_file"],
        image_column=dataset_config["image_column"],
        caption_column=dataset_config["caption_column"],
        conditioning_column=dataset_config.get("conditioning_column"),
        resolution=config["model"]["resolution"],
        center_crop=dataset_config["preprocessing"].get("center_crop", True),
        random_flip=dataset_config["preprocessing"].get("random_flip", 0.5),
        tokenizer=tokenizer,
        max_caption_length=dataset_config.get("max_caption_length", 77),
        cache_latents=dataset_config.get("cache_latents", False),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_size"],
        shuffle=True,
        num_workers=dataset_config.get("num_workers", 4),
        pin_memory=dataset_config.get("pin_memory", True),
        collate_fn=collate_fn,
    )

    # Validation dataset (optional)
    val_dataloader = None
    if dataset_config.get("val_data_dir"):
        val_dataset = StoryboardDataset(
            data_dir=dataset_config["val_data_dir"],
            metadata_file=dataset_config["metadata_file"],
            image_column=dataset_config["image_column"],
            caption_column=dataset_config["caption_column"],
            conditioning_column=dataset_config.get("conditioning_column"),
            resolution=config["model"]["resolution"],
            center_crop=True,
            random_flip=0.0,
            tokenizer=tokenizer,
            max_caption_length=dataset_config.get("max_caption_length", 77),
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config.get("val_batch_size", 1),
            shuffle=False,
            num_workers=dataset_config.get("num_workers", 4),
            pin_memory=dataset_config.get("pin_memory", True),
            collate_fn=collate_fn,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataloader:
        logger.info(f"Validation samples: {len(val_dataset)}")

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Weights & Biases run name",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        config["checkpointing"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.wandb_run_name:
        config["logging"]["wandb_run_name"] = args.wandb_run_name

    # Set random seed
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_save_path}")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DiffusionTrainer(config)

    # Setup data loaders
    logger.info("Setting up data loaders...")
    train_dataloader, val_dataloader = setup_dataloaders(config, trainer.tokenizer)

    # Apply LoRA if specified
    finetune_method = config["model"].get("finetune_method", "full")
    if finetune_method == "lora":
        logger.info("Applying LoRA to model...")
        from ml.src.training.lora import apply_lora
        trainer.model = apply_lora(trainer.model, config["lora"])
        logger.info(f"Trainable parameters after LoRA: {count_parameters(trainer.model, only_trainable=True):,}")

    # Setup logging
    if config["logging"].get("use_wandb", False):
        try:
            import wandb
            wandb.init(
                project=config["logging"].get("wandb_project", "sdxl-finetuning"),
                name=config["logging"].get("wandb_run_name"),
                config=config,
            )
            logger.info("Initialized Weights & Biases logging")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    if config["logging"].get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(config["logging"].get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"Initialized TensorBoard logging at {log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping TensorBoard logging")

    # Start training
    logger.info("Starting training...")
    trainer.train(train_dataloader, val_dataloader)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
