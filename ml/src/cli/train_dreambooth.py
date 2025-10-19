"""
DreamBooth training script for character identity learning.

Based on Hugging Face Diffusers DreamBooth implementation.
Optimized for training SDXL on specific characters (e.g., Aldar Köse).
"""
import argparse
import logging
import sys
import hashlib
import itertools
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.src.training.lora import apply_lora, save_lora_weights
from ml.src.training.utils import EMAModel, CheckpointManager, get_lr, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth training.

    Loads instance images (your character) and optionally class images
    (for prior preservation).
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        tokenizer=None,
        size: int = 1024,
        center_crop: bool = True,
    ):
        """
        Args:
            instance_data_root: Directory with instance images
            instance_prompt: Prompt for instance images (e.g., "a photo of sks aldar")
            class_data_root: Directory with class images (prior preservation)
            class_prompt: Prompt for class images (e.g., "a photo of a person")
            tokenizer: CLIP tokenizer (used for both SDXL text encoders)
            size: Image size
            center_crop: Whether to center crop
        """
        self.instance_data_root = Path(instance_data_root)
        self.instance_prompt = instance_prompt
        self.class_data_root = Path(class_data_root) if class_data_root else None
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

        # Load instance images
        self.instance_images_path = list(self.instance_data_root.glob("*.[pjP][npN][gG]"))
        self.instance_images_path.extend(list(self.instance_data_root.glob("*.[jJ][pP][eE][gG]"))
)
        self.num_instance_images = len(self.instance_images_path)

        if self.num_instance_images == 0:
            raise ValueError(f"No images found in {instance_data_root}")

        logger.info(f"Found {self.num_instance_images} instance images")

        # Load class images if using prior preservation
        self.class_images_path = []
        if self.class_data_root and self.class_data_root.exists():
            self.class_images_path = list(self.class_data_root.glob("*.[pjP][npN][gG]"))
            self.class_images_path.extend(list(self.class_data_root.glob("*.[jJ][pP][eE][gG]")))
            self.num_class_images = len(self.class_images_path)
            logger.info(f"Found {self.num_class_images} class images")
        else:
            self.num_class_images = 0

        # Determine total length
        self._length = max(self.num_instance_images, self.num_class_images)

        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = {}

        # Get instance image (cycle if needed)
        instance_idx = idx % self.num_instance_images
        instance_image = Image.open(self.instance_images_path[instance_idx]).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = self.instance_prompt

        # Tokenize instance prompt
        if self.tokenizer:
            example["instance_prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

        # Get class image if using prior preservation
        if self.num_class_images > 0:
            class_idx = idx % self.num_class_images
            class_image = Image.open(self.class_images_path[class_idx]).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

            if self.tokenizer:
                example["class_prompt_ids"] = self.tokenizer(
                    self.class_prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(0)

        return example


def collate_fn(examples):
    """Collate function for DreamBooth data loader."""
    batch = {
        "instance_images": torch.stack([ex["instance_images"] for ex in examples]),
        "instance_prompt_ids": torch.stack([ex["instance_prompt_ids"] for ex in examples]),
    }

    if "class_images" in examples[0]:
        batch["class_images"] = torch.stack([ex["class_images"] for ex in examples])
        batch["class_prompt_ids"] = torch.stack([ex["class_prompt_ids"] for ex in examples])

    return batch


def main():
    parser = argparse.ArgumentParser(description="DreamBooth training for SDXL")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="VAE model (use fp16-fix for stability)",
    )

    # Data arguments
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Directory with instance images",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Instance prompt (e.g., 'a photo of sks aldar')",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Directory with class images (prior preservation)",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="Class prompt (e.g., 'a photo of a person')",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="Number of class images to generate (if class_data_dir empty)",
    )

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # Prior preservation
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Enable prior preservation loss",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=64)

    # Text encoder training
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Train text encoders (stronger identity binding)",
    )
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6)

    # Validation
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=100)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Memory optimizations
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info(f"Loading models from {args.pretrained_model_name_or_path}")

    # Load tokenizer and text encoders
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
    )

    # Load VAE
    if args.pretrained_vae_model_name_or_path:
        logger.info(f"Loading VAE from {args.pretrained_vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    text_encoder.to(device)
    text_encoder_2.to(device)
    unet.to(device)

    # Freeze VAE
    vae.requires_grad_(False)

    # Freeze text encoders unless training them
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

    # Apply gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()

    # Apply xformers
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Applying LoRA...")
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        }
        unet = apply_lora(unet, lora_config)

        if args.train_text_encoder:
            text_encoder = apply_lora(text_encoder, lora_config)
            text_encoder_2 = apply_lora(text_encoder_2, lora_config)

    # Print parameter counts
    logger.info(f"UNet trainable parameters: {count_parameters(unet, only_trainable=True):,}")

    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Create dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt if args.with_prior_preservation else None,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Setup optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer")
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    # Get trainable parameters
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.train_text_encoder:
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder_2.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch

    # Setup LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    logger.info(f"***** Starting DreamBooth training *****")
    logger.info(f"  Instance images = {train_dataset.num_instance_images}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps = {max_train_steps}")
    logger.info(f"  Learning rate = {args.learning_rate}")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training")

    unet.train()
    if args.train_text_encoder:
        text_encoder.train()
        text_encoder_2.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Move to device
            instance_images = batch["instance_images"].to(device)
            instance_prompt_ids = batch["instance_prompt_ids"].to(device)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(instance_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]

            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings from both SDXL encoders
            with torch.no_grad() if not args.train_text_encoder else torch.enable_grad():
                # First text encoder (CLIP ViT-L)
                prompt_embeds_1 = text_encoder(instance_prompt_ids, output_hidden_states=True)
                pooled_prompt_embeds_1 = prompt_embeds_1[0]

                # Second text encoder (CLIP ViT-G) - provides pooled embeddings
                prompt_embeds_2 = text_encoder_2(instance_prompt_ids, output_hidden_states=True)
                pooled_prompt_embeds_2 = prompt_embeds_2.text_embeds

                # Concatenate hidden states from both encoders
                encoder_hidden_states = torch.cat([pooled_prompt_embeds_1, prompt_embeds_2.hidden_states[-2]], dim=-1)

                # SDXL requires add_time_ids for original size, crops, and target size
                # Format: [original_height, original_width, crop_top, crop_left, target_height, target_width]
                add_time_ids = torch.tensor([
                    [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]
                ], dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32).repeat(batch_size, 1).to(device)

                # Create added conditioning kwargs for SDXL
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds_2,
                    "time_ids": add_time_ids
                }

            # Predict noise with SDXL conditioning
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]

            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Prior preservation loss
            if args.with_prior_preservation and "class_images" in batch:
                class_images = batch["class_images"].to(device)
                class_prompt_ids = batch["class_prompt_ids"].to(device)

                with torch.no_grad():
                    class_latents = vae.encode(class_images).latent_dist.sample()
                    class_latents = class_latents * vae.config.scaling_factor

                class_noise = torch.randn_like(class_latents)
                class_timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (class_latents.shape[0],),
                    device=device,
                ).long()

                noisy_class_latents = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

                with torch.no_grad() if not args.train_text_encoder else torch.enable_grad():
                    # First text encoder (CLIP ViT-L)
                    class_prompt_embeds_1 = text_encoder(class_prompt_ids, output_hidden_states=True)
                    class_pooled_prompt_embeds_1 = class_prompt_embeds_1[0]

                    # Second text encoder (CLIP ViT-G) - provides pooled embeddings
                    class_prompt_embeds_2 = text_encoder_2(class_prompt_ids, output_hidden_states=True)
                    class_pooled_prompt_embeds_2 = class_prompt_embeds_2.text_embeds

                    # Concatenate hidden states from both encoders
                    class_encoder_hidden_states = torch.cat([class_pooled_prompt_embeds_1, class_prompt_embeds_2.hidden_states[-2]], dim=-1)

                    # Create time IDs for class images
                    class_add_time_ids = torch.tensor([
                        [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]
                    ], dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32).repeat(class_latents.shape[0], 1).to(device)

                    # Create added conditioning kwargs for class images
                    class_added_cond_kwargs = {
                        "text_embeds": class_pooled_prompt_embeds_2,
                        "time_ids": class_add_time_ids
                    }

                class_model_pred = unet(
                    noisy_class_latents,
                    class_timesteps,
                    class_encoder_hidden_states,
                    added_cond_kwargs=class_added_cond_kwargs,
                    return_dict=False
                )[0]

                prior_loss = F.mse_loss(class_model_pred.float(), class_noise.float(), reduction="mean")
                loss = loss + args.prior_loss_weight * prior_loss

            # Backward
            loss.backward()

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{get_lr(optimizer):.2e}"})

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)

                    if args.use_lora:
                        save_lora_weights(unet, checkpoint_dir / "unet_lora.pt")
                        if args.train_text_encoder:
                            save_lora_weights(text_encoder, checkpoint_dir / "text_encoder_lora.pt")
                            save_lora_weights(text_encoder_2, checkpoint_dir / "text_encoder_2_lora.pt")
                    else:
                        unet.save_pretrained(checkpoint_dir / "unet")
                        if args.train_text_encoder:
                            text_encoder.save_pretrained(checkpoint_dir / "text_encoder")
                            text_encoder_2.save_pretrained(checkpoint_dir / "text_encoder_2")

                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    # Save final model
    logger.info("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    if args.use_lora:
        save_lora_weights(unet, final_dir / "unet_lora.pt")
        if args.train_text_encoder:
            save_lora_weights(text_encoder, final_dir / "text_encoder_lora.pt")
            save_lora_weights(text_encoder_2, final_dir / "text_encoder_2_lora.pt")
    else:
        # Save as full pipeline
        pipeline = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(final_dir)

    logger.info(f"Training complete! Model saved to {final_dir}")
    logger.info(f"\nTo use your model:")
    logger.info(f'  prompt = "{args.instance_prompt} [your scene description]"')


if __name__ == "__main__":
    main()
