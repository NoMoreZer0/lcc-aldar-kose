"""
Dataset loaders for diffusion model training.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class StoryboardDataset(Dataset):
    """
    Dataset for training on storyboard/sequential images.

    Supports:
    - Single images with text prompts
    - Image sequences with narrative context
    - ControlNet conditioning images
    - Cached latent representations
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str = "metadata.jsonl",
        image_column: str = "image_path",
        caption_column: str = "prompt",
        conditioning_column: Optional[str] = None,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: float = 0.5,
        tokenizer = None,
        max_caption_length: int = 77,
        cache_latents: bool = False,
        vae = None,
    ):
        """
        Args:
            data_dir: Root directory containing images and metadata
            metadata_file: Name of metadata file (jsonl, json, or csv)
            image_column: Column name for image paths
            caption_column: Column name for text captions/prompts
            conditioning_column: Column name for conditioning images (ControlNet)
            resolution: Target image resolution
            center_crop: Whether to center crop images
            random_flip: Probability of random horizontal flip
            tokenizer: Text tokenizer for encoding captions
            max_caption_length: Maximum caption length in tokens
            cache_latents: Whether to cache VAE latents
            vae: VAE model for latent caching
        """
        self.data_dir = Path(data_dir)
        self.image_column = image_column
        self.caption_column = caption_column
        self.conditioning_column = conditioning_column
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.cache_latents = cache_latents
        self.vae = vae

        # Load metadata
        metadata_path = self.data_dir / metadata_file
        self.metadata = self._load_metadata(metadata_path)

        logger.info(f"Loaded {len(self.metadata)} samples from {metadata_path}")

        # Image preprocessing
        self.transform = self._build_transforms(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
        )

        # Cache for latents if enabled
        self.latent_cache: Dict[int, torch.Tensor] = {}
        if cache_latents and vae is not None:
            logger.info("Pre-caching latents...")
            self._precache_latents()

    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """Load metadata from jsonl, json, or csv file."""
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]

        elif suffix == ".csv":
            import csv
            data = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            return data

        else:
            raise ValueError(f"Unsupported metadata format: {suffix}")

    def _build_transforms(
        self,
        resolution: int,
        center_crop: bool,
        random_flip: float,
    ) -> transforms.Compose:
        """Build image preprocessing transforms."""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))

        # Center crop
        if center_crop:
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            transform_list.append(transforms.RandomCrop(resolution))

        # Random flip
        if random_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=random_flip))

        # Convert to tensor and normalize to [-1, 1]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5], [0.5]))

        return transforms.Compose(transform_list)

    def _precache_latents(self):
        """Pre-encode all images to latents using VAE."""
        self.vae.eval()
        device = next(self.vae.parameters()).device

        with torch.no_grad():
            for idx in range(len(self.metadata)):
                # Load and transform image
                image_path = self.data_dir / self.metadata[idx][self.image_column]
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.transform(image).unsqueeze(0).to(device)

                # Encode to latents
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

                # Cache on CPU
                self.latent_cache[idx] = latents.cpu()

                if (idx + 1) % 100 == 0:
                    logger.info(f"Cached {idx + 1}/{len(self.metadata)} latents")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample_meta = self.metadata[idx]

        # Load image
        image_path = self.data_dir / sample_meta[self.image_column]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        # Get caption
        caption = sample_meta.get(self.caption_column, "")

        # Tokenize caption
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_caption_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.squeeze(0)
        else:
            input_ids = None

        # Build output dict
        output = {
            "pixel_values": pixel_values,
            "caption": caption,
        }

        if input_ids is not None:
            output["input_ids"] = input_ids

        # Add cached latents if available
        if self.cache_latents and idx in self.latent_cache:
            output["latents"] = self.latent_cache[idx]

        # Load conditioning image if specified
        if self.conditioning_column and self.conditioning_column in sample_meta:
            cond_image_path = self.data_dir / sample_meta[self.conditioning_column]
            cond_image = Image.open(cond_image_path).convert("RGB")
            cond_pixel_values = self.transform(cond_image)
            output["conditioning_pixel_values"] = cond_pixel_values

        # Add any additional metadata
        output["metadata"] = sample_meta

        return output


class SequenceDataset(Dataset):
    """
    Dataset for training on sequential storyboard frames.

    Each sample contains a sequence of frames with shared narrative context.
    Useful for training models with temporal/sequential consistency.
    """

    def __init__(
        self,
        data_dir: str,
        index_file: str = "index.json",
        max_sequence_length: int = 8,
        resolution: int = 1024,
        tokenizer = None,
        max_caption_length: int = 77,
    ):
        """
        Args:
            data_dir: Root directory containing storyboard sequences
            index_file: Name of index file (should be 'index.json')
            max_sequence_length: Maximum frames per sequence
            resolution: Target image resolution
            tokenizer: Text tokenizer
            max_caption_length: Maximum caption length
        """
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

        # Load all sequences
        self.sequences = self._load_sequences()

        logger.info(f"Loaded {len(self.sequences)} sequences from {data_dir}")

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load all storyboard sequences from subdirectories."""
        sequences = []

        # Find all index.json files
        for index_path in self.data_dir.rglob("index.json"):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)

                sequence_dir = index_path.parent

                # Extract frames
                frames = index_data.get("frames", [])
                if not frames:
                    continue

                # Limit sequence length
                frames = frames[:self.max_sequence_length]

                sequences.append({
                    "sequence_dir": sequence_dir,
                    "frames": frames,
                    "metadata": index_data,
                })

            except Exception as e:
                logger.warning(f"Failed to load {index_path}: {e}")
                continue

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sequence sample."""
        sequence = self.sequences[idx]
        sequence_dir = sequence["sequence_dir"]
        frames = sequence["frames"]

        # Load all images in sequence
        pixel_values_list = []
        captions = []
        input_ids_list = []

        for frame in frames:
            # Load image
            image_path = sequence_dir / frame.get("filename", f"frame_{frame['frame_id']:03d}.png")
            if not image_path.exists():
                logger.warning(f"Frame not found: {image_path}, skipping")
                continue

            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(image)
            pixel_values_list.append(pixel_values)

            # Get caption
            caption = frame.get("prompt", "")
            captions.append(caption)

            # Tokenize
            if self.tokenizer is not None:
                text_inputs = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=self.max_caption_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids_list.append(text_inputs.input_ids.squeeze(0))

        # Stack into tensors
        pixel_values = torch.stack(pixel_values_list) if pixel_values_list else None
        input_ids = torch.stack(input_ids_list) if input_ids_list and self.tokenizer else None

        output = {
            "pixel_values": pixel_values,
            "captions": captions,
            "num_frames": len(frames),
            "metadata": sequence["metadata"],
        }

        if input_ids is not None:
            output["input_ids"] = input_ids

        return output


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Handles batching of samples with proper padding and stacking.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
    }

    # Add input_ids if available
    if "input_ids" in examples[0]:
        input_ids = torch.stack([example["input_ids"] for example in examples])
        batch["input_ids"] = input_ids

    # Add latents if cached
    if "latents" in examples[0]:
        latents = torch.stack([example["latents"] for example in examples])
        batch["latents"] = latents

    # Add conditioning images if available
    if "conditioning_pixel_values" in examples[0]:
        conditioning_pixel_values = torch.stack([
            example["conditioning_pixel_values"] for example in examples
        ])
        batch["conditioning_pixel_values"] = conditioning_pixel_values

    return batch
