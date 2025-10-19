"""
Generate class images for DreamBooth prior preservation.

This script generates generic images of the class (e.g., "a person")
to prevent catastrophic forgetting during DreamBooth training.
"""
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
from diffusers import StableDiffusionXLPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate class images for DreamBooth prior preservation"
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        required=True,
        help="Class prompt (e.g., 'a photo of a person')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save class images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="Number of class images to generate",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model ID to use for generation",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None for random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if images already exist
    existing_images = list(output_dir.glob("*.png"))
    if len(existing_images) >= args.num_images:
        logger.info(
            f"Found {len(existing_images)} existing images in {output_dir}, "
            f"already have enough (need {args.num_images})"
        )
        return

    # Load pipeline
    logger.info(f"Loading pipeline from {args.model_id}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None,
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()

    # Set seed if specified
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # Generate images
    num_to_generate = args.num_images - len(existing_images)
    logger.info(f"Generating {num_to_generate} class images...")

    for i in tqdm(range(0, num_to_generate, args.batch_size)):
        batch_size = min(args.batch_size, num_to_generate - i)

        # Generate batch
        images = pipe(
            prompt=[args.class_prompt] * batch_size,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images

        # Save images
        for j, image in enumerate(images):
            image_path = output_dir / f"class_image_{len(existing_images) + i + j:04d}.png"
            image.save(image_path)

    logger.info(f"Generated {num_to_generate} class images in {output_dir}")
    logger.info(f"Total class images: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
