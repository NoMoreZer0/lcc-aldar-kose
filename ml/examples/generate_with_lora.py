"""
Example: Generate storyboard using trained Aldar Köse LoRA.

This script demonstrates how to integrate your trained character LoRA
with the existing storyboard generation pipeline for consistent character
representation across all frames.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from diffusers import StableDiffusionXLPipeline
from ml.src.llm.planner import plan_shots
from ml.src.diffusion.engine import SDXLEngine


def generate_storyboard_with_lora(
    logline: str,
    num_frames: int,
    lora_path: str,
    instance_token: str = "sks aldar",
    output_dir: str = "output/storyboard_with_lora",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    """
    Generate a storyboard with character LoRA for consistency.

    Args:
        logline: Story premise
        num_frames: Number of frames to generate
        lora_path: Path to trained LoRA checkpoint
        instance_token: Trigger token for character (e.g., "sks aldar")
        output_dir: Output directory
        base_model: Base SDXL model
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Storyboard with Character LoRA")
    print("=" * 60)
    print(f"Logline: {logline}")
    print(f"Frames: {num_frames}")
    print(f"LoRA: {lora_path}")
    print(f"Token: {instance_token}")
    print()

    # Step 1: Plan shots using LLM
    print("Step 1: Planning shots with LLM...")
    shots = plan_shots(
        logline=logline,
        num_frames=num_frames,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"✓ Planned {len(shots)} shots\n")

    # Step 2: Inject instance token into all prompts
    print("Step 2: Enhancing prompts with character token...")
    for shot in shots:
        # Replace generic "Aldar Köse" references with instance token
        original_prompt = shot["prompt"]
        enhanced_prompt = original_prompt.replace("Aldar Köse", instance_token)
        enhanced_prompt = enhanced_prompt.replace("Aldar", instance_token)

        # Ensure token is always present
        if instance_token not in enhanced_prompt:
            enhanced_prompt = f"{instance_token}, {enhanced_prompt}"

        shot["prompt"] = enhanced_prompt
        print(f"  Frame {shot['frame_id']}: {enhanced_prompt[:80]}...")
    print()

    # Step 3: Load SDXL pipeline with LoRA
    print("Step 3: Loading SDXL pipeline with LoRA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None,
    )
    pipe = pipe.to(device)

    # Load LoRA weights
    lora_checkpoint = Path(lora_path)
    if lora_checkpoint.is_dir():
        # If directory, assume it's a checkpoint directory
        lora_weights = lora_checkpoint / "unet_lora.pt"
        if lora_weights.exists():
            print(f"  Loading LoRA from {lora_weights}")
            pipe.load_lora_weights(lora_checkpoint)
        else:
            # Try loading as HF checkpoint
            pipe.load_lora_weights(lora_checkpoint)
    else:
        pipe.load_lora_weights(lora_path)

    # Enable optimizations
    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()

    print("✓ Pipeline ready\n")

    # Step 4: Generate frames
    print("Step 4: Generating frames...")
    images = []

    for i, shot in enumerate(shots):
        print(f"\n  Frame {i+1}/{len(shots)}: {shot['caption']}")
        print(f"  Prompt: {shot['prompt'][:100]}...")

        # Generate image
        image = pipe(
            prompt=shot["prompt"],
            negative_prompt=shot.get("negatives", "low quality, distorted"),
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(42 + i),
        ).images[0]

        # Save frame
        frame_path = Path(output_dir) / f"frame_{i+1:03d}.png"
        image.save(frame_path)
        images.append(image)
        print(f"  ✓ Saved to {frame_path}")

    print()
    print("=" * 60)
    print(f"✓ Generated {len(images)} frames in {output_dir}")
    print("=" * 60)

    return images, shots


def compare_with_without_lora(
    prompt: str,
    lora_path: str,
    output_dir: str = "output/comparison",
):
    """
    Generate comparison images with and without LoRA.

    Useful for evaluating LoRA effectiveness.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    print("Generating comparison: With vs Without LoRA")
    print(f"Prompt: {prompt}\n")

    # Load base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    # Generate without LoRA
    print("Generating WITHOUT LoRA...")
    image_without = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=30,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]
    image_without.save(Path(output_dir) / "without_lora.png")
    print("✓ Saved without_lora.png")

    # Load LoRA
    print("\nLoading LoRA...")
    pipe.load_lora_weights(lora_path)

    # Generate with LoRA
    print("Generating WITH LoRA...")
    image_with = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=30,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]
    image_with.save(Path(output_dir) / "with_lora.png")
    print("✓ Saved with_lora.png")

    # Create side-by-side comparison
    from PIL import Image
    comparison = Image.new("RGB", (2048, 1024))
    comparison.paste(image_without, (0, 0))
    comparison.paste(image_with, (1024, 0))
    comparison.save(Path(output_dir) / "comparison.png")
    print("✓ Saved comparison.png")

    print(f"\nComparison saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate storyboard with LoRA")
    parser.add_argument(
        "--logline",
        type=str,
        default="Aldar Köse outsmarts a greedy merchant in the bazaar",
        help="Story logline",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of frames",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint",
    )
    parser.add_argument(
        "--instance_token",
        type=str,
        default="sks aldar",
        help="Instance token used in training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/storyboard_with_lora",
        help="Output directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison (with vs without LoRA)",
    )
    parser.add_argument(
        "--compare_prompt",
        type=str,
        default=None,
        help="Prompt for comparison (if --compare is set)",
    )

    args = parser.parse_args()

    if args.compare:
        # Run comparison mode
        prompt = args.compare_prompt or f"{args.instance_token} in traditional kazakh clothing, portrait, cinematic"
        compare_with_without_lora(
            prompt=prompt,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
        )
    else:
        # Run storyboard generation
        images, shots = generate_storyboard_with_lora(
            logline=args.logline,
            num_frames=args.frames,
            lora_path=args.lora_path,
            instance_token=args.instance_token,
            output_dir=args.output_dir,
        )

        # Save shot metadata
        import json
        with open(Path(args.output_dir) / "shots.json", "w") as f:
            json.dump(shots, f, indent=2)
        print(f"\nShot metadata saved to {args.output_dir}/shots.json")
