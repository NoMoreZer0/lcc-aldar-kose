from __future__ import annotations

"""
ConsiStory (training-free) for SDXL: Consistent subject identity across multiple frames.

Core ideas (inspired by NVIDIA ConsiStory, 2024):
- Subject-driven shared attention across frames: identify subject tokens/patches using cross-attention,
  and mix K/V across the batch to share identity while allowing layout diversity.
- Patch-level feature correspondence injection: compute cosine-similar patch matches between frames and blend features.
- A single forward pass over a batch of prompts (one per frame) with modified cross-attention achieves identity consistency.

This module provides:
- ConsiStoryGenerator: a wrapper that loads SDXL pipelines, applies cross-frame attention processors, and generates frames.
- CLI: python -m consistory --prompts prompts.txt --frames 4 --consistency 0.7 --seed 42

Requirements:
- Python 3.10+, PyTorch 2.0+, diffusers >= 0.25.0 (tested), xformers recommended on CUDA.
"""

import argparse
import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

# Diffusers: load both txt2img and img2img for optional identity initialization
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

from compel import Compel, ReturnedEmbeddingsType

# Our cross-frame attention utilities
from ml.src.diffusion.hooks import (
    SharedAttentionConfig,
    SharedAttentionContext,
    apply_shared_attention_to_unet,
    restore_unet_attention,
    save_attention_debug_pngs,
)


# -----------------------------
# Utilities
# -----------------------------

def _device_dtype() -> Tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype


def _load_pipelines(model_id: str, device: torch.device, dtype: torch.dtype) -> Tuple[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline]:
    """
    Load SDXL txt2img and derive an img2img pipeline using its components for efficiency.
    """
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Build compatible img2img pipeline from components if possible
    img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components).to(device)

    # Light memory optimizations
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    img2img.enable_vae_tiling()
    img2img.enable_vae_slicing()
    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            img2img.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe, img2img


def _ensure_prompts(prompts: Sequence[str], frames: Optional[int]) -> List[str]:
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("No prompts provided.")
    if frames is None or frames <= 0:
        return prompts
    if len(prompts) >= frames:
        return list(prompts[:frames])
    # Repeat last prompt to reach desired frames
    last = prompts[-1]
    return list(prompts) + [last] * (frames - len(prompts))


def _read_prompts_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # Remove empty lines
    return [ln for ln in lines if ln]


def _prepare_generators(
    seed: Optional[int],
    batch: int,
    device: torch.device,
    per_frame_offsets: bool,
) -> Union[torch.Generator, List[torch.Generator]]:
    if seed is None:
        return torch.Generator(device=device)
    if not per_frame_offsets or batch <= 1:
        return torch.Generator(device=device).manual_seed(seed)
    # Distinct but reproducible per-frame seeds for layout diversity
    gens: List[torch.Generator] = []
    for i in range(batch):
        gens.append(torch.Generator(device=device).manual_seed(seed + 9973 * i))
    return gens


def _to_pil(img: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(str(img)).convert("RGB")


def _timestamp() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class ConsiStoryParams:
    # Model
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.0

    # Consistency controls
    consistency_strength: float = 0.6
    self_keep_alpha: float = 0.7
    dropout: float = 0.1
    subject_token_top_p: float = 0.15
    subject_patch_top_p: float = 0.3
    feature_injection_weight: float = 0.5
    max_patches_for_correspondence: int = 64
    store_attention: bool = False

    # Reproducibility
    seed: Optional[int] = None
    per_frame_seed_offsets: bool = True  # if True, use different seeds per frame (derived from base)

    # Identity init (optional img2img)
    identity_image: Optional[Union[str, Path]] = None
    identity_strength: float = 0.4  # 0..1 (higher -> closer to identity_image)

    # Negative prompt
    negative_prompt: Optional[str] = None

    # Attention debug output
    attn_save_dir: Optional[Union[str, Path]] = None


# -----------------------------
# Generator
# -----------------------------

class ConsiStoryGenerator:
    def __init__(self, params: ConsiStoryParams, logger: Optional[logging.Logger] = None) -> None:
        self.params = params
        self.logger = logger or logging.getLogger("ConsiStory")
        self.device, self.dtype = _device_dtype()
        self.logger.info("Loading SDXL model: %s", params.model_id)
        self.txt2img, self.img2img = _load_pipelines(params.model_id, self.device, self.dtype)
        compel_kwargs = {
            "tokenizer": [self.txt2img.tokenizer, getattr(self.txt2img, "tokenizer_2", self.txt2img.tokenizer)],
            "text_encoder": [self.txt2img.text_encoder, getattr(self.txt2img, "text_encoder_2", self.txt2img.text_encoder)],
            "device": self.device,
        }
        if hasattr(ReturnedEmbeddingsType, "SDXL"):
            compel_kwargs["returned_embeddings_type"] = ReturnedEmbeddingsType.SDXL
        self.compel = Compel(**compel_kwargs)

    def _make_ctx(self, batch_size: int) -> SharedAttentionContext:
        cfg = SharedAttentionConfig(
            consistency_strength=self.params.consistency_strength,
            self_keep_alpha=self.params.self_keep_alpha,
            dropout=self.params.dropout,
            subject_token_top_p=self.params.subject_token_top_p,
            subject_patch_top_p=self.params.subject_patch_top_p,
            feature_injection_weight=self.params.feature_injection_weight,
            max_patches_for_correspondence=self.params.max_patches_for_correspondence,
            store_attention=bool(self.params.attn_save_dir) or self.params.store_attention,
        )
        gen = torch.Generator(device=self.device)
        if self.params.seed is not None:
            gen.manual_seed(self.params.seed)
        ctx = SharedAttentionContext(
            cfg=cfg,
            step_index=0,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
            rng=gen,
        )
        return ctx

    def _increment_step_cb(self, ctx: SharedAttentionContext):
        # diffusers callback signature: (step: int, timestep: int, latents: Tensor)
        def _cb(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            # Keep ctx in sync with current denoising step
            ctx.step_index = step
        return _cb

    @torch.no_grad()
    def generate_frames(
        self,
        prompts: Sequence[str],
        negative_prompt: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Generate N frames in a single pass (batch) with shared-identity attention.
        - prompts: list of text prompts, length N
        - negative_prompt: optional global negative prompt (applied to all)
        Returns list of N PIL images.
        """
        prompts = [p.strip() for p in prompts]
        batch = len(prompts)
        if batch == 0:
            raise ValueError("prompts must be a non-empty sequence.")
        self.logger.info("Generating %d frames with consistency strength=%.2f", batch, self.params.consistency_strength)

        # Prepare attention context and hook UNet
        ctx = self._make_ctx(batch)
        prev_processors = apply_shared_attention_to_unet(self.txt2img.unet, ctx, only_cross_attention=True)
        prev_img2img = apply_shared_attention_to_unet(self.img2img.unet, ctx, only_cross_attention=True)

        # Prepare callback to advance ctx.step_index
        callback = self._increment_step_cb(ctx)

        # Negative prompt handling
        neg = negative_prompt if negative_prompt is not None else self.params.negative_prompt

        # Generators for reproducibility
        generator = _prepare_generators(self.params.seed, batch, self.device, self.params.per_frame_seed_offsets)

        # Select pipeline (txt2img vs img2img) based on identity init
        use_img2img = self.params.identity_image is not None
        pipe = self.img2img if use_img2img else self.txt2img
        conditioning = self._build_embeddings(list(prompts))
        use_compel = conditioning is not None and conditioning[0] is not None and conditioning[1] is not None

        if use_compel:
            prompt_embeds, pooled_embeds = conditioning  # type: ignore[misc]
            pipe_kwargs = dict(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                width=self.params.width,
                height=self.params.height,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance,
                generator=generator,
                callback=callback,
                callback_steps=1,
            )

            if neg is not None:
                neg_list = [neg] * batch if isinstance(neg, str) else list(neg)
                if len(neg_list) != batch:
                    raise ValueError("negative_prompt length must match prompts length")
                neg_embeds_tuple = self._build_embeddings(neg_list)
                if neg_embeds_tuple and neg_embeds_tuple[0] is not None and neg_embeds_tuple[1] is not None:
                    pipe_kwargs["negative_prompt_embeds"] = neg_embeds_tuple[0]
                    pipe_kwargs["negative_pooled_prompt_embeds"] = neg_embeds_tuple[1]
        else:
            pipe_kwargs = dict(
                prompt=list(prompts),
                negative_prompt=[neg] * batch if isinstance(neg, str) else neg,
                width=self.params.width,
                height=self.params.height,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance,
                generator=generator,
                callback=callback,
                callback_steps=1,
            )

        # Optional identity init: replicate init image per frame
        if use_img2img:
            init_img = _to_pil(self.params.identity_image)  # type: ignore[arg-type]
            pipe_kwargs["image"] = [init_img] * batch
            pipe_kwargs["strength"] = self.params.identity_strength

        try:
            out = pipe(**pipe_kwargs)
        finally:
            # Always restore processors even on error
            restore_unet_attention(self.txt2img.unet, prev_processors)
            restore_unet_attention(self.img2img.unet, prev_img2img)

        images: List[Image.Image]
        if hasattr(out, "images"):
            images = list(out.images)
        else:
            # Old style tuple return
            images = list(out[0])

        # Save attention maps if requested
        if self.params.attn_save_dir:
            save_dir = Path(self.params.attn_save_dir)
            written = save_attention_debug_pngs(ctx.attn_debug, save_dir)
            self.logger.info("Saved %d attention debug images to %s", len(written), save_dir)

        return images

    def _build_embeddings(self, prompts: Sequence[str]) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if not prompts:
            return None
        try:
            conditioning = self.compel.build_conditioning_tensor(list(prompts))
        except Exception:
            self.logger.debug("Compel failed to encode prompts; falling back to raw text", exc_info=True)
            return None

        if isinstance(conditioning, tuple) and len(conditioning) == 2:
            return conditioning  # type: ignore[return-value]

        if isinstance(conditioning, torch.Tensor):
            pooled = self._compute_pooled_embedding(prompts)
            return conditioning, pooled

        return None

    def _compute_pooled_embedding(self, prompts: Sequence[str]) -> Optional[torch.Tensor]:
        if not prompts:
            return None
        try:
            _, _, pooled, _ = self.txt2img._encode_prompt(
                prompt=list(prompts),
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            return pooled
        except Exception:
            self.logger.debug("Failed to compute pooled embedding via pipeline", exc_info=True)
            return None


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConsiStory (training-free) SDXL multi-frame generation with shared identity.")
    parser.add_argument("--prompts", type=str, required=True, help="Path to a text file with one prompt per line.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to generate (truncate/repeat prompts to match).")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Model ID for SDXL base.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.0, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed.")
    parser.add_argument("--no-per-frame-seed-offsets", action="store_true", help="Disable per-frame seed offsets.")
    parser.add_argument("--negative", type=str, default=None, help="Global negative prompt.")
    parser.add_argument("--consistency", type=float, default=0.6, help="Consistency strength (0..1).")
    parser.add_argument("--alpha", type=float, default=0.7, help="Self keep alpha for K/V mixing (fraction of self to keep).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout on subject-token sharing (0..1).")
    parser.add_argument("--token-top-p", type=float, default=0.15, help="Top-p tokens treated as subject.")
    parser.add_argument("--patch-top-p", type=float, default=0.3, help="Top-p spatial positions treated as subject region.")
    parser.add_argument("--feat-inject", type=float, default=0.5, help="Feature correspondence injection weight (0..1).")
    parser.add_argument("--max-patches", type=int, default=64, help="Max subject patches per frame for correspondence.")
    parser.add_argument("--identity", type=str, default=None, help="Optional identity reference image path (enables img2img).")
    parser.add_argument("--identity-strength", type=float, default=0.4, help="Strength for identity init (img2img).")
    parser.add_argument("--attn-save-dir", type=str, default=None, help="Directory to save attention debug maps.")
    parser.add_argument("--output", type=str, default=None, help="Output directory to save generated frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("ConsiStoryCLI")

    prompts_path = Path(args.prompts)
    prompts = _read_prompts_file(prompts_path)
    prompts = _ensure_prompts(prompts, args.frames)
    n = len(prompts)

    # Build parameters
    params = ConsiStoryParams(
        model_id=args.model,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        per_frame_seed_offsets=not args.no_per_frame_seed_offsets,
        negative_prompt=args.negative,
        consistency_strength=args.consistency,
        self_keep_alpha=args.alpha,
        dropout=args.dropout,
        subject_token_top_p=args.token_top_p,
        subject_patch_top_p=args.patch_top_p,
        feature_injection_weight=args.feat_inject,
        max_patches_for_correspondence=args.max_patches,
        identity_image=args.identity,
        identity_strength=args.identity_strength,
        attn_save_dir=args.attn_save_dir,
    )

    logger.info("Params: %s", {k: v for k, v in asdict(params).items() if k not in {"identity_image"}})

    # Run generation
    gen = ConsiStoryGenerator(params, logger=logger)
    images = gen.generate_frames(prompts, negative_prompt=args.negative)

    # Save outputs
    out_dir = Path(args.output) if args.output else Path("ml/outputs") / f"consistory_run_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        fp = out_dir / f"frame_{i:02d}.png"
        img.save(fp)
    logger.info("Wrote %d frames to %s", n, out_dir.as_posix())

    if args.attn_save_dir:
        logger.info("Attention maps saved under %s", args.attn_save_dir)


if __name__ == "__main__":
    main()
