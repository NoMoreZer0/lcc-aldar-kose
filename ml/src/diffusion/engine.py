from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import constants as hf_hub_constants

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
hf_hub_constants.HF_HUB_ENABLE_HF_TRANSFER = False

from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

try:  # pragma: no cover - optional dependency across diffusers versions
    from diffusers import MultiControlNetModel
except ImportError:  # pragma: no cover
    MultiControlNetModel = None  # type: ignore
from PIL import Image


@dataclass
class EngineConfig:
    base_id: str
    height: int = 1024
    width: int = 1024
    steps: int = 30
    guidance: float = 7.5
    use_controlnet: bool = True
    negative_prompt: Optional[str] = None


class SDXLEngine:
    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = cfg or {}
        engine_field_names = {f.name for f in fields(EngineConfig)}
        engine_kwargs = {key: value for key, value in cfg.items() if key in engine_field_names}
        self.cfg = EngineConfig(**engine_kwargs)

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.logger.debug("Loading SDXL pipelines (%s)...", self.cfg.base_id)

        load_kwargs = {"use_safetensors": True}
        try:
            self.txt2img = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.base_id,
                torch_dtype=torch_dtype,
                **load_kwargs,
            ).to(self.device)
        except TypeError:
            self.txt2img = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.base_id,
                dtype=torch_dtype,
                **load_kwargs,
            ).to(self.device)
        self.txt2img.scheduler = DPMSolverMultistepScheduler.from_config(self.txt2img.scheduler.config)
        self.dtype = next(self.txt2img.unet.parameters()).dtype

        img_signature = inspect.signature(StableDiffusionXLImg2ImgPipeline.__init__).parameters
        img_components = {
            key: value
            for key, value in self.txt2img.components.items()
            if key in img_signature and key != "self"
        }
        if "requires_safety_checker" in img_signature:
            img_components.setdefault("requires_safety_checker", False)
        self.img2img = StableDiffusionXLImg2ImgPipeline(**img_components).to(self.device)

        self._control_pipeline: Optional[StableDiffusionXLControlNetPipeline] = None
        self._control_img2img_pipeline: Optional[StableDiffusionXLControlNetImg2ImgPipeline] = None
        self._controlnet = None

        self.enable_memory_optimizations()

    def enable_memory_optimizations(self, extra_pipes: Optional[List[Any]] = None) -> None:
        pipes: List[Any] = [self.txt2img, self.img2img]
        if self._control_pipeline is not None:
            pipes.append(self._control_pipeline)
        if getattr(self, "_control_img2img_pipeline", None) is not None:
            pipes.append(self._control_img2img_pipeline)
        if extra_pipes:
            pipes.extend(extra_pipes)
        seen = set()
        for pipe in pipes:
            if pipe is None or id(pipe) in seen:
                continue
            seen.add(id(pipe))
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()
            if (
                self.device.type == "cuda"
                and torch.cuda.is_available()):
                pipe.enable_xformers_memory_efficient_attention()

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def set_controlnet(self, controlnet) -> None:
        if not controlnet:
            self._controlnet = None
            return

        prepared: Any
        if isinstance(controlnet, list):
            nets = [net.to(self.device) for net in controlnet]
            if len(nets) == 1:
                prepared = nets[0]
            elif MultiControlNetModel is not None:
                prepared = MultiControlNetModel(nets).to(self.device)
            else:  # pragma: no cover - legacy diffusers fallback
                self.logger.warning("Multiple ControlNets requested but MultiControlNetModel unavailable; using first model only.")
                prepared = nets[0]
        else:
            prepared = controlnet.to(self.device)

        self._controlnet = prepared

        if self._control_pipeline is not None or self._control_img2img_pipeline is not None:
            if self._control_pipeline is not None:
                self._control_pipeline.controlnet = prepared
            if self._control_img2img_pipeline is not None:
                self._control_img2img_pipeline.controlnet = prepared
            self.enable_memory_optimizations(
                extra_pipes=[p for p in [self._control_pipeline, self._control_img2img_pipeline] if p is not None]
            )
            return

        control_signature = inspect.signature(StableDiffusionXLControlNetPipeline.__init__).parameters
        control_components = {
            key: value
            for key, value in self.txt2img.components.items()
            if key in control_signature and key not in {"self", "controlnet"}
        }
        if "requires_safety_checker" in control_signature:
            control_components.setdefault("requires_safety_checker", False)
        control_components["controlnet"] = prepared
        try:
            self._control_pipeline = StableDiffusionXLControlNetPipeline(**control_components).to(self.device)
        except TypeError:
            dtype = next(self.txt2img.unet.parameters()).dtype
            try:
                self._control_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.cfg.base_id,
                    controlnet=prepared,
                    torch_dtype=dtype,
                    use_safetensors=True,
                ).to(self.device)
            except TypeError:
                self._control_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.cfg.base_id,
                    controlnet=prepared,
                    dtype=dtype,
                    use_safetensors=True,
                ).to(self.device)
        else:
            self._control_pipeline.scheduler = self.txt2img.scheduler

        # Build ControlNet Img2Img pipeline
        try:
            control_img2img_signature = inspect.signature(StableDiffusionXLControlNetImg2ImgPipeline.__init__).parameters
            control_img2img_components = {
                key: value
                for key, value in self.img2img.components.items()
                if key in control_img2img_signature and key not in {"self", "controlnet"}
            }
            if "requires_safety_checker" in control_img2img_signature:
                control_img2img_components.setdefault("requires_safety_checker", False)
            control_img2img_components["controlnet"] = prepared
            self._control_img2img_pipeline = StableDiffusionXLControlNetImg2ImgPipeline(**control_img2img_components).to(self.device)
        except Exception:
            dtype = next(self.txt2img.unet.parameters()).dtype
            try:
                self._control_img2img_pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    self.cfg.base_id,
                    controlnet=prepared,
                    torch_dtype=dtype,
                    use_safetensors=True,
                ).to(self.device)
            except Exception:
                self._control_img2img_pipeline = None  # fallback: only txt2img controlnet available

        if self._control_img2img_pipeline is not None:
            self._control_img2img_pipeline.scheduler = self.img2img.scheduler

        self.enable_memory_optimizations(
            extra_pipes=[p for p in [self._control_pipeline, self._control_img2img_pipeline] if p is not None]
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        img2img_start: Optional[Image.Image] = None,
        strength: float = 0.35,
        control_images: Optional[List[Image.Image]] = None,
        controlnet_conditioning_scale: Optional[List[float]] = None,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        negative_prompt = negative_prompt or self.cfg.negative_prompt
        width = width or self.cfg.width
        height = height or self.cfg.height
        steps = steps or self.cfg.steps
        guidance = guidance or self.cfg.guidance

        pipeline = self.txt2img
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "generator": generator,
        }

        if img2img_start is not None:
            pipeline = self.img2img
            kwargs["image"] = img2img_start
            kwargs["strength"] = strength

        if control_images and self._controlnet is not None:
            n_ctrl = len(control_images)
            if img2img_start is not None and self._control_img2img_pipeline is not None:
                # Use ControlNet Img2Img pipeline
                pipeline = self._control_img2img_pipeline
                kwargs["image"] = img2img_start
                kwargs["control_image"] = control_images
                # Do not replicate init image; let the pipeline expand for CFG/batch internally.
            else:
                # Use ControlNet txt2img pipeline
                pipeline = self._control_pipeline
                kwargs["image"] = control_images
            if controlnet_conditioning_scale is not None:
                kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            elif "controlnet_conditioning_scale" not in kwargs:
                kwargs["controlnet_conditioning_scale"] = [0.9] * n_ctrl
        elif controlnet_conditioning_scale is not None:
            self.logger.debug("Ignoring controlnet_conditioning_scale without control images.")

        self.logger.info(
            "Engine.generate: using %s; keys=%s",
            pipeline.__class__.__name__,
            sorted(kwargs.keys()),
        )

        self.logger.debug(
            "Invoking %s with keys=%s",
            pipeline.__class__.__name__,
            sorted(kwargs.keys()),
        )

        output = pipeline(**kwargs)
        if hasattr(output, "images"):
            return output.images[0]
        return output[0]
