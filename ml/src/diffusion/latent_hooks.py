from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineOutput


@dataclass
class LatentReuseConfig:
    strength: float = 0.35
    reuse_seed: bool = True


class LatentHookManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        config = config or {}
        allowed_keys = {f.name for f in fields(LatentReuseConfig)}
        filtered = {key: value for key, value in config.items() if key in allowed_keys}
        self.cfg = LatentReuseConfig(**filtered)
        self._last_latents: Optional[torch.Tensor] = None

    def capture(self, output: StableDiffusionXLPipelineOutput) -> None:
        latents = getattr(output, "latents", None)
        if latents is not None:
            self._last_latents = latents.detach()

    def get_latents(self) -> Optional[torch.Tensor]:
        return self._last_latents

    def prepare_kwargs(self, seed: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.cfg.reuse_seed:
            kwargs["generator"] = torch.Generator().manual_seed(seed)
        return kwargs
