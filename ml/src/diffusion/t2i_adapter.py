from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import T2IAdapter
from PIL import Image


@dataclass
class T2IAdapterConfig:
    """Configuration for T2I-Adapter used for identity consistency."""

    model_id: str = "TencentARC/t2i-adapter-sketch-sdxl-1.0"
    adapter_conditioning_scale: float = 0.75
    adapter_conditioning_factor: float = 0.8


class T2IAdapterManager:
    """
    Manages T2I-Adapter for identity/face consistency.
    T2I-Adapters use lightweight control hints that work well with ControlNet.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.dtype = dtype

        cfg_dict = config or {}
        self.cfg = T2IAdapterConfig(
            **{k: v for k, v in cfg_dict.items() if k in T2IAdapterConfig.__annotations__}
        )

        self._adapter: Optional[T2IAdapter] = None
        self.loaded = False

    def load(self) -> T2IAdapter:
        """Load the T2I-Adapter model."""
        if self._adapter is not None:
            return self._adapter

        self.logger.info("Loading T2I-Adapter: %s", self.cfg.model_id)
        self._adapter = T2IAdapter.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.loaded = True
        self.logger.info("T2I-Adapter loaded successfully")
        return self._adapter

    def get_adapter(self) -> Optional[T2IAdapter]:
        """Get the loaded adapter."""
        if not self.loaded:
            return self.load()
        return self._adapter

    def prepare_adapter_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare an image for T2I-Adapter input.
        For sketch-based adapter, we extract edges/sketch from the image.
        """
        from ..utils.vision import canny_edges

        # Convert to grayscale sketch/edge map
        edge_image = canny_edges(image, low_threshold=50, high_threshold=150)
        return edge_image

    def get_conditioning_kwargs(
        self,
        reference_image: Image.Image,
        scale: Optional[float] = None,
        factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get conditioning kwargs for T2I-Adapter.

        Args:
            reference_image: The reference image (previous frame)
            scale: Conditioning scale override
            factor: Conditioning factor override

        Returns:
            Dictionary with adapter_image and adapter_conditioning_scale
        """
        adapter_image = self.prepare_adapter_image(reference_image)

        return {
            "adapter_image": adapter_image,
            "adapter_conditioning_scale": scale or self.cfg.adapter_conditioning_scale,
            "adapter_conditioning_factor": factor or self.cfg.adapter_conditioning_factor,
        }
