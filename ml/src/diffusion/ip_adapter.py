from __future__ import annotations

import logging
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image


@dataclass
class IPAdapterConfig:
    repository: str = "h94/IP-Adapter"
    subfolder: Optional[str] = "models/ip-adapter-plus_sdxl_vit-h"
    weight: float = 0.8
    weight_name: Optional[str] = "ip-adapter-plus_sdxl_vit-h.safetensors"
    weight_path: Optional[str] = None


class IPAdapterManager:
    def __init__(self, pipeline, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        config = config or {}
        allowed = {field.name for field in dataclasses.fields(IPAdapterConfig)}
        filtered = {key: value for key, value in config.items() if key in allowed}
        self.cfg = IPAdapterConfig(**filtered)
        self.pipeline = pipeline
        self.available = False

        if not hasattr(pipeline, "load_ip_adapter"):
            self.logger.warning("Pipeline does not support IP-Adapter integration.")
            return

        load_source = self.cfg.repository
        load_kwargs: Dict[str, Optional[str]] = {}
        default_weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"

        if self.cfg.weight_path:
            weight_path = Path(os.path.expanduser(self.cfg.weight_path)).resolve()
            if weight_path.exists():
                load_source = str(weight_path.parent)
                load_kwargs["weight_name"] = self.cfg.weight_name or weight_path.name
                if self.cfg.subfolder:
                    candidate = weight_path.parent / self.cfg.subfolder
                    if candidate.exists() and candidate.is_dir():
                        load_kwargs["subfolder"] = self.cfg.subfolder
                    else:
                        self.logger.debug("Ignoring subfolder '%s' for local IP-Adapter path; directory not found.", self.cfg.subfolder)
                self.logger.info("Loading IP-Adapter from local path: %s", weight_path)
            else:
                self.logger.warning("Configured IP-Adapter weight path does not exist: %s", weight_path)
        else:
            default_candidate = Path.home() / ".cache/huggingface/hub/IP-Adapter/ip-adapter_sdxl.safetensors"
            if default_candidate.exists():
                load_source = str(default_candidate.parent)
                load_kwargs["weight_name"] = default_candidate.name
                self.logger.info("Detected local IP-Adapter weights at %s", default_candidate)
            else:
                if self.cfg.subfolder:
                    load_kwargs["subfolder"] = self.cfg.subfolder
                if self.cfg.weight_name:
                    load_kwargs["weight_name"] = self.cfg.weight_name

        load_kwargs.setdefault("subfolder", None)
        try:
            pipeline.load_ip_adapter(
                load_source,
                **load_kwargs,
            )
            self.available = True
            self.logger.info("IP-Adapter loaded from %s", load_source)
        except Exception as exc:  # pragma: no cover - external dependency
            self.logger.warning("Failed to load IP-Adapter: %s", exc)

    def get_kwargs(self, image: Image.Image, weight: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if not self.available or not hasattr(self.pipeline, "encode_ip_adapter_image"):
            return {}

        target_weight = weight or self.cfg.weight
        image = image.convert("RGB")
        try:
            image_embeds = self.pipeline.encode_ip_adapter_image(image)
        except Exception:  # pragma: no cover - adapter integration fallback
            return {}
        return {
            "image_embeds": image_embeds,
            "ip_adapter_scale": target_weight,
        }
