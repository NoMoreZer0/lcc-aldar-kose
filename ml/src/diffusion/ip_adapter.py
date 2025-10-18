from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional
from PIL import Image

# ---- Presets ---------------------------------------------------------------
IP_ADAPTER_PRESETS = {
    "plus_sdxl_vit-h": {
        "repository": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
    },
}


@dataclass
class IPAdapterConfig:
    preset: str = "plus_sdxl_vit-h"
    repository: Optional[str] = None
    subfolder: Optional[str] = None
    weight_name: Optional[str] = None
    weight: float = 0.8


class IPAdapterManager:
    """
    Minimal IP-Adapter manager.
    - Loads the adapter weights on init.
    - Returns a scale hint; the engine computes image embeds from the previous frame.
    """

    def __init__(self, pipeline, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        cfg_dict = config or {}
        allowed = {f.name for f in dataclasses.fields(IPAdapterConfig)}
        self.cfg = IPAdapterConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
        self.pipeline = pipeline

        preset = IP_ADAPTER_PRESETS.get(self.cfg.preset)
        if preset is None:
            raise ValueError(f"Unknown IP-Adapter preset: {self.cfg.preset}")

        repository = self.cfg.repository or preset["repository"]
        subfolder = self.cfg.subfolder or preset["subfolder"]
        weight_name = self.cfg.weight_name or preset["weight_name"]

        if not repository or not subfolder or not weight_name:
            raise ValueError(
                f"Invalid IP-Adapter configuration. "
                f"repository='{repository}', subfolder='{subfolder}', weight_name='{weight_name}'"
            )

        self.repository, self.subfolder, self.weight_name = repository, subfolder, weight_name
        self.loaded = False

    def load(self) -> None:
        self.pipeline.load_ip_adapter(self.repository, self.subfolder, self.weight_name)
        self.logger.info("IP-Adapter loaded: repo=%s, subfolder=%s, weight=%s", self.repository, self.subfolder, self.weight_name)
        self.loaded = True

    def get_kwargs(self, _image: Image.Image, weight: Optional[float] = None) -> Dict[str, Any]:
        """
        Engine will compute ip_adapter_image_embeds from the previous frame.
        We only provide the intended scale here.
        """
        return {"ip_adapter_scale": weight or self.cfg.weight}
