from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import ControlNetModel
from PIL import Image


@dataclass
class ControlNetConfig:
    depth_adapter_id: str = "diffusers/controlnet-depth-sdxl-1.0"
    pose_adapter_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
    edge_adapter_id: str = "diffusers/controlnet-canny-sdxl-1.0"
    weight: float = 0.9
    union_path: Optional[str] = None
    depth_path: Optional[str] = None
    pose_path: Optional[str] = None
    edge_path: Optional[str] = None


class ControlNetManager:
    """
    Simplified ControlNet manager with a single clear code path:
    - Loads either an explicit local weight file or a known adapter id.
    - No try/except or hidden fallbacks: failures will raise immediately.
    - No nested conditionals beyond what's necessary.
    """

    def __init__(
        self,
        device,
        dtype: Optional[torch.dtype] = None,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        cfg_dict = config or {}
        allowed = {field.name for field in dataclasses.fields(ControlNetConfig)}
        self.cfg = ControlNetConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
        self.device = device
        self.dtype = dtype

        self.depth: Optional[ControlNetModel] = None
        self.pose: Optional[ControlNetModel] = None
        self.edge: Optional[ControlNetModel] = None
        self.union: Optional[ControlNetModel] = None

    def _to_device(self, model: ControlNetModel) -> ControlNetModel:
        return model.to(device=self.device, dtype=self.dtype) if self.dtype is not None else model.to(self.device)

    def _load_single_file(self, path: str) -> ControlNetModel:
        weight_path = Path(path).expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(f"ControlNet weight path not found: {weight_path}")
        model = ControlNetModel.from_single_file(str(weight_path), torch_dtype=self.dtype if self.dtype is not None else None)
        model = self._to_device(model)
        self.logger.info("Loaded ControlNet from local file: %s", weight_path)
        return model

    def _load_from_id(self, adapter_id: str) -> ControlNetModel:
        model = ControlNetModel.from_pretrained(adapter_id, torch_dtype=self.dtype if self.dtype is not None else None)
        model = self._to_device(model)
        self.logger.info("Loaded ControlNet: %s", adapter_id)
        return model

    def load_depth(self) -> ControlNetModel:
        if self.depth is None:
            self.depth = self._load_single_file(self.cfg.depth_path) if self.cfg.depth_path else self._load_from_id(self.cfg.depth_adapter_id)
        return self.depth

    def load_pose(self) -> ControlNetModel:
        if self.pose is None:
            self.pose = self._load_single_file(self.cfg.pose_path) if self.cfg.pose_path else self._load_from_id(self.cfg.pose_adapter_id)
        return self.pose

    def load_edge(self) -> ControlNetModel:
        if self.edge is None:
            self.edge = self._load_single_file(self.cfg.edge_path) if self.cfg.edge_path else self._load_from_id(self.cfg.edge_adapter_id)
        return self.edge

    def load_union(self) -> Optional[ControlNetModel]:
        if self.union is None and self.cfg.union_path:
            self.union = self._load_single_file(self.cfg.union_path)
        return self.union

    def get_controlnets(self, use_depth: bool, use_pose: bool, use_edge: bool):
        if self.cfg.union_path:
            union_model = self.load_union()
            requested = max(int(use_depth) + int(use_pose) + int(use_edge), 1)
            return [union_model] * requested

        nets = []
        if use_depth:
            nets.append(self.load_depth())
        if use_pose:
            nets.append(self.load_pose())
        if use_edge:
            nets.append(self.load_edge())
        return nets or None

    def compute_maps(self, image: Image.Image, use_depth: bool, use_pose: bool, use_edge: bool) -> List[Tuple[str, Image.Image]]:
        from ..utils import vision

        maps: List[Tuple[str, Image.Image]] = []

        if use_depth:
            depth = vision.depth_map(image)
            if depth:
                maps.append(("depth", depth))

        if use_pose:
            pose = vision.pose_map(image)
            if pose:
                maps.append(("pose", pose))

        if use_edge:
            edge = vision.canny_edges(image)
            if edge:
                maps.append(("edges", edge))

        return maps
