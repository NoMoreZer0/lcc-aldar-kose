from __future__ import annotations

import logging
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from diffusers import ControlNetModel, T2IAdapter
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
    def __init__(self, device, dtype: Optional[torch.dtype] = None, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        config = config or {}
        allowed = {field.name for field in dataclasses.fields(ControlNetConfig)}
        filtered = {key: value for key, value in config.items() if key in allowed}
        self.cfg = ControlNetConfig(**filtered)
        self.device = device
        self.dtype = dtype

        self.depth: Optional[ControlNetModel] = None
        self.pose: Optional[ControlNetModel] = None
        self.edge: Optional[ControlNetModel] = None
        self.t2i_adapters: Dict[str, T2IAdapter] = {}
        self.union: Optional[ControlNetModel] = None

    def _load_single_file(self, path: str) -> Optional[ControlNetModel]:
        weight_path = Path(os.path.expanduser(path)).resolve()
        if not weight_path.exists():
            self.logger.warning("ControlNet weight path not found: %s", weight_path)
            return None
        try:
            model = ControlNetModel.from_single_file(
                str(weight_path),
                torch_dtype=self.dtype if self.dtype is not None else None,
            )
            to_kwargs = {"device": self.device}
            if self.dtype is not None:
                to_kwargs["dtype"] = self.dtype
            model = model.to(**to_kwargs)
            self.logger.info("Loaded ControlNet from local file: %s", weight_path)
            return model
        except Exception as exc:  # pragma: no cover - external dependency
            self.logger.warning("Failed to load ControlNet from %s: %s", weight_path, exc)
            return None

    def load_depth(self) -> Optional[ControlNetModel]:
        if self.depth is None:
            if self.cfg.depth_path:
                self.depth = self._load_single_file(self.cfg.depth_path)
            else:
                try:
                    self.depth = ControlNetModel.from_pretrained(
                        self.cfg.depth_adapter_id,
                        torch_dtype=self.dtype if self.dtype is not None else None,
                    )
                    to_kwargs = {"device": self.device}
                    if self.dtype is not None:
                        to_kwargs["dtype"] = self.dtype
                    self.depth = self.depth.to(**to_kwargs)
                    self.logger.info("Loaded depth ControlNet: %s", self.cfg.depth_adapter_id)
                except Exception as exc:  # pragma: no cover - external download path
                    self.logger.warning("Failed to load depth ControlNet (%s): %s", self.cfg.depth_adapter_id, exc)
                    self.depth = None
        return self.depth

    def load_pose(self) -> Optional[ControlNetModel]:
        if self.pose is None:
            if self.cfg.pose_path:
                self.pose = self._load_single_file(self.cfg.pose_path)
            else:
                try:
                    self.pose = ControlNetModel.from_pretrained(
                        self.cfg.pose_adapter_id,
                        torch_dtype=self.dtype if self.dtype is not None else None,
                    )
                    to_kwargs = {"device": self.device}
                    if self.dtype is not None:
                        to_kwargs["dtype"] = self.dtype
                    self.pose = self.pose.to(**to_kwargs)
                    self.logger.info("Loaded pose ControlNet: %s", self.cfg.pose_adapter_id)
                except Exception as exc:  # pragma: no cover - external download path
                    self.logger.warning("Failed to load pose ControlNet (%s): %s", self.cfg.pose_adapter_id, exc)
                    self.pose = None
        return self.pose

    def load_edge(self) -> Optional[ControlNetModel]:
        if self.edge is None:
            if self.cfg.edge_path:
                self.edge = self._load_single_file(self.cfg.edge_path)
            else:
                try:
                    self.edge = ControlNetModel.from_pretrained(
                        self.cfg.edge_adapter_id,
                        torch_dtype=self.dtype if self.dtype is not None else None,
                    )
                    to_kwargs = {"device": self.device}
                    if self.dtype is not None:
                        to_kwargs["dtype"] = self.dtype
                    self.edge = self.edge.to(**to_kwargs)
                    self.logger.info("Loaded edge ControlNet: %s", self.cfg.edge_adapter_id)
                except Exception as exc:  # pragma: no cover - external download path
                    self.logger.warning("Failed to load edge ControlNet (%s): %s", self.cfg.edge_adapter_id, exc)
                    self.edge = None
        return self.edge

    def load_union(self) -> Optional[ControlNetModel]:
        path = self.cfg.union_path
        if path is None:
            default_candidate = Path.home() / ".cache/huggingface/hub/controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors"
            if default_candidate.exists():
                path = str(default_candidate)
                self.logger.info("Detected local ControlNet union weights at %s", default_candidate)
        if self.union is None and path:
            self.union = self._load_single_file(path)
        return self.union

    def get_controlnets(self, use_depth: bool, use_pose: bool, use_edge: bool):
        if self.cfg.union_path:
            union_model = self.load_union()
            if union_model is None:
                return None
            requested = sum(bool(flag) for flag in (use_depth, use_pose, use_edge))
            requested = max(requested, 1)
            return [union_model] * requested

        nets = []
        if use_depth and self.load_depth():
            nets.append(self.depth)
        if use_pose and self.load_pose():
            nets.append(self.pose)
        if use_edge and self.load_edge():
            nets.append(self.edge)
        return nets or None

    def compute_maps(
        self, image: Image.Image, use_depth: bool, use_pose: bool, use_edge: bool
    ) -> List[Tuple[str, Image.Image]]:
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
            maps.append(("edges", vision.canny_edges(image)))
        return maps
