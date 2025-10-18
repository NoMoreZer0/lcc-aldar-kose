from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class Shot(BaseModel):
    frame_id: int
    caption: str
    prompt: str
    camera_direction: str
    style_tag: str


class FrameControl(BaseModel):
    depth: bool = False
    pose: bool = False
    edges: bool = False


class IdentityRef(BaseModel):
    used: bool = False
    type: Literal["ip-adapter", "faceid", "none"] = "none"


class FrameEntry(BaseModel):
    frame_id: int
    filename: str
    caption: str
    prompt: str
    seed: int
    control: FrameControl = Field(default_factory=FrameControl)
    identity_ref: IdentityRef = Field(default_factory=IdentityRef)

    @validator("filename")
    def validate_filename(cls, value: str) -> str:
        if not value.endswith(".png"):
            raise ValueError("frame filename must end with .png")
        return value


class MetricsPayload(BaseModel):
    identity_similarity: float
    background_consistency: dict
    scene_diversity: float


class ModelInfo(BaseModel):
    base: Literal["SDXL", "SD15", "Other"] = "SDXL"
    adapters: List[str] = Field(default_factory=list)


class Timestamps(BaseModel):
    started: datetime
    finished: datetime

    @classmethod
    def from_iso(cls, started: str, finished: str) -> "Timestamps":
        return cls(started=datetime.fromisoformat(started), finished=datetime.fromisoformat(finished))


class StoryboardIndex(BaseModel):
    logline: str
    run_id: str
    frames: List[FrameEntry]
    metrics: MetricsPayload
    model: ModelInfo
    timestamps: dict


class ConfigModel(BaseModel):
    model: dict
    consistency: dict
    thresholds: dict
    evaluation: dict = Field(default_factory=dict)
    logging: dict = Field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "ConfigModel":
        from .io import load_yaml

        data = load_yaml(path)
        return cls(**data)
