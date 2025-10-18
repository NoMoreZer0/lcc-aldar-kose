from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils import vision
from ..utils.schema import MetricsPayload
from .metrics import background_consistency, identity_similarity, load_sequence_frames, scene_diversity


class SequenceEvaluator:
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.thresholds = config.get("thresholds", {})

    def evaluate_sequence(self, frames_dir: Path) -> MetricsPayload:
        frames = load_sequence_frames(frames_dir)
        if not frames:
            raise ValueError(f"No frames found in {frames_dir}")

        identity_scores: List[float] = []
        ssim_scores: List[float] = []
        lpips_scores: List[float] = []

        face_embeddings: List[np.ndarray] = []
        for _, image in frames:
            crop = vision.crop_center(image, 512)
            embed = vision.compute_face_embedding(crop)
            if embed is not None:
                face_embeddings.append(embed)

        for idx in range(1, len(frames)):
            prev_id, prev_img = frames[idx - 1]
            curr_id, curr_img = frames[idx]

            if len(face_embeddings) >= len(frames):
                identity = identity_similarity(face_embeddings[idx - 1], face_embeddings[idx])
                identity_scores.append(identity)

            background = background_consistency(prev_img, curr_img)
            ssim_scores.append(background["ssim"])
            lpips_scores.append(background["lpips"])

        seq_diversity = scene_diversity([img for _, img in frames])

        identity_avg = float(np.mean(identity_scores)) if identity_scores else 0.0
        ssim_avg = float(np.mean(ssim_scores)) if ssim_scores else 0.0
        lpips_avg = float(np.mean(lpips_scores)) if lpips_scores else 0.0

        self._log_thresholds(identity_avg, ssim_avg, lpips_avg, seq_diversity)

        metrics = MetricsPayload(
            identity_similarity=identity_avg,
            background_consistency={"ssim": ssim_avg, "lpips": lpips_avg},
            scene_diversity=seq_diversity,
        )
        return metrics

    def _log_thresholds(self, identity: float, ssim: float, lpips: float, diversity: float) -> None:
        identity_min = self.thresholds.get("identity_min", 0.0)
        ssim_min = self.thresholds.get("ssim_min", 0.0)
        lpips_max = self.thresholds.get("lpips_max", 1.0)
        diversity_min = self.thresholds.get("scene_diversity_min", 0.0)

        if identity < identity_min:
            self.logger.warning("Identity similarity below threshold: %.3f < %.3f", identity, identity_min)
        if ssim < ssim_min:
            self.logger.warning("Background SSIM below threshold: %.3f < %.3f", ssim, ssim_min)
        if lpips > lpips_max:
            self.logger.warning("Background LPIPS above threshold: %.3f > %.3f", lpips, lpips_max)
        if diversity < diversity_min:
            self.logger.warning("Scene diversity below threshold: %.3f < %.3f", diversity, diversity_min)
