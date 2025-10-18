from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


def _to_tensor(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB")).astype("float32") / 255.0
    return arr


def identity_similarity(prev_embed: np.ndarray, curr_embed: np.ndarray) -> float:
    prev_norm = prev_embed / (np.linalg.norm(prev_embed) + 1e-8)
    curr_norm = curr_embed / (np.linalg.norm(curr_embed) + 1e-8)
    return float(np.clip(np.dot(prev_norm, curr_norm), -1.0, 1.0))


def background_consistency(prev_img: Image.Image, curr_img: Image.Image) -> Dict[str, float]:
    from skimage.metrics import structural_similarity
    import lpips  # type: ignore
    import torch

    prev = _to_tensor(prev_img)
    curr = _to_tensor(curr_img)

    ssim_score = float(structural_similarity(prev, curr, channel_axis=2, data_range=1.0))

    lpips_model = lpips.LPIPS(net="vgg")
    prev_tensor = torch.tensor(prev).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    curr_tensor = torch.tensor(curr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lpips_score = float(lpips_model(prev_tensor, curr_tensor).item())

    return {"ssim": ssim_score, "lpips": lpips_score}


def scene_diversity(images: Iterable[Image.Image]) -> float:
    histograms: List[np.ndarray] = []
    for img in images:
        hist = np.histogram(_to_tensor(img), bins=32, range=(0, 1))[0]
        histograms.append(hist / (np.linalg.norm(hist) + 1e-8))

    if len(histograms) < 2:
        return 0.0

    diversity = 0.0
    pairs = 0
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            diff = np.linalg.norm(histograms[i] - histograms[j])
            diversity += diff
            pairs += 1
    return float(diversity / max(pairs, 1))


def load_sequence_frames(frames_dir: Path) -> List[Tuple[int, Image.Image]]:
    frames: List[Tuple[int, Image.Image]] = []
    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        frame_id = int(frame_path.stem.split("_")[1])
        frames.append((frame_id, Image.open(frame_path).convert("RGB")))
    return frames
