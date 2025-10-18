import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def cv_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array[:, :, ::-1])


def resize_with_aspect(image: Image.Image, max_dim: int) -> Image.Image:
    w, h = image.size
    scale = max_dim / max(w, h)
    if scale >= 1:
        return image
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def canny_edges(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    import cv2  # lazy import

    cv_img = pil_to_cv(image)
    edges = cv2.Canny(cv_img, low_threshold, high_threshold)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return cv_to_pil(edges_rgb)


def depth_map(image: Image.Image) -> Optional[Image.Image]:
    try:
        from controlnet_aux import MidasDetector
    except ImportError:
        return None
    detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    depth = detector(resize_with_aspect(image, 768))
    return depth.resize(image.size)


def pose_map(image: Image.Image) -> Optional[Image.Image]:
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError:
        return None
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    pose = detector(image)
    return pose


def seed_everything(seed: Optional[int] = None) -> int:
    import torch

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def crop_center(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    half = size // 2
    center_x, center_y = w // 2, h // 2
    left = max(center_x - half, 0)
    upper = max(center_y - half, 0)
    right = min(left + size, w)
    lower = min(upper + size, h)
    return image.crop((left, upper, right, lower))


def compute_face_embedding(image: Image.Image) -> Optional[np.ndarray]:
    try:
        import torch
        import open_clip
    except ImportError:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model.eval()
    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0)
        features = model.encode_image(tensor)
    return features.cpu().numpy()[0]


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_keypoint_variance(keypoints: np.ndarray) -> float:
    if keypoints.size == 0:
        return 0.0
    return float(np.mean(np.var(keypoints, axis=0)))
