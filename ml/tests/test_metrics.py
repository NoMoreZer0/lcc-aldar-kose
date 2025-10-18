import numpy as np
import pytest
from PIL import Image

from ml.src.evaluate.metrics import background_consistency, identity_similarity, scene_diversity

pytest.importorskip("lpips")


def create_image(color: int) -> Image.Image:
    array = np.full((32, 32, 3), color, dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_identity_similarity_basic():
    vec_a = np.array([1, 0, 0], dtype=float)
    vec_b = np.array([1, 0, 0], dtype=float)
    assert identity_similarity(vec_a, vec_b) == pytest.approx(1.0, rel=1e-6)


def test_background_consistency_scores():
    img_a = create_image(128)
    img_b = create_image(130)
    scores = background_consistency(img_a, img_b)
    assert "ssim" in scores and "lpips" in scores


def test_scene_diversity():
    frames = [create_image(50), create_image(200)]
    diversity = scene_diversity(frames)
    assert diversity >= 0.0
