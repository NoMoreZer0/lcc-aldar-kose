from pathlib import Path

from PIL import Image

from ml.src.diffusion.pipeline import StoryboardGenerationPipeline
from ml.src.utils.schema import MetricsPayload, Shot


class DummyEvaluator:
    def evaluate_sequence(self, frames_dir):
        return MetricsPayload(
            identity_similarity=1.0,
            background_consistency={"ssim": 1.0, "lpips": 0.0},
            scene_diversity=0.5,
        )


class DummyEngine:
    def __init__(self):
        self.txt2img = self
        self.device = "cpu"

    def generate(self, **kwargs):
        return Image.new("RGB", (64, 64), color="white")

    def set_controlnet(self, *_args, **_kwargs):
        return None


def test_pipeline_smoke(tmp_path, monkeypatch):
    config_path = Path("ml/configs/default.yaml")
    from ml.src.utils.schema import ConfigModel

    config_model = ConfigModel.from_path(config_path)
    config = config_model.model_dump() if hasattr(config_model, "model_dump") else config_model.dict()
    config["model"]["use_controlnet"] = True
    config["_engine"] = DummyEngine()
    config["_evaluator"] = DummyEvaluator()

    pipeline = StoryboardGenerationPipeline(config=config)
    shots = [
        Shot(frame_id=1, caption="Test 1", prompt="Prompt 1", camera_direction="wide shot", style_tag="style"),
        Shot(frame_id=2, caption="Test 2", prompt="Prompt 2", camera_direction="close-up", style_tag="style"),
    ]
    output_dir = tmp_path / "outputs"
    payload = pipeline.run(
        logline="Test logline.",
        shots=shots,
        output_dir=output_dir,
        use_controlnet=True,
        base_seed=42,
    )

    assert (output_dir / "frame_01.png").exists()
    assert payload.run_id == output_dir.name
