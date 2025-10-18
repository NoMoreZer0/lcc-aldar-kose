from pathlib import Path

from ml.src.utils.schema import ConfigModel


def test_default_config_loads():
    config_path = Path("ml/configs/default.yaml")
    config = ConfigModel.from_path(config_path)
    assert config.model["base_id"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "identity_min" in config.thresholds
