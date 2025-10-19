from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..diffusion.pipeline import StoryboardGenerationPipeline
from ..llm.planner import plan_shots
from ..utils import io as io_utils
from ..utils.determinism import configure_determinism, set_seed
from ..utils.logging_setup import configure_logging
from ..utils.schema import ConfigModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Aldar Köse storyboard frames.")
    parser.add_argument("--logline", type=str, required=True, help="2-4 sentence logline.")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames (6-10).")
    parser.add_argument("--config", type=Path, default=Path("ml/configs/default.yaml"), help="Path to config YAML.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--use_controlnet", action="store_true", default=True, help="Enable ControlNet/T2I adapters.")
    parser.add_argument("--use_img2img", action="store_true", help="Enable img2img for identity consistency (overrides config).")
    parser.add_argument("--img2img_strength", type=float, default=None, help="img2img strength 0.0-1.0 (overrides config).")
    parser.add_argument("--zip", action="store_true", help="Package outputs into a zip archive.")
    parser.add_argument(
        "--require_openai",
        action="store_true",
        default=False,
        help="Require OpenAI for planning; if missing API key, error instead of falling back."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; planner will use deterministic fallback. Plans may repeat for identical loglines.")

    config = ConfigModel.from_path(args.config)
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    else:  # Pydantic v1 fallback
        config_dict = config.dict()

    # Apply CLI overrides for img2img settings
    if args.use_img2img:
        if "consistency" not in config_dict:
            config_dict["consistency"] = {}
        config_dict["consistency"]["use_img2img"] = True
        logger.info("img2img mode enabled via CLI")

    if args.img2img_strength is not None:
        if "consistency" not in config_dict:
            config_dict["consistency"] = {}
        config_dict["consistency"]["img2img_strength"] = args.img2img_strength
        logger.info("img2img strength set to %.2f via CLI", args.img2img_strength)

    configure_determinism(True)
    base_seed = set_seed(args.seed)

    logger.info("Planning %d shots with base seed %s", args.frames, base_seed)
    shots = plan_shots(args.logline, n_frames=args.frames, require_openai=args.require_openai)

    run_id = io_utils.default_run_id(prefix="storyboard")
    output_dir = args.output or Path("ml/outputs") / run_id

    pipeline = StoryboardGenerationPipeline(config_dict, logger=logger)
    index_payload = pipeline.run(
        logline=args.logline,
        shots=shots,
        output_dir=output_dir,
        use_controlnet=args.use_controlnet or config.model.get("use_controlnet", True),
        base_seed=base_seed,
    )

    index_path = output_dir / "index.json"
    index_dict = index_payload.model_dump() if hasattr(index_payload, "model_dump") else index_payload.dict()
    io_utils.dump_json(index_path, index_dict)
    logger.info("Storyboard written to %s", output_dir)
    logger.info("Index file: %s", index_path)

    if args.zip:
        archive = io_utils.pack_outputs_to_zip(output_dir)
        logger.info("Packaged results: %s", archive)


if __name__ == "__main__":
    main()
