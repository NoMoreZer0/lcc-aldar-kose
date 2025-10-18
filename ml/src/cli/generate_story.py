from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ml.src.diffusion.pipeline import StoryboardGenerationPipeline
from ml.src.llm.story_context import build_plan_payload, frames_to_shots, generate_story_frames
from ml.src.utils import io as io_utils


def make_contact_sheet(images: List[Image.Image], cols: int = 3) -> Image.Image:
    if not images:
        raise ValueError("No images provided for contact sheet.")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        sheet.paste(img, (x, y))
    return sheet

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a consistent multi-frame story using SDXL + ConsiStory (GPT-5 prompts only).")
    parser.add_argument("--premise", type=str, required=True, help="High-level story premise.")
    parser.add_argument("--n_frames", type=int, default=6, help="Number of frames to generate.")
    parser.add_argument("--style", type=str, default="cinematic", help="Style preset (illustration, cinematic, manga, etc.)")
    parser.add_argument("--consistency", type=float, default=0.6, help="ConsiStory consistency strength (0..1).")
    parser.add_argument("--output", type=str, default="story", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    # Kept for compatibility but now a no-op; we always require OpenAI
    parser.add_argument("--force-regen", action="store_true", help="(Ignored) Plan is always regenerated from GPT-5.")
    parser.add_argument("--require-openai", action="store_true", help="(Ignored) OpenAI is always required.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("StoryGen")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set — GPT-5 prompts are required.")

    out_dir = Path(args.output)
    io_utils.ensure_dir(out_dir)

    # Step 1: ALWAYS generate frames via GPT-5 (no fallback, ignore any existing plan.json)
    logger.info("Generating frames via GPT-5 (no fallback).")
    frames = generate_story_frames(args.premise, args.n_frames)

    # Persist plan.json (for inspection only; generation always comes from GPT-5 next runs as well)
    plan_path = out_dir / "plan.json"
    plan = build_plan_payload(args.premise, frames)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    logger.info("Saved story plan to %s", plan_path)

    # Build Shot-like objects for the pipeline from frames (prompts are verbatim from GPT-5)
    shots = frames_to_shots(frames)

    pipeline_cfg: Dict[str, Any] = {
        "model": {
            "base_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "guidance": 7.0,
        },
        "consistency": {
            "use_consistory": True,
            "strength": args.consistency,
        },
    }

    pipeline = StoryboardGenerationPipeline(config=pipeline_cfg, logger=logger)
    storyboard = pipeline.run(
        logline=args.premise,
        shots=shots,
        output_dir=out_dir / "frames",
        base_seed=args.seed,
    )

    # Step 3: Create contact sheet
    frame_paths = [out_dir / "frames" / f.filename for f in storyboard.frames]
    images = [Image.open(fp) for fp in frame_paths]
    sheet = make_contact_sheet(images)
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    logger.info("Saved contact sheet to %s", sheet_path)

    logger.info("Story generation complete. Outputs saved under %s", out_dir)


if __name__ == "__main__":
    main()
