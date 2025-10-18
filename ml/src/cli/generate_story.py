from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ml.src.diffusion.pipeline import StoryboardGenerationPipeline
from ml.src.utils import io as io_utils


ALLOWED_CAMERA_DIRECTIONS = [
    "wide establishing shot",
    "mid-shot",
    "close-up",
    "over-the-shoulder",
    "dynamic three-quarter angle",
    "low-angle hero shot",
    "bird's-eye view",
    "tracking side profile",
]

ALLOWED_STYLE_TAGS = [
    "painterly realism",
    "golden hour cinematic",
    "moonlit mystery",
    "crisp morning light",
    "warm hearth glow",
    "wind-swept steppe",
]


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

def _llm_generate_frames_inline(premise: str, n_frames: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api_key)

    # Use 'instructions' for system prompt in Responses API
    instructions = (
        "You are an expert storyboard writer for respectful tales about Aldar Köse, a Kazakh folk hero. "
        "Write a cohesive short story that is culturally authentic and respectful. "
        "Keep Aldar Köse’s identity constant across all frames: middle-aged Kazakh trickster, traditional chapan robe, "
        "kalpak hat, gentle, knowing smile. "
        "Each frame must push the story forward (setup → rising action → twist/climax → resolution). "
        "Each new frame introduces at least one meaningful change (new supporting character, new place, new prop, "
        "time-of-day shift, or clear plot turn). "
        "Prompts must be concise (aim ≤ 65 tokens) and include camera/composition/lighting/mood. "
        "Avoid stereotypes or caricatures. Output JSON only when asked to."
    )

    # Provide the task as a single input string
    user_input = (
        f"Premise: {premise}\n"
        f"Frames: {n_frames}\n\n"
        "Task:\n"
        "- First, think through a tight outline for a 1-scene short story about Aldar Köse.\n"
        f"- Then split it into exactly {n_frames} frames that advance the plot.\n\n"
        f"Return ONLY a JSON array of length {n_frames}, where each item is:\n"
        "{\n"
        f'  \"frame_id\": <int 1..{n_frames}>,\n'
        '  \"caption\": \"<1 concise sentence>\",\n'
        '  \"narration\": \"<2-3 sentences of story progression>\",\n'
        '  \"prompt\": \"<compact image prompt with camera/lens/composition/lighting/mood>\",\n'
        '  \"camera_direction\": \"<one of: '
        + ", ".join(ALLOWED_CAMERA_DIRECTIONS)
        + '>\",\n'
        '  \"style_tag\": \"<one of: '
        + ", ".join(ALLOWED_STYLE_TAGS)
        + '>\"\n'
        "}\n\n"
        "Constraints:\n"
        "- Vary camera_direction and style_tag with no immediate repeats.\n"
        "- Do not include special tokens or markup.\n"
        "- Output JSON only."
    )

    resp = client.responses.create(
        model="gpt-5",
        instructions=instructions,
        input=user_input,
        # Keep budget for visible JSON:
        reasoning={"effort": "minimal"},   # reduce hidden “thinking” tokens
        max_output_tokens=4096,            # give enough room for JSON
        tool_choice="none",                # avoid tool-call inflation
    )


    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(t)
        text = "\n".join(parts).strip()

    if not text:
        # Helpful debug if the model ran out of tokens again
        raw = resp.model_dump_json() if hasattr(resp, "model_dump_json") else str(resp)
        raise RuntimeError(f"No text content (status={getattr(resp,'status',None)}). Raw: {raw[:4000]}")

    frames = json.loads(text)  # should be a JSON array per your prompt



    # Parse JSON (strip fences if any)
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
    start, end = raw.find("["), raw.rfind("]")
    parsed_text = raw[start:end + 1] if start != -1 and end != -1 and end > start else raw

    frames = json.loads(parsed_text)
    if not isinstance(frames, list) or len(frames) != n_frames:
        raise ValueError("Model did not return the expected JSON array length.")
    for i, f in enumerate(frames, start=1):
        if f.get("frame_id") != i:
            raise ValueError("Frame IDs must be 1..N in order.")
        for key in ("caption", "narration", "prompt", "camera_direction", "style_tag"):
            if key not in f or not isinstance(f[key], str) or not f[key].strip():
                raise ValueError(f"Missing/invalid field '{key}' in frame {i}.")
    return frames


def _postprocess_frames_metadata_only(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Only adjusts metadata to avoid immediate repeats of camera/style.
    DOES NOT touch 'prompt' content (keeps GPT-5 text verbatim).
    """
    for i in range(1, len(frames)):
        if frames[i]["camera_direction"] == frames[i - 1]["camera_direction"]:
            for cand in ALLOWED_CAMERA_DIRECTIONS:
                if cand != frames[i - 1]["camera_direction"]:
                    frames[i]["camera_direction"] = cand
                    break
        if frames[i]["style_tag"] == frames[i - 1]["style_tag"]:
            for cand in ALLOWED_STYLE_TAGS:
                if cand != frames[i - 1]["style_tag"]:
                    frames[i]["style_tag"] = cand
                    break
    return frames


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
    frames = _llm_generate_frames_inline(args.premise, args.n_frames)
    frames = _postprocess_frames_metadata_only(frames)

    # Persist plan.json (for inspection only; generation always comes from GPT-5 next runs as well)
    plan_path = out_dir / "plan.json"
    plan = {
        "premise": args.premise,
        "beats": [
            {
                "id": f["frame_id"],
                "beat": f["caption"],
                "narration": f.get("narration", ""),
                "frame_prompt": f["prompt"],
                "camera_direction": f["camera_direction"],
                "style_tag": f["style_tag"],
            }
            for f in frames
        ],
        "llm": {"model": "gpt-5", "provider": "openai"},
        "note": "Prompts used verbatim from GPT-5; no local rewriting.",
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    logger.info("Saved story plan to %s", plan_path)

    # Build Shot-like objects for the pipeline from frames (prompts are verbatim from GPT-5)
    shots = [
        type(
            "Shot",
            (),
            {
                "frame_id": f["frame_id"],
                "caption": f["caption"],
                "prompt": f["prompt"],  # <- GPT-5 text, not altered
                "camera_direction": f.get("camera_direction", "mid-shot"),
                "style_tag": f.get("style_tag", "painterly realism"),
            },
        )
        for f in frames
    ]

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
