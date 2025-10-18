import json
import os
import random
from typing import List

from ..utils.schema import Shot
from .prompts import (
    FALLBACK_CAMERA_DIRECTIONS,
    FALLBACK_STYLE_TAGS,
    PRIMARY_SYSTEM_PROMPT,
    SHOT_PROMPT_TEMPLATE,
)


def _call_openai(logline: str, n_frames: int) -> List[Shot]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SHOT_PROMPT_TEMPLATE.format(logline=logline, num_frames=n_frames),
                },
            ],
            temperature=0.6,
            max_output_tokens=1200,
        )
        message = response.output[0].content[0].text  # type: ignore[index]
        shots_payload = json.loads(message)
        return [Shot(**shot) for shot in shots_payload]
    except Exception as exc:  # pragma: no cover - network path
        raise RuntimeError(f"OpenAI planning failed: {exc}") from exc


def _fallback_plan(logline: str, n_frames: int) -> List[Shot]:
    random.seed(hash(logline) % (2**32))
    base_prompt = (
        "Aldar Köse, the clever Kazakh folk hero, wearing his embroidered chapan robe and kalpak hat, "
        "with a warm, knowing smile. "
        "Setting: the vast Kazakh steppe with yurts, herds, and traditional textiles."
    )
    action_clauses = [
        "outwits a pompous noble by swapping gifts",
        "shares tea with travelers inside a cozy yurt",
        "rides across the steppe at golden hour",
        "reveals a hidden stash of coins with a grin",
        "comforts a villager with a clever plan",
        "faces a greedy merchant under starry skies",
        "listens to musicians by a campfire",
        "inspires children with tales of wisdom",
        "meets a mysterious messenger by the riverside",
        "celebrates a victory at dawn",
    ]

    shots: List[Shot] = []
    for idx in range(n_frames):
        frame_id = idx + 1
        action = action_clauses[idx % len(action_clauses)]
        camera_direction = FALLBACK_CAMERA_DIRECTIONS[idx % len(FALLBACK_CAMERA_DIRECTIONS)]
        style_tag = FALLBACK_STYLE_TAGS[idx % len(FALLBACK_STYLE_TAGS)]
        caption = f"Aldar Köse {action}."
        prompt = (
            f"{base_prompt} Scene: {action}. Camera: {camera_direction}. "
            f"Visual style: {style_tag}. Natural lighting, respectful depiction."
        )
        shots.append(
            Shot(
                frame_id=frame_id,
                caption=caption,
                prompt=prompt,
                camera_direction=camera_direction,
                style_tag=style_tag,
            )
        )
    return shots


def plan_shots(logline: str, n_frames: int = 8) -> List[Shot]:
    if n_frames < 6 or n_frames > 10:
        raise ValueError("n_frames must be between 6 and 10")

    try:
        return _call_openai(logline, n_frames)
    except RuntimeError:
        return _fallback_plan(logline, n_frames)
