import json
import os
import random
from typing import List

from ..utils.schema import Shot
from .prompts import (
    PRIMARY_SYSTEM_PROMPT,
    OUTLINE_PROMPT_TEMPLATE,
    SHOT_PROMPT_TEMPLATE,
    CHARACTER_BIBLE,
    FALLBACK_CAMERA_DIRECTIONS,
    FALLBACK_STYLE_TAGS,
    enforce_diversity,
)


def _call_openai(logline: str, n_frames: int) -> List[Shot]:
    """
    Two-stage LLM planning:
    1) Create a compact outline from the logline.
    2) Generate id-aligned shots following the outline and identity rules.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Stage 1: Outline
        outline_prompt = OUTLINE_PROMPT_TEMPLATE.format(logline=logline, num_frames=n_frames)
        outline_resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                {"role": "user", "content": outline_prompt},
            ],
            max_output_tokens=800,
        )
        outline_text = outline_resp.output[0].content[0].text  # type: ignore[index]
        outline = json.loads(outline_text)

        # Stage 2: Shots
        shot_prompt = SHOT_PROMPT_TEMPLATE.format(
            logline=logline,
            outline_json=json.dumps(outline, ensure_ascii=False),
            num_frames=n_frames,
            aldar_desc=CHARACTER_BIBLE["canonical_description"],
            aldar_token=CHARACTER_BIBLE["aldar_token"],
            camera_directions=FALLBACK_CAMERA_DIRECTIONS,
            style_tags=FALLBACK_STYLE_TAGS,
        )
        shots_resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                {"role": "user", "content": shot_prompt},
            ],
            max_output_tokens=1400,
        )
        shots_text = shots_resp.output[0].content[0].text  # type: ignore[index]
        shots_payload = json.loads(shots_text)

        # Post-process for diversity and identity token injection
        shots_payload = enforce_diversity(shots_payload)

        # Convert to schema objects (ignore extra keys like new_element/continuity_note)
        shots: List[Shot] = []
        for s in shots_payload:
            shots.append(
                Shot(
                    frame_id=int(s["frame_id"]),
                    caption=s["caption"],
                    prompt=s["prompt"],
                    camera_direction=s["camera_direction"],
                    style_tag=s["style_tag"],
                )
            )
        return shots

    except Exception as exc:  # pragma: no cover - network path
        raise RuntimeError(f"OpenAI planning failed: {exc}") from exc


def _fallback_plan(logline: str, n_frames: int) -> List[Shot]:
    """
    Deterministic fallback that still provides variety and respects identity.
    Will produce the same plan for the same logline (seeded by logline hash).
    """
    random.seed(hash(logline) % (2**32))

    supporting_cast = ["greedy merchant", "caravan guard", "tea seller", "elder", "child", "camel handler"]
    locations = ["desert bazaar", "yurt interior", "steppe dunes", "oasis edge", "caravan trail", "dune ridge at dusk"]
    props = ["coin pouch", "weighing scale", "ledger book", "tea bowl", "camel rope", "silk bolt"]

    base_identity = "Aldar Köse [AldarMain], middle-aged Kazakh trickster in a chapan robe and kalpak hat, gentle smile"

    beat_templates = [
        "arrives at the {loc}, meets a {char}, spots a {prop}",
        "tests the {char} with a riddle at the {loc}, hiding a {prop}",
        "notices a trick with the {prop} and sets a counter-plan near the {loc}",
        "turns the tables on the {char} at the {loc}",
        "reveals the twist using the {prop} at the {loc}",
        "rides off from the {loc}, leaving the {char} stunned",
        "shares tea and a lesson learned at the {loc}",
        "retells the tale to children near the {loc}",
    ]

    shots: List[Shot] = []
    for idx in range(n_frames):
        frame_id = idx + 1
        camera_direction = FALLBACK_CAMERA_DIRECTIONS[idx % len(FALLBACK_CAMERA_DIRECTIONS)]
        style_tag = FALLBACK_STYLE_TAGS[idx % len(FALLBACK_STYLE_TAGS)]

        loc = random.choice(locations)
        char = random.choice(supporting_cast)
        prop = random.choice(props)
        action = beat_templates[idx % len(beat_templates)].format(loc=loc, char=char, prop=prop)

        caption = f"Aldar Köse {action}."
        prompt = (
            f"{base_identity}. Setting: {loc}. Scene: {action}. "
            f"Camera: {camera_direction}. Visual style: {style_tag}. "
            f"Natural lighting, respectful depiction."
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


def plan_shots(logline: str, n_frames: int = 3, require_openai: bool = True) -> List[Shot]:
    # if n_frames < 6 or n_frames > 10:
    #     raise ValueError("n_frames must be between 6 and 10")
    return _call_openai(logline, n_frames)
