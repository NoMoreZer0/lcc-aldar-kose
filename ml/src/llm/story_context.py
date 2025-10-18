from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..utils.schema import Shot

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


def _request_story_frames(premise: str, n_frames: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)

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
        reasoning={"effort": "minimal"},
        max_output_tokens=4096,
        tool_choice="none",
    )

    text = getattr(resp, "output_text", None)
    if not text:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []) or []:
                    value = getattr(content, "text", None)
                    if value:
                        parts.append(value)
        text = "\n".join(parts).strip()

    if not text:
        raw = resp.model_dump_json() if hasattr(resp, "model_dump_json") else str(resp)
        raise RuntimeError(f"No text content (status={getattr(resp, 'status', None)}). Raw: {raw[:4000]}")

    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
    start, end = raw.find("["), raw.rfind("]")
    parsed_text = raw[start : end + 1] if start != -1 and end != -1 and end > start else raw

    frames = json.loads(parsed_text)
    if not isinstance(frames, list) or len(frames) != n_frames:
        raise ValueError("Model did not return the expected JSON array length.")

    for i, frame in enumerate(frames, start=1):
        if frame.get("frame_id") != i:
            raise ValueError("Frame IDs must be 1..N in order.")
        for key in ("caption", "narration", "prompt", "camera_direction", "style_tag"):
            value = frame.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing/invalid field '{key}' in frame {i}.")

    return frames


def _postprocess_frames_metadata_only(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index in range(1, len(frames)):
        if frames[index]["camera_direction"] == frames[index - 1]["camera_direction"]:
            for candidate in ALLOWED_CAMERA_DIRECTIONS:
                if candidate != frames[index - 1]["camera_direction"]:
                    frames[index]["camera_direction"] = candidate
                    break
        if frames[index]["style_tag"] == frames[index - 1]["style_tag"]:
            for candidate in ALLOWED_STYLE_TAGS:
                if candidate != frames[index - 1]["style_tag"]:
                    frames[index]["style_tag"] = candidate
                    break
    return frames


def generate_story_frames(premise: str, n_frames: int) -> List[Dict[str, Any]]:
    frames = _request_story_frames(premise, n_frames)
    return _postprocess_frames_metadata_only(frames)


def frames_to_shots(frames: List[Dict[str, Any]]) -> List[Shot]:
    shots: List[Shot] = []
    for frame in frames:
        shots.append(
            Shot(
                frame_id=int(frame["frame_id"]),
                caption=frame["caption"],
                prompt=frame["prompt"],
                camera_direction=frame.get("camera_direction", "mid-shot"),
                style_tag=frame.get("style_tag", "painterly realism"),
            )
        )
    return shots


def build_plan_payload(premise: str, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "premise": premise,
        "beats": [
            {
                "id": frame["frame_id"],
                "beat": frame["caption"],
                "narration": frame.get("narration", ""),
                "frame_prompt": frame["prompt"],
                "camera_direction": frame.get("camera_direction", ""),
                "style_tag": frame.get("style_tag", ""),
            }
            for frame in frames
        ],
        "llm": {"model": "gpt-5", "provider": "openai"},
        "note": "Prompts used verbatim from GPT-5; no local rewriting.",
    }


__all__ = [
    "ALLOWED_CAMERA_DIRECTIONS",
    "ALLOWED_STYLE_TAGS",
    "generate_story_frames",
    "frames_to_shots",
    "build_plan_payload",
]
