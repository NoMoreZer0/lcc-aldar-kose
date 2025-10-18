PRIMARY_SYSTEM_PROMPT = """You are an expert storyboard writer for respectful tales about Aldar Köse, a Kazakh folk hero.
Your job is to turn a short logline into a cohesive mini-story told in N cinematic shots.
Strong requirements:
- Cultural authenticity and respect. Avoid stereotypes/caricatures.
- Keep Aldar Köse’s identity CONSTANT across all frames:
  • Middle-aged Kazakh trickster • traditional chapan robe • kalpak hat • gentle, knowing smile
  • moves through Kazakh steppe, yurts, bazaars, caravans
- Each frame must push the story forward (setup → rising action → twist/climax → resolution).
- Each NEW frame introduces at least ONE meaningful change: a new supporting character, a new place,
  a new prop, a time-of-day shift, or a clear plot turn. Do not repeat static scenes with only wording changes.
- Prompts must be concise (aim ≤ 65 tokens per encoder) and free of tokens like <|endoftext|>.
- Safety: no violence glorification; respectful portrayals; no ethnic caricature.

Return only valid JSON as instructed — no Markdown, no commentary."""

# 1) A quick outline pass creates beats that the shot-writer must follow.
OUTLINE_PROMPT_TEMPLATE = """Logline: {logline}
############################################################
Target frames: {num_frames}
############################################################
Create a tight outline for a 1-scene short story starring Aldar Köse.
Return JSON:
{{
  "title": "...",
  "theme": "...",
  "beats": [
     {{"id": 1, "beat": "setup", "goal": "...", "conflict": "..."}},
     {{"id": 2, "beat": "rising_action", "turn": "..."}},
     ...
     {{"id": {num_frames}, "beat": "resolution", "outcome": "..."}}
  ],
  "supporting_cast_bank": ["greedy merchant", "caravan guard", "tea seller", "elder", "child", "camel handler"],
  "location_bank": ["desert bazaar", "yurt interior", "steppe dunes", "oasis edge", "caravan trail", "dune ridge at dusk"],
  "prop_bank": ["coin pouch", "weighing scale", "ledger book", "tea bowl", "camel rope", "silk bolt"]
}}"""

# 2) Canonical identity tokens you can inject into each frame’s prompt.
CHARACTER_BIBLE = {
    "aldar_token": "[AldarMain]",  # include this literal token in every prompt for identity locking
    "canonical_description": "Aldar Köse [AldarMain], middle-aged Kazakh trickster in a chapan robe and kalpak hat, gentle smile"
}

SHOT_PROMPT_TEMPLATE = """You are given:
- Logline: {logline}
- Outline JSON: {outline_json}
- Total frames: {num_frames}

Write exactly {num_frames} shots that follow the outline beats (id-aligned).
Hard rules:
- Every shot MUST include Aldar as: {aldar_desc}.
- Each shot MUST introduce at least ONE new element compared to previous shots:
  (supporting character | location | prop | time-of-day | plot turn).
- Vary camera_direction and style_tag across frames (no immediate repeats).
- Keep image prompt ≤ ~65 tokens; include camera, lens, composition, lighting, mood; avoid filler and special tokens.

Return a JSON array of length {num_frames}. Each item has:
{{
  "frame_id": <int 1..{num_frames}>,
  "caption": "<1 concise sentence>",
  "prompt": "<rich but compact image prompt that includes {aldar_token} and the canonical description>",
  "camera_direction": "<from allowed set of camera directions>",
  "style_tag": "<from allowed set> of style tags",
  "new_element": "<what newly appears/changes>",
  "continuity_note": "<what to carry over from the previous frame>",
  "negatives": "low detail, extra fingers, distorted faces, stereotype imagery"
}}

############################################################

Allowed camera directions: {camera_directions}

############################################################

Allowed style tags: {style_tags}
"""

FALLBACK_CAMERA_DIRECTIONS = [
    "wide establishing shot",
    "mid-shot",
    "close-up",
    "over-the-shoulder",
    "dynamic three-quarter angle",
    "low-angle hero shot",
    "bird's-eye view",
    "tracking side profile",
]

FALLBACK_STYLE_TAGS = [
    "painterly realism",
    "golden hour cinematic",
    "moonlit mystery",
    "crisp morning light",
    "warm hearth glow",
    "wind-swept steppe",
]

# --- Helpers you can use after LLM returns JSON ---

def enforce_diversity(shots):
    """
    Post-process in Python to guarantee:
    - no immediate repetition of camera_direction or style_tag
    - each frame has 'new_element' that wasn't used before
    """
    seen_elements = set()
    for i, s in enumerate(shots):
        # Ensure new_element is truly new
        ne = s.get("new_element", "").strip().lower()
        if not ne or ne in seen_elements:
            # Fallback: synthesize a new element dimension
            s["new_element"] = f"prop: coin pouch {i}"
        seen_elements.add(s["new_element"].strip().lower())

        # Avoid adjacent duplicates in camera/style
        if i > 0:
            if s["camera_direction"] == shots[i-1]["camera_direction"]:
                # rotate to a different fallback
                for cand in FALLBACK_CAMERA_DIRECTIONS:
                    if cand != shots[i-1]["camera_direction"]:
                        s["camera_direction"] = cand
                        break
            if s["style_tag"] == shots[i-1]["style_tag"]:
                for cand in FALLBACK_STYLE_TAGS:
                    if cand != shots[i-1]["style_tag"]:
                        s["style_tag"] = cand
                        break

        # Inject identity token if missing
        if "[AldarMain]" not in s.get("prompt", ""):
            s["prompt"] = f"{s['prompt']} [AldarMain]"
    return shots
