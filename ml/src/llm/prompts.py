PRIMARY_SYSTEM_PROMPT = """You are an expert storyboard writer for respectful tales about Aldar Köse, a Kazakh folk hero.
Your job is to turn a short logline into a cohesive mini-story told in N cinematic shots, all staged within ONE consistent Kazakh setting.
Strong requirements:
- Cultural authenticity and respect. Avoid stereotypes/caricatures.
- Keep Aldar Köse’s identity CONSTANT across all frames:
  • Middle-aged Kazakh trickster • traditional chapan robe • kalpak hat • gentle, knowing smile
  • rooted in Kazakh green steppe heritage, folk textiles, and oral storytelling traditions
- Keep the ENVIRONMENT CONSTANT across the entire storyboard:
  • same green steppe encampment with a central yurt, glowing campfire, tethered horse, low tea table, embroidered carpets
  • static props (yurt, campfire, horse tether, tea set, saddle bags, distant mountains) stay present every frame
  • Only dynamic elements (people, animals, gestures, wind, sun position, handheld props) may move or change emphasis
- Each frame must push the story forward (setup → rising action → twist/climax → resolution).
- Each NEW frame introduces at least ONE meaningful dynamic change (interaction, expression, prop usage, animal behavior,
  storytelling beat) while the environment and static objects stay anchored.
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
  "environment_blueprint": "Single Kazakh green steppe encampment at golden hour with a central yurt, steady campfire, tethered horse, low tea table, embroidered carpets, distant Tien Shan mountains. This environment and these static props stay visible every frame.",
  "beats": [
     {{"id": 1, "beat": "setup", "goal": "...", "conflict": "..."}},
     {{"id": 2, "beat": "rising_action", "turn": "..."}},
     ...
     {{"id": {num_frames}, "beat": "resolution", "outcome": "..."}}
  ],
  "supporting_cast_bank": ["greedy merchant", "shepherd", "wise elder", "musician", "child apprentice", "nomad rider"],
  "location_bank": ["single green steppe encampment (unchanged across frames)"],
  "prop_bank": ["dombra lute", "embroidered saddlebag", "tea bowl", "horse rope", "story scroll", "coin pouch"]
}}"""

# 2) Canonical identity tokens you can inject into each frame’s prompt.
CHARACTER_BIBLE = {
    "aldar_token": "[AldarMain]",  # include this literal token in every prompt for identity locking
    "canonical_description": "Aldar Köse [AldarMain], middle-aged Kazakh trickster in an indigo embroidered chapan robe and white kalpak hat, gentle knowing smile, carrying a carved wooden staff"
}

SHOT_PROMPT_TEMPLATE = """You are given:
- Logline: {logline}
- Outline JSON: {outline_json}
- Total frames: {num_frames}

Write exactly {num_frames} shots that follow the outline beats (id-aligned).
Hard rules:
- Every shot MUST include Aldar as: {aldar_desc}.
- The environment NEVER changes: keep the same Kazakh green steppe encampment, same yurt, same campfire, same tethered horse, same horizon line.
- Static props stay consistent each frame; do not remove or relocate them.
- Each shot MUST introduce at least ONE new dynamic element compared to previous shots:
  (supporting character action, emotional beat, gesture, wind movement, animal behavior, use of a prop, storytelling flourish) while the static set stays fixed.
- Vary camera_direction and style_tag across frames (no immediate repeats).
- Keep image prompt ≤ ~65 tokens; include camera, lens, composition, lighting, mood; avoid filler and special tokens.
- Highlight Kazakh folk culture details, textiles, motifs, instruments, and oral storytelling energy.

Return a JSON array of length {num_frames}. Each item has:
{{
  "frame_id": <int 1..{num_frames}>,
  "caption": "<1 concise sentence>",
  "prompt": "<rich but compact image prompt that includes {aldar_token} and the canonical description>",
  "camera_direction": "<from allowed set of camera directions>",
  "style_tag": "<from allowed set> of style tags",
  "new_element": "<dynamic action/prop/interaction that changes while environment stays the same>",
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
    "painterly realism with Kazakh folk motifs",
    "golden hour cinematic over the Kazakh green steppe",
    "moonlit mystery above the yurt encampment",
    "crisp dawn light on Kazakh grasslands",
    "warm hearth glow with embroidered textiles",
    "wind-swept green steppe folklore illustration",
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
        if not ne or ne in seen_elements or "location" in ne or "environment" in ne:
            # Fallback: synthesize a dynamic interaction beat
            s["new_element"] = f"dynamic interaction: Aldar shares proverb {i}"
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
