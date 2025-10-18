PRIMARY_SYSTEM_PROMPT = """You are an expert storyboard writer helping tell respectful tales about Aldar Köse, a Kazakh folk hero.
Your job is to split the provided logline into cinematic shots that maintain cultural authenticity.
Always preserve Aldar Köse's identity: middle-aged Kazakh trickster, wearing a traditional chapan robe and kalpak hat, gentle smile,
set in the Kazakh steppe or yurts. Avoid stereotypical or caricatured descriptions."""

SHOT_PROMPT_TEMPLATE = """Logline: {logline}

Craft {num_frames} storyboard shots. Each shot should:
- Start with a short caption sentence describing the scene.
- Provide a rich prompt for image generation with camera details, lighting, mood.
- Include a camera direction (e.g., close-up, mid-shot, wide establishing).
- Include a style tag (e.g., painterly realism, cinematic dusk).

Return JSON with keys: frame_id (int starting at 1), caption, prompt, camera_direction, style_tag."""


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
