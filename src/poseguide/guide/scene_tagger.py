"""Scene tagger: heuristic/CV stub for background tags (#6)."""
from __future__ import annotations

from pathlib import Path

PRESET_TAGS = {
    "beach": ["beach", "outdoor", "golden_hour", "portrait", "daylight"],
    "urban": ["urban", "wall", "street", "daylight", "casual"],
    "studio": ["studio", "indoor", "portrait", "business", "confident"],
    "forest": ["forest", "outdoor", "golden_hour", "romantic", "portrait"],
    "office": ["indoor", "business", "urban", "confident", "studio"],
    "mountain": ["mountain", "outdoor", "daylight", "adventure", "landscape"],
    "night_city": ["urban", "night", "cityscape", "portrait", "neon"],
    "cafe": ["cafe", "indoor", "warm-light", "casual", "portrait"],
    "park": ["park", "outdoor", "daylight", "nature", "romantic"],
    "home": ["home", "indoor", "soft_light", "casual", "portrait"],
}

def tag_scene(description: str | None = None, preset: str | None = None, image_path: Path | None = None) -> list[str]:
    """Produce scene tags from a description, preset name, or image path."""
    if preset and preset.lower() in PRESET_TAGS:
        return list(PRESET_TAGS[preset.lower()])
    if description:
        desc_lower = description.lower()
        tags = []
        for preset_name, preset_tags in PRESET_TAGS.items():
            if preset_name.replace("_", " ") in desc_lower or preset_name in desc_lower:
                tags = preset_tags
                break
        if not tags:
            tags = ["outdoor", "portrait", "daylight"]
        return tags
    if image_path:
        # Heuristic stub: use filename keywords
        stem = image_path.stem.lower()
        for kw, preset_tags in [("beach", PRESET_TAGS["beach"]), ("urban", PRESET_TAGS["urban"]),
                                 ("studio", PRESET_TAGS["studio"]), ("forest", PRESET_TAGS["forest"])]:
            if kw in stem:
                return preset_tags
        return ["outdoor", "portrait", "daylight"]
    return ["outdoor", "portrait", "daylight"]
