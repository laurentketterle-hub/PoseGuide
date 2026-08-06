"""Scene tagger: heuristic CV-based background tag extraction.

Given a background image (or text description), produces scene tags used
by the ranker. Stub/heuristic OK; documented upgrade path to CLIP below.

Upgrade path to CLIP
--------------------
For production-quality tagging, replace _heuristic_tags_from_image with
a CLIP-based zero-shot classifier:

1. Install: pip install open-clip-torch pillow
2. Load: model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
3. Tokenize candidate tags
4. Score: image_features = model.encode_image(preprocess(pil_image).unsqueeze(0))
5. Return: tags = [candidates[i] for i in top_k_indices]

The SceneTagger class is designed to accept a callable backend, making
the swap from heuristic to CLIP a one-line change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Tag vocabulary (must stay in sync with pose catalog tags in data/poses/*.json)
# ---------------------------------------------------------------------------
_TAG_VOCABULARY: list[str] = sorted(
    {
        "beach", "outdoor", "indoor", "urban", "studio", "nature",
        "portrait", "casual", "golden_hour", "daylight", "night",
        "window", "cafe", "office", "home", "garden", "forest",
        "mountain", "sports", "street", "market", "library",
        "rooftop", "subway", "yoga", "festival", "harbor",
        "balcony", "loft", "nook", "alley", "field", "trail",
        "pier", "crossing", "wall", "path", "lobby", "morning",
        "sunset", "warm", "cool", "natural_light",
    }
)


def _colour_heuristics(pixels):
    if not pixels:
        return {}

    n = len(pixels)
    avg_r = sum(p[0] for p in pixels) / n
    avg_g = sum(p[1] for p in pixels) / n
    avg_b = sum(p[2] for p in pixels) / n
    luminance = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b

    green_score = max(0.0, (avg_g - max(avg_r, avg_b)) / 255.0)
    blue_score = max(0.0, (avg_b - max(avg_r, avg_g)) / 255.0)
    warm_score = max(0.0, (avg_r - avg_b) / 255.0)
    dark_score = max(0.0, (100.0 - luminance) / 100.0)
    bright_score = max(0.0, (luminance - 150.0) / 105.0)

    scores = {}

    if green_score > 0.05:
        scores["nature"] = min(1.0, green_score * 3)
        scores["outdoor"] = min(1.0, green_score * 2)
    if blue_score > 0.05:
        scores["outdoor"] = max(scores.get("outdoor", 0), min(1.0, blue_score * 2))
        scores["beach"] = max(scores.get("beach", 0), min(0.8, blue_score * 1.5))
    if warm_score > 0.05:
        scores["indoor"] = min(1.0, warm_score * 2)
        scores["warm"] = min(0.8, warm_score * 1.5)
        if warm_score > 0.15:
            scores["golden_hour"] = min(0.7, warm_score * 1.2)
            scores["sunset"] = min(0.6, warm_score * 1.0)
    if dark_score > 0.3:
        scores["night"] = min(0.9, dark_score * 1.5)
        scores["indoor"] = max(scores.get("indoor", 0), min(0.7, dark_score))
    if bright_score > 0.3:
        scores["daylight"] = min(0.9, bright_score)
        scores["outdoor"] = max(scores.get("outdoor", 0), min(0.8, bright_score * 1.2))
    if luminance < 80:
        scores["studio"] = min(0.7, (80 - luminance) / 80)

    return {k: min(1.0, max(0.0, v)) for k, v in scores.items()}


def _heuristic_tags_from_image(image_path, top_k=5):
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Pillow is required for image-based scene tagging. "
            "Install with: pip install Pillow"
        )

    img = Image.open(image_path).convert("RGB")
    img = img.resize((64, 64), Image.NEAREST)
    pixels = list(img.getdata())

    scores = _colour_heuristics(pixels)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    tags = [tag for tag, score in ranked if score > 0.15][:top_k]

    if not tags:
        tags = ["portrait", "casual"]
    if "portrait" not in tags:
        tags.append("portrait")
    if "casual" not in tags and len(tags) < top_k:
        tags.append("casual")

    return tags[:max(1, top_k)]


def _heuristic_tags_from_text(text):
    text_lower = text.lower()
    matched = []

    for tag in _TAG_VOCABULARY:
        tag_lower = tag.replace("_", " ")
        if tag_lower in text_lower or tag in text_lower:
            matched.append((tag, 1.0))
            continue
        first_word = text_lower.split()[0] if text_lower else ""
        if text_lower.startswith(tag_lower) or tag_lower.startswith(first_word):
            matched.append((tag, 0.6))

    matched.sort(key=lambda x: x[1], reverse=True)
    tags = [t for t, _ in matched[:8]]

    if not tags:
        if any(w in text_lower for w in ("beach", "sea", "ocean", "sand", "coast")):
            tags = ["beach", "outdoor", "daylight", "portrait"]
        elif any(w in text_lower for w in ("city", "urban", "street", "building")):
            tags = ["urban", "outdoor", "street", "portrait"]
        elif any(w in text_lower for w in ("studio", "backdrop", "plain")):
            tags = ["studio", "indoor", "portrait"]
        elif any(w in text_lower for w in ("cafe", "coffee", "restaurant")):
            tags = ["cafe", "indoor", "portrait", "casual"]
        elif any(w in text_lower for w in ("garden", "park", "flower")):
            tags = ["garden", "outdoor", "nature", "portrait"]
        elif any(w in text_lower for w in ("forest", "wood", "tree")):
            tags = ["forest", "outdoor", "nature", "portrait"]
        elif any(w in text_lower for w in ("night", "dark", "neon")):
            tags = ["night", "urban", "portrait"]
        elif any(w in text_lower for w in ("indoor", "room", "house", "home")):
            tags = ["indoor", "home", "portrait", "casual"]
        else:
            tags = ["portrait", "casual"]

    return tags[:5]


def tag_from_image(image_path, top_k=5):
    return _heuristic_tags_from_image(image_path, top_k=top_k)


def tag_from_text(description):
    return _heuristic_tags_from_text(description)


def tag_scene(image_path=None, description=None, top_k=5):
    if image_path is not None and image_path.exists():
        tags = _heuristic_tags_from_image(image_path, top_k=top_k)
        if description:
            text_tags = _heuristic_tags_from_text(description)
            seen = set(tags)
            for t in text_tags:
                if t not in seen and len(tags) < top_k + 2:
                    tags.append(t)
                    seen.add(t)
        return tags[:top_k]
    if description:
        return _heuristic_tags_from_text(description)
    return ["portrait", "casual"]


class SceneTagger:
    def __init__(self, backend=None):
        self._backend = backend or _heuristic_tags_from_image

    def tag(self, image_path, top_k=5):
        try:
            return self._backend(image_path)[:top_k]
        except Exception:
            return ["portrait", "casual"]

    @staticmethod
    def from_text(description):
        return _heuristic_tags_from_text(description)
