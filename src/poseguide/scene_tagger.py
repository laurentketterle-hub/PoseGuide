"""Scene tagger: CV heuristics for background tags.

Bounty #6 — [50 MRG] Scene tagger: simple CV/heuristics or CLIP-stub for background tags.
Matches scenes to pose recommendations based on visual tags.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

SCENE_TAGS = [
    "beach", "urban", "forest", "indoor", "studio", "garden",
    "mountain", "water", "night", "sunset", "snow", "desert"
]


class SceneTagger:
    """Tag a scene image with CV heuristics or CLIP."""

    def __init__(self, use_clip: bool = False):
        self.use_clip = use_clip and HAS_CLIP
        if self.use_clip:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _dominant_colors(image: Image.Image, n: int = 3) -> list[tuple[int, int, int]]:
        small = image.resize((50, 50)).convert("RGB")
        arr = np.array(small)
        pixels = arr.reshape(-1, 3)
        buckets = {}
        for p in pixels:
            key = tuple((int(c) // 32) * 32 for c in p)
            buckets[key] = buckets.get(key, 0) + 1
        return [k for k, _ in sorted(buckets.items(), key=lambda x: -x[1])[:n]]

    def tag_heuristics(self, image_path: Path | str) -> dict:
        """Tag using simple color heuristics."""
        img = Image.open(image_path).convert("RGB")
        colors = self._dominant_colors(img)
        arr = np.array(img.resize((100, 100)))
        r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()

        tags = []
        if b > r + 20 and b > g + 20:
            tags.extend(["water", "beach" if r > 150 else "indoor"])
        elif g > r + 15 and g > b + 15:
            tags.extend(["forest", "garden"])
        elif r > g + 15 and r > b + 15 and r < 180:
            tags.append("urban")
        elif all(c[0] > 200 and c[1] > 200 and c[2] > 200 for c in colors[:2]):
            tags.append("studio")
        if sum((r, g, b)) / 3 < 80:
            tags.append("night" if b < 60 else "mountain")

        return {"tags": list(dict.fromkeys(tags))[:3], "confidence": min(0.85, len(tags) * 0.3),
                "method": "cv_heuristics", "dominant_colors": [[int(c) for c in col] for col in colors]}

    def tag(self, image_path: Path | str) -> dict:
        """Tag a scene image."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Scene image not found: {path}")
        if self.use_clip:
            return self._tag_clip(path)
        return self.tag_heuristics(path)

    def _tag_clip(self, image_path: Path) -> dict:
        img = Image.open(image_path)
        inputs = self.clip_processor(text=SCENE_TAGS, images=img, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image[0].detach().numpy()
        probs = np.exp(logits) / np.sum(np.exp(logits))
        top_idx = np.argsort(-probs)[:3]
        return {"tags": [SCENE_TAGS[i] for i in top_idx],
                "confidence": float(probs[top_idx[0]]), "method": "clip"}


__all__ = ["SceneTagger", "SCENE_TAGS"]
