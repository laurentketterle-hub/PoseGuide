"""Vision utilities: scene tagging, image analysis, and CLIP upgrade path."""
from poseguide.vision.scene_tagger import SceneTagger, tag_scene, tag_from_image, tag_from_text

__all__ = ["SceneTagger", "tag_scene", "tag_from_image", "tag_from_text"]
