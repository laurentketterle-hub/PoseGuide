
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Canonical standing-skeleton joints understood by the pose engine.
JOINT_KEYS = (
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)

# Joints that must be present for a pose to be scoreable.
REQUIRED_JOINTS = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")


class SubjectPose(BaseModel):
    """A single subject's pose within a multi-subject composition."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, description="Subject role (e.g. 'bride', 'groom', 'partner_a')")
    position: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="3D offset [x, y, z] relative to composition center",
    )
    joints: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator("joints")
    @classmethod
    def _check_joints(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        missing = [key for key in REQUIRED_JOINTS if key not in value]
        if missing:
            raise ValueError(f"missing required joints: {', '.join(missing)}")
        for key, coords in value.items():
            if key not in JOINT_KEYS:
                raise ValueError(f"unknown joint key: {key!r}")
            if len(coords) < 2:
                raise ValueError(f"joint {key!r} needs at least [x, y] coordinates")
        return value

    @field_validator("position")
    @classmethod
    def _check_position(cls, value: list[float]) -> list[float]:
        if len(value) < 2:
            raise ValueError("position needs at least [x, y] coordinates")
        return value


class Pose(BaseModel):
    """Validated schema for a shipped pose template (``data/poses/*.json``)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    standing: bool = True
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    camera_cues: list[str] = Field(default_factory=list)
    joints: dict[str, list[float]] = Field(default_factory=dict)
    subjects: Optional[list[SubjectPose]] = Field(
        default=None, description="Multi-subject poses (2+ people)"
    )
    subject_count: int = Field(default=1, ge=1, le=20, description="Number of subjects in pose")
    scene_rankings: dict[str, float] = Field(
        default_factory=dict,
        description="Per-scene ranking scores (e.g. {'wedding_garden': 0.95, 'studio': 0.8})",
    )

    @field_validator("difficulty")
    @classmethod
    def _check_difficulty(cls, value: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if value not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}, got {value!r}")
        return value

    @field_validator("joints")
    @classmethod
    def _check_joints(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        # Single-subject poses must have required joints
        return value


class Scene(BaseModel):
    """Validated schema for a shipped scene template (``data/scenes/*.json``)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    expected_poses: list[str] = Field(default_factory=list)
    composition: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


def validate_pose(payload: dict) -> Pose:
    """Validate a single pose payload, raising ``ValidationError`` on bad data."""
    return Pose.model_validate(payload)


def validate_scene(payload: dict) -> Scene:
    """Validate a single scene payload, raising ``ValidationError`` on bad data."""
    return Scene.model_validate(payload)


def validate_pose_file(path: Path) -> Pose:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_pose(payload)


def validate_scene_file(path: Path) -> Scene:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_scene(payload)


def load_poses(paths: list[Path] | None = None) -> list[Pose]:
    """Validate and return every shipped pose (or the given ``paths``)."""
    from poseguide.data.loader import list_pose_files

    files = paths if paths is not None else list_pose_files()
    return [validate_pose_file(path) for path in files]


def load_scenes(paths: list[Path] | None = None) -> list[Scene]:
    """Validate and return every shipped scene (or the given ``paths``)."""
    from poseguide.data.loader import list_scene_files

    files = paths if paths is not None else list_scene_files()
    return [validate_scene_file(path) for path in files]


def rank_pose_for_scene(pose: Pose, scene: Scene) -> float:
    """Rank a pose's suitability for a given scene using tags and scene_rankings.
    
    Returns a score between 0.0 and 1.0.
    """
    # Direct scene ranking if available (multi-subject poses)
    if scene.id in pose.scene_rankings:
        return pose.scene_rankings[scene.id]
    
    # Tag-based ranking fallback
    pose_tags = set(pose.tags)
    scene_tags = set(scene.tags)
    overlap = pose_tags & scene_tags
    if overlap:
        return min(0.3 + 0.1 * len(overlap), 0.9)
    return 0.2
