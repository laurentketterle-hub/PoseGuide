"""Score subject against target pose, with optional MediaPipe path (#34)."""

from __future__ import annotations

from pathlib import Path

from poseguide.data.loader import load_subject
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker


def score_subject_against_pose(
    pose_id: str, subject_path: Path, *, use_mediapipe: bool = False
) -> dict:
    """Score a subject JSON against a catalog pose. Falls back to toy (cosine) when MediaPipe unavailable."""
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")
    subject = load_subject(subject_path)
    result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
    result["subject_id"] = subject.get("id")
    result["source"] = str(subject_path)
    result["method"] = "mediapipe" if use_mediapipe else "toy_cosine"
    return result
