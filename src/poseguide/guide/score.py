"""Scoring functions for subject-vs-pose matching."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from poseguide.data.loader import load_subject
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker


def _try_mediapipe_score(pose, subject: dict) -> dict | None:
    """Attempt MediaPipe-based scoring. Returns None if unavailable."""
    try:
        from poseguide.data.extract import JOINT_KEYS
        from poseguide.eval.metrics import cosine_similarity

        joint_vector = subject.get("joint_vector", [])
        if not joint_vector or len(joint_vector) != len(JOINT_KEYS):
            return None

        pose_vector = pose.get("joint_vector", [])
        if not pose_vector:
            return None

        sim = cosine_similarity(joint_vector, pose_vector)
        return {
            "method": "mediapipe",
            "similarity": round(sim, 4),
            "joint_count": len(joint_vector),
        }
    except (ImportError, Exception):
        return None


def score_subject_against_pose(
    pose_id: str,
    subject_path: Path,
    *,
    use_mediapipe: bool = False,
) -> dict:
    """Score a subject against a pose.

    Args:
        pose_id: The catalog pose ID to score against.
        subject_path: Path to a subject JSON file.
        use_mediapipe: If True, attempt MediaPipe-based scoring with
            fallback to the toy/default ranker on failure.
    """
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")

    subject = load_subject(subject_path)

    # Try MediaPipe scoring if requested
    mp_result = None
    if use_mediapipe:
        mp_result = _try_mediapipe_score(pose, subject)

    # Default: toy ranker (always works)
    result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
    result["subject_id"] = subject.get("id")
    result["source"] = str(subject_path)

    # Attach MediaPipe result if available
    if mp_result:
        result["mediapipe"] = mp_result
        result["method"] = "mediapipe_with_fallback"

    return result
