from __future__ import annotations

from pathlib import Path

from poseguide.data.loader import load_subject
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker


def score_subject_against_pose(pose_id: str, subject_path: Path) -> dict:
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")
    subject = load_subject(subject_path)
    result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
    result["subject_id"] = subject.get("id")
    result["source"] = str(subject_path)
    return result


def score_subject(
    pose_id: str,
    subject_path: Path,
    *,
    vision: bool = False,
    allow_fallback: bool = True,
) -> dict:
    """Score a subject against a target pose.

    The offline toy path is the default (``vision=False``): it scores a
    pre-extracted subject JSON with no vision dependency. Set ``vision=True`` to
    score a photograph via the optional MediaPipe path, which degrades to the
    toy baseline when the ``vision`` extra is unavailable (issue #34).
    """
    if vision:
        from poseguide.models.vision import score_subject_image

        return score_subject_image(pose_id, subject_path, allow_fallback=allow_fallback)
    return score_subject_against_pose(pose_id, subject_path)
