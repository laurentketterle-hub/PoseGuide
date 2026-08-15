"""Optional MediaPipe subject-scoring path (issue #34).

Adds a *vision* path that scores a photograph against a target pose by
extracting the subject skeleton with MediaPipe and matching it with the offline
ranker. The toy path remains the default offline behaviour: vision is opt-in,
and every vision entry point degrades to a toy baseline when the optional
``vision`` extra is unavailable instead of crashing.

Dependency injection keeps the path testable: pass a ``detector`` callable to
``score_subject_image`` to exercise the vision path without MediaPipe installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from poseguide.data.loader import joints_to_vector
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker


class VisionUnavailableError(RuntimeError):
    """Raised when the optional ``vision`` extra (MediaPipe/OpenCV) is missing."""


class LandmarkDetector(Protocol):
    """Anything that turns an image path into a flat MediaPipe landmark list."""

    def __call__(self, image_path: Path) -> Sequence | None: ...


def score_subject_image(
    pose_id: str,
    image_path: str | Path,
    *,
    detector: LandmarkDetector | None = None,
    allow_fallback: bool = True,
) -> dict:
    """Score a photograph against a target pose via the optional vision path.

    The subject skeleton is extracted from ``image_path`` with MediaPipe and
    matched against ``pose_id`` using the offline ranker's joint metrics. When
    the ``vision`` extra is unavailable (or no subject is detected) the call
    falls back to the toy baseline and reports ``vision_available: False`` —
    issue #34 keeps the toy path as the default offline behaviour.

    Parameters
    ----------
    pose_id:
        Target pose id (catalog key or name).
    image_path:
        Source photograph. Must exist on disk.
    detector:
        Injectable landmark detector. Defaults to the MediaPipe-backed detector
        from :mod:`poseguide.data.extract` (requires the ``vision`` extra).
    allow_fallback:
        When ``True`` (default) a missing vision extra yields a toy-baseline
        result. Set ``False`` to raise :class:`VisionUnavailableError` instead.

    Raises
    ------
    KeyError
        If ``pose_id`` is unknown.
    FileNotFoundError
        If ``image_path`` does not exist.
    VisionUnavailableError
        If ``allow_fallback`` is ``False`` and the vision path cannot run.
    """
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from poseguide.data.extract import default_mediapipe_detector, extract_pose

        detector = detector or default_mediapipe_detector()
        payload = extract_pose(path, detector=detector)
    except RuntimeError as exc:
        if not allow_fallback:
            raise VisionUnavailableError(str(exc)) from exc
        return _toy_baseline(pose, reason=str(exc))

    subject_vector = joints_to_vector(payload.get("joints") or {})
    result = ToyPoseRanker().score_match(pose, subject_vector)
    result["path"] = "vision"
    result["vision_available"] = True
    result["subject_id"] = payload.get("id")
    result["source"] = "mediapipe"
    return result


def _toy_baseline(pose: dict, *, reason: str) -> dict:
    """Offline fallback result used when the vision path is unavailable."""
    return {
        "pose_id": pose.get("id"),
        "name": pose.get("name"),
        "path": "toy",
        "vision_available": False,
        "fallback_reason": reason,
        "similarity": None,
        "confidence": None,
        "cues": ["Vision extra unavailable — install 'poseguide[vision]' for photo scoring"],
        "pass": None,
    }
