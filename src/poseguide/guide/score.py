from __future__ import annotations

from pathlib import Path

import numpy as np

from poseguide.data.loader import joints_to_vector, load_subject
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class VisionScoreUnavailableError(RuntimeError):
    """Raised when the optional MediaPipe score path cannot run.

    The ``vision`` extra (opencv + mediapipe) is optional. When it is missing,
    or no person is detected in the image, callers should fall back to the
    offline toy path (:func:`score_subject_against_pose` with a JSON subject).
    """


def _resolve_pose(pose_id: str) -> dict:
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")
    return pose


def _score_match(pose: dict, joint_vector: np.ndarray) -> dict:
    return ToyPoseRanker().score_match(pose, joint_vector)


def score_subject_against_pose(pose_id: str, subject_path: Path) -> dict:
    """Score an offline JSON subject against a pose (toy path).

    This is the default, dependency-free path: it runs without the optional
    ``vision`` extra and works fully offline.
    """
    pose = _resolve_pose(pose_id)
    subject = load_subject(subject_path)
    result = _score_match(pose, subject["joint_vector"])
    result["subject_id"] = subject.get("id")
    result["source"] = str(subject_path)
    result["engine"] = "toy"
    return result


def score_image_against_pose(
    pose_id: str,
    image_path: Path,
    *,
    detector=None,
) -> dict:
    """Score a subject photo against a pose via the optional MediaPipe path.

    MediaPipe (and OpenCV) are imported lazily and are only required for this
    path. When the ``vision`` extra is missing or no person is found, a
    :class:`VisionScoreUnavailableError` is raised so callers can fall back to
    the offline toy path.
    """
    from poseguide.data.extract import extract_pose

    pose = _resolve_pose(pose_id)
    try:
        subject = extract_pose(image_path, detector=detector)
    except FileNotFoundError:
        raise
    except RuntimeError as exc:  # vision extra missing OR no person detected
        raise VisionScoreUnavailableError(str(exc)) from exc

    result = _score_match(pose, joints_to_vector(subject["joints"]))
    result["subject_id"] = subject.get("id")
    result["source"] = "mediapipe"
    result["engine"] = "mediapipe"
    return result


def score_subject(pose_id: str, subject_path: Path, *, engine: str = "auto") -> dict:
    """Score a subject against a pose, routing to the right path.

    ``engine`` selects the scoring path:

    - ``"auto"`` (default): image files use the optional MediaPipe path and
      JSON files use the offline toy path.
    - ``"toy"``: force the offline JSON path.
    - ``"mediapipe"``: force the MediaPipe image path.

    This keeps the toy path the default offline behaviour while making the
    MediaPipe path opt-in and optional.
    """
    engine = (engine or "auto").strip().lower()
    if engine == "toy":
        return score_subject_against_pose(pose_id, subject_path)
    if engine == "mediapipe":
        return score_image_against_pose(pose_id, subject_path)
    if engine == "auto":
        if Path(subject_path).suffix.lower() in _IMAGE_SUFFIXES:
            return score_image_against_pose(pose_id, subject_path)
        return score_subject_against_pose(pose_id, subject_path)
    raise ValueError(f"unknown engine {engine!r}; use 'auto', 'toy', or 'mediapipe'")
