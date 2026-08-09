from __future__ import annotations

from pathlib import Path

from poseguide.data.loader import load_subject
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.toy import ToyPoseRanker

# Image extensions that should trigger MediaPipe extraction
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def score_subject_against_pose(
    pose_id: str,
    subject_path: Path,
    *,
    use_mediapipe: bool = False,
) -> dict:
    """Score a subject (or image) against a catalog pose.

    Parameters
    ----------
    pose_id:
        Catalog pose ID.
    subject_path:
        Path to a subject JSON file (``data/samples/*.json``) or an image
        (``.jpg`` / ``.png`` / ``.webp`` ...).  When ``subject_path`` points to
        an image the function attempts MediaPipe extraction automatically;
        if the ``vision`` extra is not installed it falls back to a synthetic
        toy vector so the scoring path still works offline.
    use_mediapipe:
        When ``True`` and ``subject_path`` is a JSON file, MediaPipe is still
        *not* applied (JSON already carries joint data).  Set ``True`` only
        when you want to force the image→extract path even for JSON inputs
        (rare — typically only for testing the extraction pipeline).
    """
    pose = get_pose_by_id(pose_id)
    if pose is None:
        raise KeyError(f"unknown pose {pose_id!r}")

    suffix = subject_path.suffix.lower()
    is_image = suffix in _IMAGE_EXTS

    if is_image:
        # Try MediaPipe extraction; if unavailable, synthesize a toy fallback
        try:
            from poseguide.data.extract import extract_pose  # noqa: F811
        except ImportError:
            extract_pose = None

        if extract_pose is not None:
            try:
                subject = extract_pose(subject_path)
            except (RuntimeError, FileNotFoundError) as exc:
                # Extraction failed (no person found, bad image, etc.) → fallback
                subject = _toy_fallback_subject(subject_path)
                result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
                result["subject_id"] = subject.get("id")
                result["source"] = str(subject_path)
                result["extractor"] = "toy-fallback"
                result["extraction_error"] = str(exc)
                return result
        else:
            subject = _toy_fallback_subject(subject_path)
            result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
            result["subject_id"] = subject.get("id")
            result["source"] = str(subject_path)
            result["extractor"] = "toy-fallback"
            result["extraction_error"] = "vision extra not installed"
            return result

        subject = {
            "id": subject.get("id", subject_path.stem),
            "joint_vector": poseguide_vector_from_extract(subject),
        }
        result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
        result["subject_id"] = subject.get("id")
        result["source"] = str(subject_path)
        result["extractor"] = "mediapipe"
        return result

    # Regular JSON subject path (toy default)
    subject = load_subject(subject_path)
    result = ToyPoseRanker().score_match(pose, subject["joint_vector"])
    result["subject_id"] = subject.get("id")
    result["source"] = str(subject_path)
    result["extractor"] = "json-toy"
    return result


def _toy_fallback_subject(path: Path) -> dict:
    """Return a synthetic subject payload when MediaPipe is unavailable."""
    import numpy as np

    rng = np.random.default_rng(hash(path.stem) % (2**31))
    vec = rng.uniform(0.2, 0.8, size=39).astype(np.float64)
    return {
        "id": path.stem,
        "joint_vector": vec,
        "source": "toy-fallback",
    }


def poseguide_vector_from_extract(subject: dict) -> "np.ndarray":
    """Convert a MediaPipe extract payload joints to the joint_vector format."""
    import numpy as np

    from poseguide.data.loader import joints_to_vector

    return joints_to_vector(subject.get("joints") or {})
