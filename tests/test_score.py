from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from poseguide.data.extract import MEDIAPIPE_LANDMARK_MAP
from poseguide.data.loader import list_sample_files
from poseguide.guide.score import (
    VisionScoreUnavailableError,
    score_image_against_pose,
    score_subject,
    score_subject_against_pose,
)


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def _fake_landmarks() -> list[FakeLandmark]:
    """A full 33-entry MediaPipe landmark list with known values."""
    landmarks = [FakeLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for index in MEDIAPIPE_LANDMARK_MAP:
        landmarks[index] = FakeLandmark(index / 100.0, index / 50.0, index / 200.0, 0.9)
    return landmarks


def _fake_detector(landmarks):
    def _detect(image_path: Path):
        return landmarks

    return _detect


def _contrapposto_sample() -> Path:
    return next(p for p in list_sample_files() if "contrapposto" in p.name)


def test_toy_path_scores_offline_without_vision() -> None:
    """The toy/JSON path stays the default and needs no vision deps."""
    result = score_subject_against_pose("contrapposto", _contrapposto_sample())
    assert result["engine"] == "toy"
    assert result["confidence"] >= 0.75


def test_mediapipe_path_scores_image_with_fake_detector(tmp_path: Path) -> None:
    image = tmp_path / "subject.jpg"
    image.write_bytes(b"not-a-real-image-but-file-exists")

    result = score_image_against_pose(
        "contrapposto", image, detector=_fake_detector(_fake_landmarks())
    )

    assert result["engine"] == "mediapipe"
    assert result["source"] == "mediapipe"
    assert "pose_id" in result
    assert "confidence" in result


def test_mediapipe_path_falls_back_when_no_person(tmp_path: Path) -> None:
    """A detector that finds nobody raises a catchable fallback error."""
    image = tmp_path / "empty.jpg"
    image.write_bytes(b"fake")

    with pytest.raises(VisionScoreUnavailableError):
        score_image_against_pose("contrapposto", image, detector=_fake_detector(None))


def test_dispatcher_routes_json_to_toy() -> None:
    result = score_subject("contrapposto", _contrapposto_sample())
    assert result["engine"] == "toy"


def test_dispatcher_routes_image_to_mediapipe(tmp_path: Path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"fake")
    # Auto-detection routes image suffixes to the MediaPipe path. The real
    # MediaPipe detector is unavailable here, so it must surface the catchable
    # fallback error rather than an ImportError or a JSON decode failure.
    with pytest.raises(VisionScoreUnavailableError):
        score_subject("contrapposto", image)


def test_dispatcher_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="engine"):
        score_subject("contrapposto", _contrapposto_sample(), engine="bogus")
