"""Fallback tests for the optional vision subject-score path (issue #34).

The vision path must degrade gracefully to the offline toy baseline whenever the
optional ``vision`` extra (MediaPipe/OpenCV) is unavailable, while the toy path
stays the default.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from poseguide.data.extract import MEDIAPIPE_LANDMARK_MAP
from poseguide.data.loader import list_sample_files
from poseguide.guide.score import score_subject, score_subject_against_pose
from poseguide.models.catalog import get_pose_by_id
from poseguide.models.vision import VisionUnavailableError, score_subject_image


def _fake_landmarks(pose_id: str) -> list:
    """Build MediaPipe-style landmarks reproducing a catalog pose's joints."""
    pose = get_pose_by_id(pose_id)
    joints = pose["joints"]
    key_to_index = {key: idx for idx, key in MEDIAPIPE_LANDMARK_MAP.items()}
    landmarks: list = [None] * 33
    for key, xyz in joints.items():
        idx = key_to_index[key]
        landmarks[idx] = types.SimpleNamespace(
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]) if len(xyz) > 2 else 0.0,
            visibility=1.0,
        )
    for i in range(33):
        if landmarks[i] is None:
            landmarks[i] = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0)
    return landmarks


def test_default_path_is_offline_toy() -> None:
    """The default scoring path stays the offline toy path (no vision involved)."""
    sample = next(p for p in list_sample_files() if "contrapposto" in p.name)
    result = score_subject("contrapposto", sample)  # vision=False is the default
    assert result["confidence"] >= 0.75
    assert result.get("path", "toy") != "vision"


def test_toy_path_unaffected() -> None:
    """The legacy toy entry point keeps working unchanged."""
    sample = next(p for p in list_sample_files() if "contrapposto" in p.name)
    result = score_subject_against_pose("contrapposto", sample)
    assert result["confidence"] >= 0.75
    assert result["cues"]


def test_vision_falls_back_when_extra_missing(tmp_path: Path, monkeypatch) -> None:
    """Without the vision extra, photo scoring degrades to the toy baseline."""
    image = tmp_path / "subject.png"
    image.write_bytes(b"fake-image-bytes")

    def _raise() -> None:
        raise RuntimeError("MediaPipe pose extraction requires the optional 'vision' extra.")

    monkeypatch.setattr("poseguide.data.extract.default_mediapipe_detector", _raise)

    result = score_subject("contrapposto", image, vision=True)
    assert result["vision_available"] is False
    assert result["path"] == "toy"
    assert result["confidence"] is None
    assert result["fallback_reason"]


def test_vision_raises_without_fallback(tmp_path: Path, monkeypatch) -> None:
    """allow_fallback=False surfaces the missing vision extra as an error."""
    image = tmp_path / "subject.png"
    image.write_bytes(b"fake-image-bytes")

    def _raise() -> None:
        raise RuntimeError("vision extra missing")

    monkeypatch.setattr("poseguide.data.extract.default_mediapipe_detector", _raise)

    with pytest.raises(VisionUnavailableError):
        score_subject("contrapposto", image, vision=True, allow_fallback=False)


def test_vision_path_scores_with_injected_detector(tmp_path: Path) -> None:
    """With an injected detector the vision path returns a real joint match."""
    image = tmp_path / "subject.png"
    image.write_bytes(b"fake-image-bytes")

    def detector(path: Path) -> list:
        return _fake_landmarks("contrapposto")

    result = score_subject_image("contrapposto", image, detector=detector, allow_fallback=False)
    assert result["vision_available"] is True
    assert result["path"] == "vision"
    assert result["source"] == "mediapipe"
    assert result["confidence"] >= 0.75


def test_vision_unknown_pose_raises(tmp_path: Path) -> None:
    image = tmp_path / "subject.png"
    image.write_bytes(b"fake-image-bytes")
    with pytest.raises(KeyError):
        score_subject("does-not-exist", image, vision=True)


def test_vision_missing_image_raises() -> None:
    with pytest.raises(FileNotFoundError):
        score_subject("contrapposto", Path("no/such/image.png"), vision=True)
