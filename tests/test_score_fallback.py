"""Fallback tests for the optional MediaPipe subject-score path (#34).

Issue #34 asks for an *optional* vision path while keeping the toy path as the
offline default, with fallback tests. These tests cover every branch of
``score_subject_against_pose`` without requiring the ``vision`` extra:

* JSON subject  -> offline toy path (default).
* image, MediaPipe unavailable / no person found -> toy fallback.
* image, extraction succeeds -> MediaPipe path.
* unknown pose -> ``KeyError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poseguide.data.extract import JOINT_KEYS
from poseguide.data.loader import list_sample_files
from poseguide.guide.score import score_subject_against_pose


def _contrapposto_sample() -> Path:
    samples = list_sample_files()
    assert samples
    return next(p for p in samples if "contrapposto" in p.name)


def test_json_subject_uses_toy_path_offline() -> None:
    """A JSON subject is scored with the toy ranker (offline default)."""
    result = score_subject_against_pose("contrapposto", _contrapposto_sample())

    assert result["extractor"] == "json-toy"
    assert result["subject_id"]
    assert "confidence" in result
    assert "cues" in result


def test_image_without_vision_extra_falls_back_to_toy(monkeypatch, tmp_path: Path) -> None:
    """When MediaPipe cannot be imported/created, an image falls back to toy."""

    def _boom(image_path, *args, **kwargs):
        raise RuntimeError(
            "MediaPipe pose extraction requires the optional 'vision' extra. "
            "Install it with: pip install 'poseguide[vision]'"
        )

    monkeypatch.setattr("poseguide.data.extract.extract_pose", _boom)

    image = tmp_path / "subject.jpg"
    image.write_bytes(b"not-a-real-image")

    result = score_subject_against_pose("contrapposto", image)

    assert result["extractor"] == "toy-fallback"
    assert result["subject_id"] == "subject"
    assert "vision" in result["extraction_error"]
    assert "confidence" in result


def test_image_with_no_person_falls_back_to_toy(monkeypatch, tmp_path: Path) -> None:
    """A detector that finds no person triggers the toy fallback."""

    def _no_person(image_path, *args, **kwargs):
        raise RuntimeError("No pose landmarks detected in image")

    monkeypatch.setattr("poseguide.data.extract.extract_pose", _no_person)

    image = tmp_path / "empty.png"
    image.write_bytes(b"fake")

    result = score_subject_against_pose("contrapposto", image)

    assert result["extractor"] == "toy-fallback"
    assert result["extraction_error"]
    assert "confidence" in result


def test_image_with_successful_extraction_uses_mediapipe(monkeypatch, tmp_path: Path) -> None:
    """When extraction succeeds, the result is tagged as the MediaPipe path."""

    joints = {key: [0.5, 0.5, 0.0] for key in JOINT_KEYS}

    def _ok(image_path, *args, **kwargs):
        return {"id": "subject", "joints": joints}

    monkeypatch.setattr("poseguide.data.extract.extract_pose", _ok)

    image = tmp_path / "subject.jpg"
    image.write_bytes(b"fake")

    result = score_subject_against_pose("contrapposto", image)

    assert result["extractor"] == "mediapipe"
    assert result["subject_id"] == "subject"
    assert "confidence" in result


def test_unknown_pose_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown pose"):
        score_subject_against_pose("does-not-exist", _contrapposto_sample())
