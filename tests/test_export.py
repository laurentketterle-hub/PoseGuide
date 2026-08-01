"""Round-trip tests for COCO and MediaPipe export/import of pose joints."""

import json
import pytest

from poseguide.data.export_formats import (
    pose_to_coco,
    coco_to_poseguide,
    pose_to_mediapipe,
    mediapipe_to_poseguide,
    export_poses,
    POSEGUIDE_JOINT_ORDER,
)


# --- Unit: COCO round-trip ---

def test_pose_to_coco_shape():
    pose = {
        "id": "test_pose",
        "name": "Test Pose",
        "joints": {
            "nose": [0.5, 0.1],
            "l_shoulder": [0.4, 0.2],
            "r_shoulder": [0.6, 0.2],
            "l_hip": [0.45, 0.7],
            "r_hip": [0.55, 0.7],
        },
    }
    result = pose_to_coco(pose)
    assert result["image_id"] == "test_pose"
    assert len(result["keypoints"]) == 51  # 17 × 3
    # Nose should be mapped to COCO index 0
    assert result["keypoints"][0:3] == [0.5, 0.1, 2.0]
    assert result["num_keypoints"] == 5  # 5 joints mapped


def test_pose_to_coco_missing_joints():
    """Unmapped PoseGuide joints in COCO (eyes, ears) become [0, 0, 0]."""
    pose = {
        "id": "min",
        "name": "Minimal",
        "joints": {},
    }
    result = pose_to_coco(pose)
    assert result["num_keypoints"] == 0
    assert result["keypoints"] == [0.0] * 51


def test_coco_round_trip_preserves_joints():
    """Round-trip: PoseGuide → COCO → PoseGuide preserves joint values."""
    pose = {
        "id": "roundtrip",
        "name": "Round Trip",
        "joints": {
            "nose": [0.5, 0.1],
            "l_shoulder": [0.4, 0.2, 0.1],
            "r_shoulder": [0.6, 0.2, 0.1],
            "l_elbow": [0.35, 0.4],
            "r_elbow": [0.65, 0.4],
            "l_wrist": [0.3, 0.6],
            "r_wrist": [0.7, 0.6],
            "l_hip": [0.45, 0.7],
            "r_hip": [0.55, 0.7],
            "l_knee": [0.4, 0.85],
            "r_knee": [0.6, 0.85],
            "l_ankle": [0.35, 0.95],
            "r_ankle": [0.65, 0.95],
        },
    }
    coco = pose_to_coco(pose)
    recovered = coco_to_poseguide(coco)
    # All PoseGuide joints should round-trip
    for key in POSEGUIDE_JOINT_ORDER:
        assert key in recovered, f"Missing {key} after round-trip"
        assert recovered[key][:2] == pose["joints"][key][:2], f"Mismatch for {key}"


# --- Unit: MediaPipe round-trip ---

def test_pose_to_mediapipe_shape():
    pose = {
        "id": "test_mp",
        "name": "Test MP",
        "joints": {
            "nose": [0.5, 0.1, 0.0],
            "l_shoulder": [0.4, 0.2],
            "r_shoulder": [0.6, 0.2],
            "l_hip": [0.45, 0.7],
            "r_hip": [0.55, 0.7],
        },
    }
    result = pose_to_mediapipe(pose)
    assert len(result["landmarks"]) == 33
    # Nose at MP index 0
    assert result["landmarks"][0]["x"] == 0.5
    assert result["landmarks"][0]["y"] == 0.1
    assert result["landmarks"][0]["visibility"] == 1.0
    # Unmapped landmark should be zero
    assert result["landmarks"][1]["visibility"] == 0.0


def test_mediapipe_round_trip_preserves_joints():
    """Round-trip: PoseGuide → MediaPipe → PoseGuide preserves joint values."""
    pose = {
        "id": "mp_roundtrip",
        "name": "MP Round Trip",
        "joints": {
            "nose": [0.5, 0.1, 0.0],
            "l_shoulder": [0.4, 0.2, 0.1],
            "r_shoulder": [0.6, 0.2, 0.1],
            "l_elbow": [0.35, 0.4, 0.1],
            "r_elbow": [0.65, 0.4, 0.1],
            "l_wrist": [0.3, 0.6, 0.1],
            "r_wrist": [0.7, 0.6, 0.1],
            "l_hip": [0.45, 0.7, 0.2],
            "r_hip": [0.55, 0.7, 0.2],
            "l_knee": [0.4, 0.85, 0.3],
            "r_knee": [0.6, 0.85, 0.3],
            "l_ankle": [0.35, 0.95, 0.5],
            "r_ankle": [0.65, 0.95, 0.5],
        },
    }
    mp = pose_to_mediapipe(pose)
    recovered = mediapipe_to_poseguide(mp)
    for key in POSEGUIDE_JOINT_ORDER:
        assert key in recovered, f"Missing {key} after MP round-trip"
        for dim in range(len(pose["joints"][key])):
            assert recovered[key][dim] == pytest.approx(pose["joints"][key][dim], abs=1e-5), \
                f"Mismatch for {key}[{dim}]"


# --- Integration: export_poses CLI function ---

def test_export_poses_coco(monkeypatch, tmp_path):
    """export_poses with coco format writes annotations."""
    from poseguide.data.loader import load_pose
    from poseguide.config import POSES_DIR, OUT_DIR
    
    # Use real data directory
    try:
        result = export_poses(format="coco", out_path=tmp_path / "coco_export.json")
    except FileNotFoundError:
        pytest.skip("No pose fixture files found")

    assert result["_count"] > 0
    assert "annotations" in result
    for ann in result["annotations"]:
        assert len(ann["keypoints"]) == 51
        assert "image_id" in ann


def test_export_poses_mediapipe(monkeypatch, tmp_path):
    """export_poses with mediapipe format writes landmarks."""
    try:
        result = export_poses(format="mediapipe", out_path=tmp_path / "mp_export.json")
    except FileNotFoundError:
        pytest.skip("No pose fixture files found")

    assert result["_count"] > 0
    assert "poses" in result
    for mp_pose in result["poses"]:
        assert len(mp_pose["landmarks"]) == 33


def test_export_poses_invalid_format():
    """Invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Unknown format"):
        export_poses(format="invalid")


def test_export_poses_round_trip_coco(tmp_path):
    """Full pipeline: export to COCO, then re-import all poses."""
    try:
        result = export_poses(format="coco", out_path=tmp_path / "full_coco.json")
    except FileNotFoundError:
        pytest.skip("No pose fixture files found")

    for ann in result["annotations"]:
        recovered = coco_to_poseguide(ann)
        # At minimum, the required joints should exist
        assert "l_shoulder" in recovered
        assert "r_shoulder" in recovered
        assert "l_hip" in recovered
        assert "r_hip" in recovered
