"""Tests for studio_hand_on_hip_power pose pack."""

from __future__ import annotations

import json
from pathlib import Path

from poseguide.config import POSES_DIR
from poseguide.data.loader import load_pose, list_pose_files


def test_studio_hand_on_hip_power_exists() -> None:
    """The studio_hand_on_hip_power pose JSON must be present in data/poses/."""
    pose_path = POSES_DIR / "studio_hand_on_hip_power.json"
    assert pose_path.exists(), f"Missing pose file: {pose_path}"


def test_studio_hand_on_hip_power_valid_json() -> None:
    """The pose file must be valid JSON."""
    pose_path = POSES_DIR / "studio_hand_on_hip_power.json"
    data = json.loads(pose_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_studio_hand_on_hip_power_required_fields() -> None:
    """The pose must have all required fields: id, name, standing, tags, joints."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    assert pose["id"] == "studio_hand_on_hip_power"
    assert isinstance(pose["name"], str) and len(pose["name"]) > 0
    assert pose["standing"] is True


def test_studio_hand_on_hip_power_tags() -> None:
    """The pose must have indoor, studio, portrait, and power tags."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    tags = [t.lower() for t in pose.get("tags", [])]
    assert "studio" in tags
    assert "indoor" in tags
    assert "portrait" in tags
    assert "power" in tags or "confident" in tags


def test_studio_hand_on_hip_power_joint_count() -> None:
    """The pose must define at least 10 joints for a full standing skeleton."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    joints = pose.get("joints") or {}
    # 13 standard joints: nose, l/r shoulder, elbow, wrist, hip, knee, ankle
    assert len(joints) >= 10, f"Expected >= 10 joints, got {len(joints)}"


def test_studio_hand_on_hip_power_hand_on_hip() -> None:
    """Right wrist should be near the right hip to represent the hand-on-hip pose."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    joints = pose.get("joints") or {}
    r_wrist = joints.get("r_wrist")
    r_hip = joints.get("r_hip")
    assert r_wrist is not None, "Missing r_wrist joint"
    assert r_hip is not None, "Missing r_hip joint"
    # Right wrist X should be close to right hip X (hand rests on hip)
    wrist_x, hip_x = r_wrist[0], r_hip[0]
    assert abs(wrist_x - hip_x) < 0.25, (
        f"Right wrist X ({wrist_x}) too far from right hip X ({hip_x})"
    )


def test_studio_hand_on_hip_power_tips() -> None:
    """The pose must have meaningful photography tips."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    tips = pose.get("tips") or []
    assert len(tips) >= 3, f"Expected >= 3 tips, got {len(tips)}"
    assert any("chest" in t.lower() or "shoulder" in t.lower() for t in tips)


def test_studio_hand_on_hip_power_camera_cues() -> None:
    """Camera cues should guide studio portrait lighting and framing."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    cues = pose.get("camera_cues") or []
    assert len(cues) >= 2, f"Expected >= 2 camera cues, got {len(cues)}"


def test_studio_hand_on_hip_power_joint_vector() -> None:
    """The joint_vector must be a 39-element array (13 joints × 3 coords)."""
    pose = load_pose(POSES_DIR / "studio_hand_on_hip_power.json")
    jv = pose.get("joint_vector")
    assert jv is not None, "joint_vector missing after load_pose"
    assert jv.size == 39, f"Expected 39 (13×3), got {jv.size}"


def test_list_pose_files_includes_studio_pose() -> None:
    """list_pose_files() must include the new studio pose."""
    files = list_pose_files()
    names = {f.stem for f in files}
    assert "studio_hand_on_hip_power" in names
