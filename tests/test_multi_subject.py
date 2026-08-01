"""Tests for multi-subject/couple pose support (Issue #14)."""
import json
import os
from pathlib import Path

import pytest

from poseguide.data.schema import (
    Pose,
    SubjectPose,
    validate_pose,
    validate_pose_file,
    rank_pose_for_scene,
    Scene,
)


COUPLE_POSE_IDS = [
    "couple_standing_side_by_side",
    "couple_embrace_face_to_face",
    "couple_walking_hand_in_hand",
    "couple_back_to_back",
    "couple_dip_kiss",
    "couple_first_dance",
    "couple_piggyback",
    "couple_sitting_bench",
]


def test_all_couple_poses_exist():
    """Verify all 8 couple pose files exist in data/poses/."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        assert path.exists(), f"Missing: {path}"


def test_couple_poses_have_subjects():
    """Verify couple poses have the subjects field."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        data = json.loads(path.read_text())
        assert "subjects" in data, f"{pose_id}: missing subjects"
        assert len(data["subjects"]) >= 2, f"{pose_id}: need >=2 subjects, got {len(data['subjects'])}"


def test_couple_poses_have_valid_joints():
    """Verify each subject has valid joint coordinates."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    required = {"l_shoulder", "r_shoulder", "l_hip", "r_hip"}
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        data = json.loads(path.read_text())
        for subj in data["subjects"]:
            joints = set(subj.get("joints", {}).keys())
            missing = required - joints
            assert not missing, f"{pose_id}/{subj['label']}: missing {missing}"


def test_couple_poses_have_scene_rankings():
    """Verify couple poses have scene_rankings for multi-scene support."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        data = json.loads(path.read_text())
        rankings = data.get("scene_rankings", {})
        assert len(rankings) >= 2, f"{pose_id}: need >=2 scene rankings, got {len(rankings)}"


def test_subject_count_matches_subjects():
    """Verify subject_count equals the number of subjects."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        data = json.loads(path.read_text())
        assert data.get("subject_count", 0) == len(data.get("subjects", [])),             f"{pose_id}: subject_count mismatch"


def test_validate_couple_poses():
    """All couple poses pass schema validation."""
    poses_dir = Path(__file__).parent.parent / "data" / "poses"
    for pose_id in COUPLE_POSE_IDS:
        path = poses_dir / f"{pose_id}.json"
        pose = validate_pose_file(path)
        assert pose.id == pose_id
        assert pose.subject_count >= 2
        assert pose.subjects is not None
        assert len(pose.subjects) == pose.subject_count


def test_separate_models():
    """SubjectPose validates independently."""
    subj = SubjectPose(
        label="test_subject",
        joints={"l_shoulder": [0.4, 0.3, 0.0], "r_shoulder": [0.6, 0.3, 0.0],
                "l_hip": [0.45, 0.6, 0.0], "r_hip": [0.55, 0.6, 0.0]}
    )
    assert subj.label == "test_subject"


def test_rank_pose_for_scene_direct():
    """Scene rankings use direct scores when available."""
    pose = validate_pose_file(
        Path(__file__).parent.parent / "data" / "poses" / "couple_dip_kiss.json"
    )
    scene = Scene(id="wedding_garden", name="Wedding Garden", tags=["outdoor", "wedding"])
    score = rank_pose_for_scene(pose, scene)
    assert score == 1.0, f"Expected 1.0 for wedding_garden, got {score}"


def test_rank_pose_for_scene_tag_fallback():
    """Scene rankings fall back to tag matching when no direct score."""
    pose = validate_pose_file(
        Path(__file__).parent.parent / "data" / "poses" / "couple_back_to_back.json"
    )
    scene = Scene(id="festival_crowd", name="Festival Crowd", tags=["street", "night"])
    score = rank_pose_for_scene(pose, scene)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
