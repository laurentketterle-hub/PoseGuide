"""Export pose joints to COCO / MediaPipe-compatible formats (#22)."""
from __future__ import annotations

import json
from pathlib import Path

from poseguide.config import OUT_DIR
from poseguide.data.loader import list_pose_files, load_pose

# MediaPipe Pose landmark mapping (33 landmarks)
MEDIAPIPE_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner",
    "right_eye","right_eye_outer","left_ear","right_ear","mouth_left",
    "mouth_right","l_shoulder","r_shoulder","l_elbow","r_elbow",
    "l_wrist","r_wrist","l_pinky","r_pinky","l_index",
    "r_index","l_thumb","r_thumb","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle","l_heel",
    "r_heel","l_foot_index","r_foot_index"
]

def pose_to_mediapipe(pose: dict) -> dict:
    """Convert PoseGuide joints to MediaPipe 33-landmark format."""
    joints = pose.get("joints", {})
    landmarks = []
    for i, name in enumerate(MEDIAPIPE_NAMES):
        pt = joints.get(name, [0.0, 0.0, 0.0])
        landmarks.append({"id": i, "x": pt[0], "y": pt[1], "z": pt[2] if len(pt)>2 else 0.0})
    return {"pose_id": pose.get("id"), "name": pose.get("name"), "landmarks": landmarks}

def pose_to_coco(pose: dict) -> dict:
    """Convert PoseGuide joints to COCO 17-keypoint format."""
    joints = pose.get("joints", {})
    coco_map = {
        "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
        "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16
    }
    keypoints = []
    for name, idx in sorted(coco_map.items(), key=lambda x: x[1]):
        pt = joints.get(name, [0, 0, 0])
        keypoints.extend([float(pt[0]), float(pt[1]), 2.0 if pt[0] or pt[1] else 0.0])
    return {"pose_id": pose.get("id"), "name": pose.get("name"), "keypoints": keypoints, "num_keypoints": 17}

def export_all_poses(fmt: str = "mediapipe", out_dir: Path | None = None):
    """Export all poses to the requested format."""
    out_dir = out_dir or (OUT_DIR / "export")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in list_pose_files():
        pose = load_pose(path)
        if fmt == "coco":
            data = pose_to_coco(pose)
        else:
            data = pose_to_mediapipe(pose)
        out_path = out_dir / f"{pose['id']}_{fmt}.json"
        out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out_dir
