"""Export PoseGuide poses to COCO / MediaPipe keypoint formats.

Round-trip fidelity: PoseGuide ↔ COCO ↔ MediaPipe, where the PoseGuide joint
set is a strict subset of both COCO (17-keypoint) and MediaPipe (33-landmark).
Unmapped joints are padded with ``[0, 0, 0]`` and visibility/confidence 0.
"""

from __future__ import annotations

import json
from pathlib import Path

# PoseGuide joint order (matches data/loader.py joints_to_vector)
POSEGUIDE_JOINT_ORDER: tuple[str, ...] = (
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)

# COCO 17-keypoint order
COCO_KEYPOINT_ORDER: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# PoseGuide key → COCO index
_POSEGUIDE_TO_COCO: dict[str, int] = {}
_COCO_MAP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
# Reverse: COCO index → PoseGuide key
COCO_TO_POSEGUIDE: dict[int, str] = {}
for _coco_name, _idx in _COCO_MAP.items():
    _pg_name = _coco_name.replace("left_", "l_").replace("right_", "r_")
    if _pg_name in frozenset(POSEGUIDE_JOINT_ORDER):
        _POSEGUIDE_TO_COCO[_pg_name] = _idx
        COCO_TO_POSEGUIDE[_idx] = _pg_name

# MediaPipe Pose landmark index → PoseGuide key (from data/extract.py)
MEDIAPIPE_LANDMARK_MAP: dict[int, str] = {
    0: "nose",
    11: "l_shoulder",
    12: "r_shoulder",
    13: "l_elbow",
    14: "r_elbow",
    15: "l_wrist",
    16: "r_wrist",
    23: "l_hip",
    24: "r_hip",
    25: "l_knee",
    26: "r_knee",
    27: "l_ankle",
    28: "r_ankle",
}

# PoseGuide key → MediaPipe index
POSEGUIDE_TO_MEDIAPIPE: dict[str, int] = {
    v: k for k, v in MEDIAPIPE_LANDMARK_MAP.items()
}

# MediaPipe index → PoseGuide key (alias)
MEDIAPIPE_TO_POSEGUIDE: dict[int, str] = dict(MEDIAPIPE_LANDMARK_MAP)


def pose_to_coco(pose: dict) -> dict:
    """Convert a PoseGuide pose dict to COCO keypoints format.

    Returns a dict with ``"image_id"``, ``"category_id"``, ``"keypoints"`` (list
    of 51 floats: 17 × [x, y, v]), and ``"num_keypoints"``.
    """
    joints: dict[str, list[float]] = pose.get("joints", {})
    keypoints: list[float] = []
    num_keypoints = 0
    for i in range(17):
        pg_key = COCO_TO_POSEGUIDE.get(i)
        if pg_key and pg_key in joints and len(joints[pg_key]) >= 2:
            x, y = float(joints[pg_key][0]), float(joints[pg_key][1])
            keypoints.extend([x, y, 2.0])  # v=2: labeled and visible
            num_keypoints += 1
        else:
            keypoints.extend([0.0, 0.0, 0.0])  # v=0: not labeled
    return {
        "image_id": pose.get("id", "unknown"),
        "category_id": 1,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "poseguide_id": pose.get("id"),
        "poseguide_name": pose.get("name"),
    }


def coco_to_poseguide(coco_annotation: dict) -> dict:
    """Convert a COCO annotation back to PoseGuide joints dict.

    Only recovers the 13 PoseGuide joints; face keypoints are discarded.
    """
    keypoints = coco_annotation.get("keypoints", [])
    joints: dict[str, list[float]] = {}
    for i, pg_key in COCO_TO_POSEGUIDE.items():
        offset = i * 3
        if offset + 2 < len(keypoints):
            x, y, v = keypoints[offset], keypoints[offset + 1], keypoints[offset + 2]
            if v > 0:
                joints[pg_key] = [round(float(x), 6), round(float(y), 6)]
            else:
                joints[pg_key] = [0.0, 0.0]
    return joints


def pose_to_mediapipe(pose: dict) -> dict:
    """Convert a PoseGuide pose dict to MediaPipe landmark list format.

    Returns a dict with ``"landmarks"`` (list of 33 ``{x, y, z, visibility}``
    dicts) and metadata.
    """
    joints: dict[str, list[float]] = pose.get("joints", {})
    landmarks: list[dict] = []
    for i in range(33):
        pg_key = MEDIAPIPE_TO_POSEGUIDE.get(i)
        if pg_key and pg_key in joints and len(joints[pg_key]) >= 2:
            coords = joints[pg_key]
            x, y = float(coords[0]), float(coords[1])
            z = float(coords[2]) if len(coords) > 2 else 0.0
            landmarks.append({
                "x": round(x, 6),
                "y": round(y, 6),
                "z": round(z, 6),
                "visibility": 1.0,
            })
        else:
            landmarks.append({
                "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0,
            })
    return {
        "landmarks": landmarks,
        "poseguide_id": pose.get("id"),
        "poseguide_name": pose.get("name"),
    }


def mediapipe_to_poseguide(mp_result: dict) -> dict:
    """Convert a MediaPipe landmark list back to PoseGuide joints dict."""
    landmarks = mp_result.get("landmarks", [])
    joints: dict[str, list[float]] = {}
    for mp_idx, pg_key in MEDIAPIPE_TO_POSEGUIDE.items():
        if mp_idx < len(landmarks):
            lm = landmarks[mp_idx]
            if lm.get("visibility", 0) > 0:
                joints[pg_key] = [
                    round(float(lm["x"]), 6),
                    round(float(lm["y"]), 6),
                    round(float(lm.get("z", 0.0)), 6),
                ]
            else:
                joints[pg_key] = [0.0, 0.0, 0.0]
    return joints


def export_poses(
    format: str = "coco",
    *,
    pose_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    """Export all shipped poses to COCO or MediaPipe format.

    Parameters
    ----------
    format: ``"coco"`` or ``"mediapipe"``.
    pose_dir: Optional directory of pose JSON files (default: data/poses/).
    out_path: Optional output file path.

    Returns
    -------
    dict with ``"annotations"`` (COCO) or ``"poses"`` (MediaPipe) and file path.
    """
    from poseguide.config import POSES_DIR
    from poseguide.data.loader import list_pose_files, load_pose

    if format not in ("coco", "mediapipe"):
        raise ValueError(f"Unknown format: {format!r}. Use 'coco' or 'mediapipe'.")

    pose_files = list_pose_files(pose_dir)
    if not pose_files:
        raise FileNotFoundError("No pose files found.")

    if format == "coco":
        annotations = []
        for path in pose_files:
            pose = load_pose(path)
            annotations.append(pose_to_coco(pose))
        result = {"annotations": annotations}
    else:
        poses = []
        for path in pose_files:
            pose = load_pose(path)
            poses.append(pose_to_mediapipe(pose))
        result = {"poses": poses}

    if out_path is None:
        from poseguide.config import OUT_DIR
        out_path = OUT_DIR / f"poses_export_{format}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["_exported_to"] = str(out_path)
    result["_count"] = len(pose_files)
    return result
