"""Background-aware pose placement preview.

Given a background scene and a chosen pose, produce a placement preview
(silhouette / stick figure) respecting horizon lines, safe margins,
and background context (beach, urban, studio, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poseguide.data.loader import load_pose, load_scene, list_scene_files


# ── background context detection ──────────────────────────────────────────


def detect_background_context(scene: dict) -> str:
    """Return the dominant background context from scene tags and composition.

    Possible values: 'beach', 'urban', 'studio', 'nature', 'indoor', 'unknown'.
    """
    tags = {str(t).strip().lower() for t in (scene.get("tags") or [])}
    mood = {str(m).strip().lower() for m in (scene.get("mood") or [])}
    all_tags = tags | mood
    comp = scene.get("composition") or {}

    if "beach" in all_tags or "ocean" in all_tags or "coastal" in all_tags:
        return "beach"
    if "urban" in all_tags or "street" in all_tags or "city" in all_tags or "alley" in all_tags:
        return "urban"
    if "studio" in all_tags or comp.get("backdrop") == "seamless":
        return "studio"
    if "nature" in all_tags or "forest" in all_tags or "mountain" in all_tags or "garden" in all_tags or "trail" in all_tags:
        return "nature"
    if "indoor" in all_tags or "cafe" in all_tags or "office" in all_tags or "home" in all_tags or "library" in all_tags:
        return "indoor"
    return "unknown"


# ── horizon estimation ────────────────────────────────────────────────────


def estimate_horizon(scene: dict, bg_height: int = 1080) -> int:
    """Estimate horizon Y position (in pixels) from scene composition hints.

    Falls back to rule-of-thirds upper line (1/3 from top) when no explicit
    horizon information is available.
    """
    comp = scene.get("composition") or {}
    horizon_hint = comp.get("horizon", "")

    if horizon_hint == "lower_third":
        return int(bg_height * 2 / 3)
    if horizon_hint == "upper_third":
        return int(bg_height / 3)
    if horizon_hint == "mid":
        return int(bg_height / 2)
    if horizon_hint == "none":
        return bg_height  # full-height subject, no visible horizon

    context = detect_background_context(scene)
    if context == "beach":
        return int(bg_height * 2 / 3)  # horizon typically lower third
    if context == "urban":
        return int(bg_height / 2)  # mid-height for city scenes
    if context == "studio":
        return bg_height  # seamless backdrop, no horizon
    if context == "nature":
        return int(bg_height / 2)

    # Default: rule-of-thirds upper line
    return int(bg_height / 3)


# ── safe margins ──────────────────────────────────────────────────────────


def calculate_safe_margins(
    bg_width: int,
    bg_height: int,
    context: str,
) -> dict[str, int]:
    """Return pixel margins for safe subject placement per context type."""
    base = min(bg_width, bg_height) // 20  # 5% of smaller dimension

    margins = {"top": base, "bottom": base, "left": base, "right": base}

    if context == "studio":
        # Studio portraits: tighter framing, smaller margins
        margins = {k: base * 2 // 3 for k in margins}
    elif context == "beach":
        # Beach: wider framing, generous left/right for negative space
        margins["left"] = base * 2
        margins["right"] = base * 2
    elif context == "urban":
        # Urban: tighter framing, subject fills more of frame
        margins["left"] = base // 2
        margins["right"] = base // 2

    return margins


# ── pose placement calculator ─────────────────────────────────────────────


def calculate_pose_placement(
    bg_width: int,
    bg_height: int,
    horizon_y: int | None = None,
    pose_joints: dict | None = None,
    context: str = "unknown",
    margin: int = 20,
) -> dict[str, Any]:
    """Calculate silhouette/stick-figure placement bounding box.

    Respects horizon line and safe margins. Returns placement box coordinates
    and metadata suitable for rendering.

    Parameters
    ----------
    bg_width : int
        Background width in pixels.
    bg_height : int
        Background height in pixels.
    horizon_y : int or None
        Horizon Y position in pixels. If None, estimated as bg_height/3.
    pose_joints : dict or None
        Normalized joint positions (0–1). Used to compute subject aspect ratio.
    context : str
        Background context: 'beach', 'urban', 'studio', 'nature', 'indoor'.
    margin : int
        Minimum pixel margin from frame edges.

    Returns
    -------
    dict with keys:
        placement_box: {x_min, y_min, width, height}
        fits_safe_margin: bool
        horizon_y: int
        context: str
        scale_factor: float
        subject_placement: str  # description of where subject sits
    """
    if bg_width <= 0 or bg_height <= 0:
        raise ValueError(f"Invalid background dimensions: {bg_width}x{bg_height}")

    if horizon_y is None:
        horizon_y = bg_height // 3

    if horizon_y < 0 or horizon_y > bg_height:
        raise ValueError(f"Horizon Y {horizon_y} out of bounds [0, {bg_height}]")

    margin = max(margin, 5)  # enforce minimum margin

    # Compute subject aspect ratio from joints if available
    if pose_joints:
        ys = [j[1] for j in pose_joints.values() if isinstance(j, (list, tuple)) and len(j) >= 2]
        xs = [j[0] for j in pose_joints.values() if isinstance(j, (list, tuple)) and len(j) >= 2]
        if ys and xs:
            subject_height_norm = max(ys) - min(ys)
            subject_width_norm = max(xs) - min(xs)
            aspect = subject_width_norm / max(subject_height_norm, 0.01)
        else:
            aspect = 0.5  # default human standing aspect
    else:
        aspect = 0.5

    # Available frame area below horizon
    available_height = bg_height - horizon_y

    # Scale subject to fill available height with safe margins
    safe_margins = calculate_safe_margins(bg_width, bg_height, context)
    usable_height = available_height - safe_margins["top"] - safe_margins["bottom"]
    usable_width = bg_width - safe_margins["left"] - safe_margins["right"]

    # Subject height: fill usable height, capped by width constraint
    subject_height = usable_height
    subject_width = subject_height * aspect

    if subject_width > usable_width:
        # Too wide for frame — constrain by width
        subject_width = usable_width
        subject_height = subject_width / max(aspect, 0.01)

    scale_factor = subject_height / max(bg_height, 1)

    # Center horizontally, anchor feet near bottom of usable area
    x_min = (bg_width - int(subject_width)) // 2
    y_min = horizon_y + safe_margins["top"]

    # Adjust for margin
    x_min = max(x_min, margin)
    y_min = max(y_min, horizon_y)

    placement_box = {
        "x_min": x_min,
        "y_min": y_min,
        "width": int(subject_width),
        "height": int(subject_height),
    }

    # Check safe margins
    fits_safe_margin = (
        placement_box["x_min"] >= margin
        and placement_box["y_min"] >= horizon_y
        and placement_box["x_min"] + placement_box["width"] <= bg_width - margin
        and placement_box["y_min"] + placement_box["height"] <= bg_height - margin
    )

    # Determine subject placement description
    x_center = x_min + int(subject_width) // 2
    frame_third = bg_width // 3
    if x_center < frame_third:
        placement_desc = "left_third"
    elif x_center > 2 * frame_third:
        placement_desc = "right_third"
    else:
        placement_desc = "center"

    # Context-specific guidance
    if context == "studio":
        placement_desc = "center_portrait"
    elif context == "beach":
        placement_desc = f"{placement_desc}_rule_of_thirds"

    return {
        "placement_box": placement_box,
        "fits_safe_margin": fits_safe_margin,
        "horizon_y": horizon_y,
        "context": context,
        "scale_factor": round(scale_factor, 4),
        "subject_placement": placement_desc,
    }


# ── scene + pose preview ──────────────────────────────────────────────────


def preview_pose_in_scene(
    scene_id: str,
    pose_id: str,
    bg_width: int = 1920,
    bg_height: int = 1080,
) -> dict[str, Any]:
    """Full placement preview for a pose against a scene background.

    Loads scene metadata, detects context, estimates horizon, loads
    pose joints, and returns the computed placement preview.

    Parameters
    ----------
    scene_id : str
        Scene identifier (e.g., 'beach_sunset', 'studio_gray').
    pose_id : str
        Pose identifier (e.g., 'studio_hand_on_hip_power').
    bg_width, bg_height : int
        Virtual background dimensions in pixels.

    Returns
    -------
    dict with scene info, pose info, and placement preview.
    """
    # Locate scene file
    scene_path = None
    for sf in list_scene_files():
        if sf.stem == scene_id:
            scene_path = sf
            break
    if scene_path is None:
        raise FileNotFoundError(f"Scene '{scene_id}' not found in data/scenes/")

    scene = load_scene(scene_path)
    context = detect_background_context(scene)
    horizon_y = estimate_horizon(scene, bg_height)

    # Load pose
    from poseguide.config import POSES_DIR

    pose_path = POSES_DIR / f"{pose_id}.json"
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose '{pose_id}' not found in data/poses/")

    pose = load_pose(pose_path)
    joints = pose.get("joints") or {}

    placement = calculate_pose_placement(
        bg_width=bg_width,
        bg_height=bg_height,
        horizon_y=horizon_y,
        pose_joints=joints,
        context=context,
    )

    return {
        "scene": {
            "id": scene.get("id"),
            "name": scene.get("name"),
            "tags": scene.get("tags"),
            "context": context,
            "horizon_y": horizon_y,
        },
        "pose": {
            "id": pose.get("id"),
            "name": pose.get("name"),
            "joint_count": len(joints),
        },
        "placement": placement,
    }


# ── batch preview ─────────────────────────────────────────────────────────


def recommend_placement_for_scene(
    scene_id: str,
    pose_ids: list[str] | None = None,
    bg_width: int = 1920,
    bg_height: int = 1080,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Generate placement previews for multiple poses against one scene.

    If pose_ids is None, uses all available poses.
    Returns top_k results sorted by best fit score.
    """
    from poseguide.data.loader import list_pose_files

    if pose_ids is None:
        pose_ids = [p.stem for p in list_pose_files()]

    results = []
    for pid in pose_ids[:top_k]:
        try:
            preview = preview_pose_in_scene(scene_id, pid, bg_width, bg_height)
            preview["score"] = _placement_score(preview["placement"])
            results.append(preview)
        except FileNotFoundError:
            continue

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _placement_score(placement: dict) -> float:
    """Score a placement result (0–1). Higher is better."""
    score = 1.0
    if not placement.get("fits_safe_margin", False):
        score *= 0.5
    sf = placement.get("scale_factor", 0)
    if sf < 0.3:
        score *= 0.7  # subject too small
    elif sf > 0.9:
        score *= 0.8  # subject too large (cropped feel)
    return round(score, 3)


# ── CLI helper ────────────────────────────────────────────────────────────


def cli_preview(scene_id: str, pose_id: str) -> None:
    """Print a human-readable placement preview to stdout."""
    try:
        result = preview_pose_in_scene(scene_id, pose_id)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    p = result["placement"]
    box = p["placement_box"]

    print(f"Scene:  {result['scene']['name']} ({result['scene']['context']})")
    print(f"Pose:   {result['pose']['name']}")
    print(f"Horizon: {p['horizon_y']}px")
    print(f"Placement: {p['subject_placement']}")
    print(f"Bounding box: x={box['x_min']}, y={box['y_min']}, "
          f"w={box['width']}, h={box['height']}")
    print(f"Safe margin: {'PASS' if p['fits_safe_margin'] else 'FAIL'}")
    print(f"Scale: {p['scale_factor']}")
