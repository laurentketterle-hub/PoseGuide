"""Tests for background-aware pose placement preview."""

from __future__ import annotations

import pytest

from poseguide.bg_placement import (
    calculate_pose_placement,
    calculate_safe_margins,
    cli_preview,
    detect_background_context,
    estimate_horizon,
    preview_pose_in_scene,
    recommend_placement_for_scene,
)


# ── context detection ─────────────────────────────────────────────────────


def test_detect_beach_context() -> None:
    scene = {"tags": ["beach", "outdoor", "golden_hour"], "mood": ["warm"]}
    assert detect_background_context(scene) == "beach"


def test_detect_urban_context() -> None:
    scene = {"tags": ["urban", "street", "night"], "mood": []}
    assert detect_background_context(scene) == "urban"


def test_detect_studio_context_via_tags() -> None:
    scene = {"tags": ["studio", "indoor", "portrait"], "mood": []}
    assert detect_background_context(scene) == "studio"


def test_detect_studio_context_via_composition() -> None:
    scene = {"tags": ["indoor"], "composition": {"backdrop": "seamless"}}
    assert detect_background_context(scene) == "studio"


def test_detect_nature_context() -> None:
    scene = {"tags": ["forest", "trail", "daylight"], "mood": []}
    assert detect_background_context(scene) == "nature"


def test_detect_indoor_context() -> None:
    scene = {"tags": ["cafe", "indoor"], "mood": []}
    assert detect_background_context(scene) == "indoor"


def test_detect_unknown_context() -> None:
    scene = {"tags": ["abstract"], "mood": []}
    assert detect_background_context(scene) == "unknown"


# ── horizon estimation ────────────────────────────────────────────────────


def test_estimate_horizon_lower_third() -> None:
    scene = {"composition": {"horizon": "lower_third"}}
    assert estimate_horizon(scene, 1080) == 720


def test_estimate_horizon_upper_third() -> None:
    scene = {"composition": {"horizon": "upper_third"}}
    assert estimate_horizon(scene, 1080) == 360


def test_estimate_horizon_mid() -> None:
    scene = {"composition": {"horizon": "mid"}}
    assert estimate_horizon(scene, 1080) == 540


def test_estimate_horizon_none() -> None:
    scene = {"composition": {"horizon": "none"}}
    assert estimate_horizon(scene, 1080) == 1080


def test_estimate_horizon_studio_context() -> None:
    scene = {"tags": ["studio", "indoor"], "composition": {"backdrop": "seamless"}}
    h = estimate_horizon(scene, 1080)
    assert h == 1080  # no horizon for seamless backdrop


def test_estimate_horizon_beach_context() -> None:
    scene = {"tags": ["beach", "outdoor"]}
    h = estimate_horizon(scene, 1080)
    assert h == 720  # lower third


def test_estimate_horizon_default() -> None:
    scene = {"tags": ["unknown"]}
    h = estimate_horizon(scene, 1080)
    assert h == 360  # default: upper third


# ── safe margins ──────────────────────────────────────────────────────────


def test_safe_margins_studio() -> None:
    m = calculate_safe_margins(1920, 1080, "studio")
    assert all(v >= 0 for v in m.values())
    assert m["left"] < 54  # tighter than default


def test_safe_margins_beach() -> None:
    m = calculate_safe_margins(1920, 1080, "beach")
    assert m["left"] > m["top"]  # wider left/right for negative space


def test_safe_margins_urban() -> None:
    m = calculate_safe_margins(1920, 1080, "urban")
    assert m["left"] < 54  # tighter framing


# ── pose placement calculation ────────────────────────────────────────────


def test_calculate_placement_valid() -> None:
    joints = {
        "nose": [0.5, 0.12, 0.0],
        "l_ankle": [0.45, 0.9, 0.0],
        "r_ankle": [0.55, 0.9, 0.0],
    }
    result = calculate_pose_placement(1920, 1080, 540, joints, "studio", margin=20)
    assert result["placement_box"]["x_min"] >= 20
    assert result["placement_box"]["y_min"] >= 540
    assert result["placement_box"]["width"] > 0
    assert result["placement_box"]["height"] > 0
    assert result["fits_safe_margin"] is True
    assert result["horizon_y"] == 540


def test_calculate_placement_beach() -> None:
    joints = {
        "nose": [0.5, 0.12, 0.0],
        "l_ankle": [0.45, 0.9, 0.0],
        "r_ankle": [0.55, 0.9, 0.0],
    }
    result = calculate_pose_placement(1920, 1080, 720, joints, "beach", margin=20)
    assert result["context"] == "beach"
    assert result["fits_safe_margin"] is True


def test_calculate_placement_no_joints() -> None:
    """Placement should still work without joints (default aspect ratio)."""
    result = calculate_pose_placement(1920, 1080, 540, context="urban")
    assert result["placement_box"]["width"] > 0
    assert result["placement_box"]["height"] > 0


def test_placement_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="Invalid background dimensions"):
        calculate_pose_placement(0, 1080, 540, context="studio")


def test_placement_horizon_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="Horizon Y"):
        calculate_pose_placement(1920, 1080, 9999, context="studio")


def test_placement_fits_margin_studio() -> None:
    joints = {
        "nose": [0.5, 0.12, 0.0],
        "l_ankle": [0.45, 0.9, 0.0],
        "r_ankle": [0.55, 0.9, 0.0],
    }
    result = calculate_pose_placement(1920, 1080, 1080, joints, "studio")
    # Studio has no horizon (1080=full height), subject fills entire frame
    assert result["context"] == "studio"


# ── scene + pose preview ──────────────────────────────────────────────────


def test_preview_beach_sunset_with_power_stance() -> None:
    """Integration test: preview a pose against a real scene."""
    result = preview_pose_in_scene("beach_sunset", "power_stance")
    assert result["scene"]["context"] == "beach"
    assert result["pose"]["id"] == "power_stance"
    assert result["placement"]["fits_safe_margin"] is True
    assert "placement_box" in result["placement"]


def test_preview_studio_gray_with_hands_on_hips() -> None:
    """Studio scene with a known studio-friendly pose."""
    result = preview_pose_in_scene("studio_gray", "hands_on_hips")
    assert result["scene"]["context"] == "studio"
    assert result["placement"]["horizon_y"] == 1080  # no horizon in studio


def test_preview_urban_wall_with_lean_pose() -> None:
    """Urban scene preview."""
    result = preview_pose_in_scene("urban_wall", "lean_on_rail")
    assert result["scene"]["context"] == "urban"


def test_preview_scene_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Scene"):
        preview_pose_in_scene("nonexistent_scene", "power_stance")


def test_preview_pose_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Pose"):
        preview_pose_in_scene("beach_sunset", "nonexistent_pose")


# ── batch recommendation ──────────────────────────────────────────────────


def test_recommend_placement_for_scene() -> None:
    results = recommend_placement_for_scene("beach_sunset", top_k=3)
    assert len(results) >= 1
    assert len(results) <= 3
    for r in results:
        assert "placement" in r
        assert "score" in r
        assert 0 <= r["score"] <= 1


def test_recommend_placement_sorted_by_score() -> None:
    results = recommend_placement_for_scene("studio_gray", top_k=5)
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


# ── CLI preview ───────────────────────────────────────────────────────────


def test_cli_preview_output(capsys) -> None:
    """CLI preview should print to stdout without crashing."""
    cli_preview("beach_sunset", "power_stance")
    captured = capsys.readouterr()
    assert "Scene:" in captured.out
    assert "Pose:" in captured.out
    assert "Placement:" in captured.out


def test_cli_preview_missing_scene(capsys) -> None:
    cli_preview("nope", "power_stance")
    captured = capsys.readouterr()
    assert "Error:" in captured.out
