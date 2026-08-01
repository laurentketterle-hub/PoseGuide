"""Tests for Vietnamese tips."""
from src.poseguide.vietnamese_tips import get_tips_for_pose_family, get_all_tips, VIETNAMESE_TIPS

def test_get_tips_for_family():
    tips = get_tips_for_pose_family("standing")
    assert len(tips) == 4
    assert all(isinstance(t, str) for t in tips)

def test_get_all_tips():
    all_tips = get_all_tips()
    assert "standing" in all_tips
    assert "portrait" in all_tips

def test_tips_structure():
    for family, tips in VIETNAMESE_TIPS.items():
        assert isinstance(family, str)
        assert isinstance(tips, list)
        assert len(tips) >= 2
