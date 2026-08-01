"""Tests for webcam coach."""
from src.poseguide.webcam import WebcamCoach, PoseFeedback

def test_coach_creation():
    target = {"name": "test_pose", "joints": {"head": {"x": 0.5, "y": 0.2}}}
    coach = WebcamCoach(target)
    assert coach.target_pose == target

def test_analyze_frame():
    target = {"name": "test_pose", "joints": {"head": {"x": 0.5, "y": 0.2}}}
    coach = WebcamCoach(target, threshold=0.5)
    frame = {"joints": {"head": {"x": 0.5, "y": 0.2}}}
    fb = coach.analyze_frame(frame)
    assert fb.score == 1.0  # Perfect match
    assert fb.pose_name == "test_pose"

def test_analyze_frame_partial():
    target = {"name": "test_pose", "joints": {"head": {"x": 0.5, "y": 0.2}}}
    coach = WebcamCoach(target, threshold=0.5)
    frame = {"joints": {"head": {"x": 0.7, "y": 0.4}}}
    fb = coach.analyze_frame(frame)
    assert fb.score < 1.0  # Not perfect
