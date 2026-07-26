"""Tests for export module."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import pytest

def test_export_coco():
    from poseguide.export import export_coco
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        poses = [{"joints": [{"x": 100, "y": 200, "confidence": 0.95}]}]
        result = export_coco(poses, (640, 480), Path(f.name))
        data = json.loads(result.read_text())
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["keypoints"] == [100, 200, 0.95]

def test_export_mediapipe():
    from poseguide.export import export_mediapipe
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        poses = [{"joints": [{"x": 0.5, "y": 0.3, "z": 0.1, "confidence": 0.99}]}]
        result = export_mediapipe(poses, Path(f.name))
        data = json.loads(result.read_text())
        assert len(data["landmarks"]) == 1
