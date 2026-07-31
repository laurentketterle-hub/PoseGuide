"""Tests for SVG render module."""
import json
import os
import tempfile
import pytest
from pathlib import Path
from poseguide.svg_render import pose_to_svg, render_pose_pack, batch_render


SAMPLE_POSE = {
    "name": "test_pose",
    "description": "A test pose",
    "joints": [
        {"name": "head", "x": 150, "y": 100, "radius": 8, "color": "#e94560"},
        {"name": "shoulder_l", "x": 100, "y": 180, "radius": 6, "color": "#0f3460"},
        {"name": "shoulder_r", "x": 200, "y": 180, "radius": 6, "color": "#0f3460"},
        {"name": "hip_l", "x": 110, "y": 300, "radius": 6, "color": "#533483"},
        {"name": "hip_r", "x": 190, "y": 300, "radius": 6, "color": "#533483"},
    ],
    "connections": [
        ["head", "shoulder_l"],
        ["head", "shoulder_r"],
        ["shoulder_l", "hip_l"],
        ["shoulder_r", "hip_r"],
        ["shoulder_l", "shoulder_r"],
        ["hip_l", "hip_r"],
    ]
}


class TestPoseToSvg:
    
    def test_basic_svg(self):
        svg = pose_to_svg(SAMPLE_POSE)
        assert svg.startswith('<svg')
        assert svg.endswith('</svg>')
        assert 'test_pose' in svg
        assert '<circle' in svg
        assert '<line' in svg
    
    def test_svg_dimensions(self):
        svg = pose_to_svg(SAMPLE_POSE, width=400, height=600)
        assert 'viewBox="0 0 400 600"' in svg
        assert 'width="400"' in svg
    
    def test_empty_pose(self):
        svg = pose_to_svg({"name": "empty"})
        assert '<svg' in svg
        assert 'empty' in svg


class TestRenderPosePack:
    
    def test_render_single_pose(self):
        pack = {"poses": [SAMPLE_POSE]}
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = Path(tmp) / "test_pack.json"
            with open(pack_path, 'w') as f:
                json.dump(pack, f)
            
            out_dir = Path(tmp) / "out"
            rendered = render_pose_pack(pack_path, out_dir)
            
            assert len(rendered) == 1
            assert rendered[0].endswith('.svg')
            assert os.path.exists(rendered[0])
    
    def test_render_multiple_poses(self):
        pack = {"poses": [
            {"name": "pose_a", "joints": [{"name": "j1", "x": 100, "y": 100}]},
            {"name": "pose_b", "joints": [{"name": "j1", "x": 200, "y": 200}]},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = Path(tmp) / "multi.json"
            with open(pack_path, 'w') as f:
                json.dump(pack, f)
            
            out_dir = Path(tmp) / "out"
            rendered = render_pose_pack(pack_path, out_dir)
            
            assert len(rendered) == 2


class TestBatchRender:
    
    def test_batch_directory(self):
        packs = {
            "pack_a.json": {"poses": [{"name": "a1", "joints": [{"name": "j", "x": 100, "y": 100}]}]},
            "pack_b.json": {"poses": [{"name": "b1", "joints": [{"name": "j", "x": 150, "y": 150}]}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            pose_dir = Path(tmp) / "poses"
            pose_dir.mkdir()
            for name, content in packs.items():
                with open(pose_dir / name, 'w') as f:
                    json.dump(content, f)
            
            out_dir = Path(tmp) / "out"
            results = batch_render(str(pose_dir), str(out_dir))
            
            assert results["total_files"] == 2
            assert len(results["rendered"]) == 2
            for r in results["rendered"]:
                assert r["count"] == 1
