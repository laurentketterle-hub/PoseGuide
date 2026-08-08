"""
Batch SVG renderer for pose packs.
Renders pose joint skeletons as SVG overlays for documentation and preview.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from xml.etree import ElementTree as ET


@dataclass
class PoseJoint:
    id: int
    name: str
    x: float
    y: float
    visibility: float = 1.0


@dataclass 
class PoseConnection:
    joint_a: int
    joint_b: int
    label: str = ""


class SVGPoseRenderer:
    """Renders pose skeletons as SVG files."""
    
    JOINT_RADIUS = 4.0
    BONE_WIDTH = 2.0
    LABEL_OFFSET = 10
    
    # Standard MediaPipe pose connections
    STANDARD_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),     # left arm
        (0, 4), (4, 5), (5, 6), (6, 8),     # right arm
        (9, 10),                              # shoulders
        (11, 12), (11, 13), (13, 15),        # left leg
        (12, 14), (14, 16),                   # right leg
        (11, 23), (12, 24), (23, 24),        # torso
    ]
    
    COLORS = {
        "joint": "#FF6B35",
        "bone": "#004E89",
        "label": "#333333",
        "background": "#FFFFFF",
    }
    
    def __init__(self, width: int = 400, height: int = 600, padding: int = 40):
        self.width = width
        self.height = height
        self.padding = padding
    
    def render_single(self, joints: List[PoseJoint],
                      connections: Optional[List[Tuple[int, int]]] = None,
                      title: str = "") -> str:
        """Render a single pose as SVG string."""
        if connections is None:
            connections = self.STANDARD_CONNECTIONS
        
        svg = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {self.width} {self.height}",
            "width": str(self.width),
            "height": str(self.height),
        })
        
        # Background
        ET.SubElement(svg, "rect", {
            "width": "100%", "height": "100%",
            "fill": self.COLORS["background"],
        })
        
        # Connections (bones)
        for a, b in connections:
            ja = next((j for j in joints if j.id == a), None)
            jb = next((j for j in joints if j.id == b), None)
            if ja and jb and ja.visibility > 0.1 and jb.visibility > 0.1:
                ET.SubElement(svg, "line", {
                    "x1": str(ja.x), "y1": str(ja.y),
                    "x2": str(jb.x), "y2": str(jb.y),
                    "stroke": self.COLORS["bone"],
                    "stroke-width": str(self.BONE_WIDTH),
                    "stroke-linecap": "round",
                })
        
        # Joints
        for j in joints:
            if j.visibility > 0.1:
                ET.SubElement(svg, "circle", {
                    "cx": str(j.x), "cy": str(j.y),
                    "r": str(self.JOINT_RADIUS),
                    "fill": self.COLORS["joint"],
                    "opacity": str(j.visibility),
                })
                # Label
                label = ET.SubElement(svg, "text", {
                    "x": str(j.x + self.LABEL_OFFSET),
                    "y": str(j.y - self.LABEL_OFFSET),
                    "fill": self.COLORS["label"],
                    "font-size": "10",
                    "font-family": "sans-serif",
                })
                label.text = j.name
        
        # Title
        if title:
            title_el = ET.SubElement(svg, "text", {
                "x": str(self.width // 2),
                "y": "20",
                "text-anchor": "middle",
                "fill": self.COLORS["label"],
                "font-size": "14",
                "font-weight": "bold",
                "font-family": "sans-serif",
            })
            title_el.text = title
        
        ET.indent(svg)
        return ET.tostring(svg, encoding="unicode")
    
    def render_batch(self, pose_pack_path: str, output_dir: str) -> List[str]:
        """Render all poses in a pose pack directory to SVG files."""
        pack_path = Path(pose_pack_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        rendered = []
        
        for json_file in sorted(pack_path.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                joints = [
                    PoseJoint(
                        id=p["id"],
                        name=p.get("name", str(p["id"])),
                        x=p["x"],
                        y=p["y"],
                        visibility=p.get("visibility", 1.0),
                    )
                    for p in data.get("joints", [])
                ]
                
                connections = data.get("connections")
                title = data.get("name", json_file.stem)
                
                svg_content = self.render_single(joints, connections, title)
                
                out_file = out_path / f"{json_file.stem}.svg"
                out_file.write_text(svg_content)
                rendered.append(str(out_file))
                
            except Exception as e:
                print(f"  Warning: Skipping {json_file}: {e}")
        
        return rendered


def batch_render_pose_pack(pack_dir: str, output_dir: str,
                           width: int = 400, height: int = 600) -> List[str]:
    """CLI-friendly batch render entry point."""
    renderer = SVGPoseRenderer(width=width, height=height)
    return renderer.render_batch(pack_dir, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch SVG renderer for PoseGuide pose packs")
    parser.add_argument("--pack-dir", "-p", required=True, help="Directory containing pose JSON files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for SVG files")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()
    
    rendered = batch_render_pose_pack(args.pack_dir, args.output, args.width, args.height)
    print(f"Rendered {len(rendered)} SVGs to {args.output}:")
    for f in rendered:
        print(f"  {f}")
