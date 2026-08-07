"""Batch SVG render for pose packs — CLI command."""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def pose_to_svg(pose: Dict[str, Any], width: int = 300, height: int = 500) -> str:
    """Convert a single pose definition to SVG markup."""
    name = pose.get("name", "pose")
    description = pose.get("description", "")
    joints = pose.get("joints", [])
    connections = pose.get("connections", [])

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'  <rect width="100%" height="100%" fill="#1a1a2e"/>',
        f'  <text x="{width//2}" y="20" text-anchor="middle" fill="#e0e0e0" font-size="12" font-family="sans-serif">{name}</text>',
        f'  <text x="{width//2}" y="35" text-anchor="middle" fill="#888" font-size="9" font-family="sans-serif">{description[:60]}</text>',
    ]

    for conn in connections:
        if len(conn) >= 2:
            j1_name, j2_name = conn[0], conn[1]
            j1 = next((j for j in joints if j.get("name") == j1_name), None)
            j2 = next((j for j in joints if j.get("name") == j2_name), None)
            if j1 and j2:
                x1, y1 = j1.get("x", width // 2), j1.get("y", height // 2)
                x2, y2 = j2.get("x", width // 2), j2.get("y", height // 2)
                svg_parts.append(
                    f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="#666" stroke-width="2" stroke-linecap="round"/>'
                )

    for joint in joints:
        x, y = joint.get("x", width // 2), joint.get("y", height // 2)
        r = joint.get("radius", 5)
        color = joint.get("color", "#e94560")
        label = joint.get("name", "")
        svg_parts.append(
            f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="#fff" stroke-width="1"/>'
        )
        if label:
            svg_parts.append(
                f'  <text x="{x}" y="{y + r + 10}" text-anchor="middle" fill="#aaa" font-size="8" font-family="sans-serif">{label}</text>'
            )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def render_pose_pack(pack_path: Path, output_dir: Path) -> List[str]:
    """Render all poses in a pack to SVG files."""
    rendered = []
    with open(pack_path) as f:
        pack = json.load(f)

    poses = pack.get("poses", [pack]) if isinstance(pack, dict) else pack
    if not isinstance(poses, list):
        poses = [poses]

    pack_name = pack_path.stem
    pack_out = output_dir / pack_name
    pack_out.mkdir(parents=True, exist_ok=True)

    for i, pose in enumerate(poses):
        svg_str = pose_to_svg(pose)
        out_name = f"{pose.get('name', f'pose_{i}')}.svg"
        out_path = pack_out / out_name
        with open(out_path, 'w') as f:
            f.write(svg_str)
        rendered.append(str(out_path))

    return rendered


def batch_render(pose_dir: str, output_dir: str, pattern: str = "*.json") -> Dict[str, Any]:
    """Batch render all pose packs in a directory to SVG."""
    pose_path = Path(pose_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {"rendered": [], "errors": [], "total_files": 0}

    for pack_file in sorted(pose_path.glob(pattern)):
        results["total_files"] += 1
        try:
            rendered = render_pose_pack(pack_file, out_path)
            results["rendered"].append({"pack": pack_file.name, "svgs": rendered, "count": len(rendered)})
        except Exception as e:
            results["errors"].append({"pack": pack_file.name, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch SVG render for pose packs")
    parser.add_argument("pose_dir", help="Directory containing pose pack JSON files")
    parser.add_argument("-o", "--output", default="out/svg", help="Output directory for SVGs")
    parser.add_argument("-p", "--pattern", default="*.json", help="File pattern for pose packs")
    args = parser.parse_args()

    results = batch_render(args.pose_dir, args.output, args.pattern)

    print(f"Rendered {sum(r['count'] for r in results['rendered'])} SVGs "
          f"from {results['total_files']} packs")

    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for e in results["errors"]:
            print(f"  {e['pack']}: {e['error']}")

    return results


if __name__ == "__main__":
    main()
