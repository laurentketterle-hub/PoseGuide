# Batch SVG Render for Pose Packs

Render all poses in a pose pack directory to SVG files via CLI.

## Quick Start

```bash
# Render all JSON pose packs to out/svg/
python scripts/render_svg.py poses/ -o out/svg/

# With a custom file pattern
python scripts/render_svg.py poses/ -o out/svg/ -p "*.json"
```

## Output

Each pose pack gets its own subdirectory under the output path:

```
out/svg/
├── pack_name/
│   ├── pose_1.svg
│   └── pose_2.svg
└── another_pack/
    └── pose_a.svg
```

## SVG Format

Each SVG includes:
- Dark background (#1a1a2e)
- Pose name and description header
- Joint circles with coordinates, radius, and color
- Connection lines between joints with stroke styling

## Programmatic API

```python
from poseguide.svg_render import batch_render, render_pose_pack, pose_to_svg

# Render a single pack
render_pose_pack(Path("poses/pack.json"), Path("out/svg/"))

# Batch render all packs
results = batch_render("poses/", "out/svg/")
print(results["rendered"])  # list of rendered packs with counts
```

## Smoke Test

```bash
python -m pytest tests/test_svg_render.py -v
```
