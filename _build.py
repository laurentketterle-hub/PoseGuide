
import json, os

# docs/POSES.md
md = '# POSES.md - PoseGuide Pose Catalog

81 poses across 15+ families.

See data/poses/ for full catalog.

## Extension Guide
1. Create data/poses/<id>.json
2. Run poseguide poses list
3. Run poseguide poses svg --pose <id>
'
with open('docs/POSES.md', 'w') as f: f.write(md)
print('POSES.md done')
