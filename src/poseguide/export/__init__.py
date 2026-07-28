"""Export module for COCO and MediaPipe formats."""
import json
from pathlib import Path

def export_coco(poses, image_size, out_path):
    keypoints = []
    for pose in poses:
        kp = []
        for joint in pose.get("joints", []):
            kp.extend([joint["x"], joint["y"], joint.get("confidence", 1.0)])
        keypoints.append(kp)
    coco = {"images": [{"id": 0, "width": image_size[0], "height": image_size[1]}],
            "annotations": [{"id": i, "image_id": 0, "keypoints": kp, "num_keypoints": len(kp)//3} for i, kp in enumerate(keypoints)],
            "categories": [{"id": 0, "name": "person"}]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    return out_path

def export_mediapipe(poses, out_path):
    landmarks = []
    for pose in poses:
        for joint in pose.get("joints", []):
            landmarks.append({"x": joint["x"], "y": joint["y"], "z": joint.get("z", 0), "visibility": joint.get("confidence", 1.0)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"landmarks": landmarks}, indent=2))
    return out_path


def batch_export(poses_list, output_dir, format='coco', image_sizes=None):
    results = []
    for i, poses in enumerate(poses_list):
        size = image_sizes[i] if image_sizes else (640, 480)
        name = f'pose_{i:04d}'
        if format == 'coco':
            out = output_dir / f'{name}.json'
            results.append(export_coco(poses, size, out))
        elif format == 'mediapipe':
            out = output_dir / f'{name}_mp.json'
            results.append(export_mediapipe(poses, out))
    return results

def export_all_formats(poses, img_size, output_dir, prefix='pose'):
    results = {}
    results['coco'] = export_coco(poses, img_size, output_dir / f'{prefix}_coco.json')
    results['mediapipe'] = export_mediapipe(poses, output_dir / f'{prefix}_mp.json')
    return results
