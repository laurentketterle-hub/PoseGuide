"""Public pose/photography dataset index loader (#20)."""

from __future__ import annotations

DATASETS = [
    {"name":"COCO 2017 Keypoints","domain":"General/Person","license":"CC BY 4.0","keypoints":"17 keypoints","notes":"250k person instances"},
    {"name":"MPII Human Pose","domain":"Human Pose","license":"BSD-like academic","keypoints":"16 keypoints","notes":"25k images"},
    {"name":"Leeds Sports Pose","domain":"Sports","license":"Custom research","keypoints":"14 keypoints","notes":"2k sport images"},
    {"name":"Flickr30k Entities","domain":"Scene+Person","license":"CC BY","keypoints":"captions","notes":"Image-caption pairs"},
    {"name":"Places365","domain":"Scenes","license":"CC BY","keypoints":"scene labels","notes":"365 categories, 1.8M images"},
    {"name":"Open Images V7","domain":"General","license":"CC BY 4.0","keypoints":"17 COCO","notes":"9M images"},
    {"name":"AI Challenger","domain":"Human Keypoints","license":"Free research","keypoints":"14 keypoints","notes":"380k images"},
    {"name":"Yoga-82","domain":"Pose Classification","license":"MIT","keypoints":"pose labels","notes":"82 yoga classes, 28k images"},
    {"name":"FashionPose","domain":"Fashion/Ecom","license":"CC BY","keypoints":"17 COCO","notes":"400k images"},
    {"name":"CrowdPose","domain":"Crowded Scenes","license":"Academic","keypoints":"14 keypoints","notes":"Multi-person"},
]

def list_datasets(*, domain=None):
    if domain is None: return list(DATASETS)
    q = domain.lower()
    return [d for d in DATASETS if q in d["domain"].lower()]
