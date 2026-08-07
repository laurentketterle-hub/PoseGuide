# Implementation #17: End-to-End Product Path

## Overview

This document describes the end-to-end product path for PoseGuide covering:
1. **Photo background processing** — isolate or remove the background from user photos
2. **Pose coach** — analyze and provide feedback on user poses
3. **Overlay export** — generate and export overlay visualizations

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Photo Input    │────▶│  Background     │────▶│  Pose Coach     │
│  (User Upload)  │     │  Processor      │     │  Analyzer       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Overlay Export │
                                               │  Generator      │
                                               └─────────────────┘
```

## Workflow

### 1. Photo Background Processing
- Accept user-uploaded photos (JPEG, PNG, WebP)
- Remove or isolate background using segmentation models
- Output: foreground-only image with transparent background

### 2. Pose Coach Analysis
- Detect human pose keypoints from the foreground image
- Compare against reference/target pose
- Generate coaching feedback (angles, alignment, positioning)

### 3. Overlay Export
- Render pose skeleton overlay on original image
- Highlight deviations from target pose
- Export as annotated image (PNG with overlay)

## CI/CD

See `.github/workflows/ci-17.yml` for automated validation of this path.

## References

- Issue: [#17](https://github.com/mergeos-bounties/PoseGuide/issues/17)
- Bounty: 200 MRG
