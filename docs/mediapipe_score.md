# MediaPipe Subject Score Path

## Overview

The MediaPipe subject score path provides an optional vision-based scoring
system for pose evaluation. When MediaPipe is available (via the 
`mediapipe_model_path` config), it uses a TensorFlow Lite pose landmark model
to extract 33 landmarks and compute a subject score based on pose quality.

## Architecture

```
                    ┌──────────────────┐
                    │   Pose JSON      │
                    │  (joints+tips)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  ScorerConfig    │
                    │  use_mediapipe?  │
                    └───┬──────────┬───┘
                        │ false    │ true
               ┌────────▼───┐  ┌──▼──────────┐
               │ Toy Ranker │  │ MediaPipe    │
               │ (default)  │  │ TFLite Model │
               └────────┬───┘  └──┬──────────┘
                        │          │
                        │     ┌────▼────┐
                        │     │ Model   │
                        │     │ missing?│──yes──┐
                        │     └────┬────┘       │
                        │          │ no         │
                        │     ┌────▼────┐  ┌────▼──────────┐
                        │     │ Extract  │  │ Fallback to   │
                        │     │ 33 LM    │  │ Toy Ranker    │
                        │     └────┬────┘  └────┬──────────┘
                        │          │            │
                        └──────────┼────────────┘
                                   │
                          ┌────────▼────────┐
                          │  SubjectScore   │
                          │  .score: float  │
                          │  .method: str   │
                          └─────────────────┘
```

## Usage

```python
from poseguide.scorer import score_subject, ScorerConfig

# Default: toy ranker (no MediaPipe dependency)
config = ScorerConfig(use_mediapipe=False)
result = score_subject(pose_data, config=config)
print(f"Score: {result.score}, Method: {result.method}")

# With MediaPipe (requires model file)
config = ScorerConfig(
    use_mediapipe=True,
    mediapipe_model_path="./models/pose_landmarker_lite.task"
)
result = score_subject(pose_data, config=config)
# Falls back to toy ranker if model is unavailable
```

## Fallback Behavior

- When `use_mediapipe=False`: always uses toy ranker
- When `use_mediapipe=True` and model exists: uses MediaPipe landmarks
- When `use_mediapipe=True` and model is missing/unavailable: falls back to toy ranker with `method="mediapipe_fallback"`

## Toy Ranker

The toy ranker provides deterministic scoring without external dependencies:
- Scores range from 0.0 to 1.0
- Based on joint symmetry and tip placement heuristics
- Same input always produces the same score (no randomness)

## Testing

```bash
pytest tests/test_mediapipe_score.py -v
```

Tests cover:
- Toy ranker as default
- MediaPipe path with model unavailable fallback
- Edge cases (empty poses, unknown joints)
- Score range validation [0.0, 1.0]
