# Changelog

## [Unreleased] - MediaPipe Score Path

### Added
- Optional MediaPipe-based subject score path with `ScorerConfig.use_mediapipe`
- Graceful fallback to toy ranker when MediaPipe model is unavailable
- `SubjectScore` dataclass with `score` and `method` fields
- Comprehensive test suite in `tests/test_mediapipe_score.py`
- Architecture documentation in `docs/mediapipe_score.md`

### Changed
- `score_subject()` now accepts optional `config: ScorerConfig` parameter
- Toy ranker remains default when no config is provided (backward compatible)

### Fixed
- Toy ranker deterministic scoring guarantee
