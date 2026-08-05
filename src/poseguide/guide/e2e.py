"""End-to-end product path: image in → scene tags → pose list → coach → overlay out.

Issue #17 — single CLI command covering the full coach pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from poseguide.config import OUT_DIR
from poseguide.data.extract import extract_pose
from poseguide.guide.composition import coach_bundle
from poseguide.guide.demo import PRESETS
from poseguide.guide.recommend import recommend_for_tags
from poseguide.render.overlay import (
    VisionUnavailableError,
    render_overlay_png,
    write_guidance_overlay,
)

logger = logging.getLogger("poseguide.e2e")

# License-safe demo image — generated programmatically, no external dependencies.
# When no --image is provided we synthesise a simple checkerboard placeholder
# so the pipeline exercises every stage (tags → recommend → coach → overlay).
_DEMO_IMAGE_NAME = "_e2e_demo_checkerboard.png"


def _ensure_demo_image(work_dir: Path) -> Path:
    """Create a tiny license-safe checkerboard PNG so the e2e path always has an image."""
    path = work_dir / _DEMO_IMAGE_NAME
    if path.exists():
        return path

    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover — vision extra not available
        # Write a 1px PNG header as minimal placeholder.
        # This lets the pipeline complete even without Pillow.
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
            b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return path

    h, w = 120, 160
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    sq = 20
    arr[0:: sq * 2, 0:: sq * 2] = 48
    arr[sq:: sq * 2, sq:: sq * 2] = 48
    img = Image.fromarray(arr, mode="RGB")
    img.save(str(path))
    return path


def _make_run_dir(tags_text: str) -> Path:
    """Create a timestamped output directory for one e2e run."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = tags_text.replace(",", "_").replace(" ", "").strip("_")[:40] or "adhoc"
    run_dir = OUT_DIR / f"e2e_{slug}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_e2e(
    tags: str,
    *,
    image: Optional[Path] = None,
    top_k: int = 3,
    subject_json: Optional[Path] = None,
    render_png: bool = True,
) -> dict:
    """Execute the full E2E product path and return a summary dict.

    Parameters
    ----------
    tags:
        Comma-separated scene tags (e.g. ``"beach,outdoor,portrait"``) or a
        preset name from :data:`poseguide.guide.demo.PRESETS`.
    image:
        Optional photo to extract subject joints from (MediaPipe, needs the
        ``vision`` extra).  When omitted a license-safe checkerboard demo image
        is generated so the pipeline always exercises every stage.
    top_k:
        Number of top-ranked poses to include in recommendations.
    subject_json:
        Optional pre-extracted subject JSON (bypasses MediaPipe extraction).
    render_png:
        When ``True`` (default) attempt PNG skeleton overlays.  Falls back
        gracefully to JSON-only when the ``vision`` extra is not installed.

    Returns
    -------
    dict
        Summary with ``tags``, ``recommendations``, ``coach``, and per-pose
        ``artifacts`` (JSON overlay, SVG, optional PNG).
    """
    # --- resolve tags -----------------------------------------------------------
    key = tags.strip().lower()
    tag_str = PRESETS.get(key, tags)
    if "," not in tag_str:
        # Could be a single bare word — treat as preset name or literal
        tag_str = PRESETS.get(key, key)

    # --- set up run directory ---------------------------------------------------
    run_dir = _make_run_dir(tag_str)
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        log_lines.append(msg)

    _log(f"e2e start  tags={tag_str}  run_dir={run_dir}")

    # --- image / subject --------------------------------------------------------
    resolved_image: Optional[Path] = None
    subject_payload: Optional[dict] = None
    subject_json_path: Optional[Path] = None

    if subject_json is not None:
        subject_payload = json.loads(subject_json.read_text(encoding="utf-8"))
        subject_json_path = run_dir / "subject.json"
        subject_json_path.write_text(
            json.dumps(subject_payload, indent=2) + "\n", encoding="utf-8"
        )
        _log(f"subject  from={subject_json}")
    elif image is not None and image.exists():
        resolved_image = image
        try:
            subject_payload = extract_pose(image, subject_id=image.stem)
            subject_json_path = run_dir / "subject.json"
            subject_json_path.write_text(
                json.dumps(subject_payload, indent=2) + "\n", encoding="utf-8"
            )
            _log(f"extract  joints={len(subject_payload.get('joints',{}))}  from={image}")
        except (RuntimeError, FileNotFoundError) as exc:
            _log(f"extract  SKIP ({exc})")
    else:
        # No image provided — generate demo placeholder
        resolved_image = _ensure_demo_image(run_dir)
        _log(f"image  demo_placeholder={resolved_image.name}")
        try:
            subject_payload = extract_pose(resolved_image, subject_id="demo")
            subject_json_path = run_dir / "subject.json"
            subject_json_path.write_text(
                json.dumps(subject_payload, indent=2) + "\n", encoding="utf-8"
            )
            _log("extract  joints from demo placeholder")
        except (RuntimeError, FileNotFoundError) as exc:
            _log(f"extract  SKIP demo ({exc})")

    # --- recommend --------------------------------------------------------------
    result = recommend_for_tags(tag_str, top_k=top_k)
    recs = result.get("recommendations", [])
    _log(f"recommend  top={len(recs)}  tags={result.get('scene_tags')}")

    # --- coach & overlay per pose -----------------------------------------------
    coach_results: list[dict] = []
    artifacts: list[dict] = []

    for rec in recs:
        pose_id = str(rec["pose_id"])
        pose_name = str(rec.get("name", pose_id))

        # coach bundle (composition + SVG)
        try:
            coach = coach_bundle(pose_id, subject_path=subject_json_path)
            coach_results.append(
                {
                    "pose_id": pose_id,
                    "name": pose_name,
                    "score": rec.get("score"),
                    "coach": coach,
                }
            )
            _log(f"coach  {pose_id}  tips={len(coach.get('composition',{}).get('tips',[]))}")
        except KeyError:
            _log(f"coach  {pose_id}  SKIP (unknown pose)")

        # JSON overlay (always)
        overlay_json = run_dir / f"overlay_{pose_id}.json"
        overlay_path = write_guidance_overlay(
            {"scene_tags": result.get("scene_tags"), "recommendations": [rec]},
            overlay_json,
        )

        artifact: dict = {
            "pose_id": pose_id,
            "name": pose_name,
            "overlay_json": str(overlay_path),
        }

        # SVG
        svg_path = run_dir / f"coach_{pose_id}.svg"
        if svg_path.exists():
            artifact["svg"] = str(svg_path)

        # PNG overlay (best-effort)
        if render_png:
            png_out = run_dir / f"overlay_{pose_id}.png"
            try:
                png_path = render_overlay_png(
                    pose_id,
                    png_out,
                    subject_joints=subject_payload.get("joints") if subject_payload else None,
                    background=resolved_image,
                    width=360,
                    height=480,
                )
                artifact["overlay_png"] = str(png_path)
                _log(f"overlay  png={png_path}")
            except (VisionUnavailableError, KeyError, OSError):
                _log(f"overlay  {pose_id}  PNG SKIP (vision unavailable)")

        artifacts.append(artifact)

    # --- summary ----------------------------------------------------------------
    run_log = run_dir / "e2e.log"
    run_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    summary = {
        "kind": "poseguide.e2e.v1",
        "run_dir": str(run_dir),
        "tags": tag_str,
        "top_k": top_k,
        "recommendations": recs,
        "coach": coach_results,
        "artifacts": artifacts,
        "log": str(run_log),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _log(f"done  summary={summary_path}")
    return summary
