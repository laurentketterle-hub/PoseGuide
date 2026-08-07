"""Production training loop with YAML config, seeds, and checkpoint resume."""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from poseguide.config import RUNS_DIR
from poseguide.data.loader import list_scene_files, load_scene
from poseguide.models.toy import ToyPoseRanker


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load training configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    history: list[dict],
    state: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint to disk."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ckpt_dir / f"ckpt_epoch{epoch:03d}_{ts}.json"
    payload = {
        "epoch": epoch,
        "history": history,
        "state": state or {},
        "saved_at": ts,
    }
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the path of the latest checkpoint, or None."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    files = sorted(ckpt_dir.glob("ckpt_epoch*.json"))
    return files[-1] if files else None


def cleanup_old_checkpoints(run_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints beyond the N most recent."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return
    files = sorted(ckpt_dir.glob("ckpt_epoch*.json"))
    for f in files[:-keep_last_n]:
        f.unlink()


def train(config: dict[str, Any], resume: bool = True) -> dict[str, Any]:
    """Run a full training loop driven by a config dict.

    Supports seed-based reproducibility, epoch checkpoints, and resume from
    the latest checkpoint when *resume* is True.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    train_cfg = config.get("training", {})
    epochs = max(1, train_cfg.get("epochs", 5))

    ckpt_cfg = config.get("checkpoint", {})
    ckpt_enabled = ckpt_cfg.get("enabled", True)
    save_every = max(1, ckpt_cfg.get("save_every_n_epochs", 1))
    keep_last_n = max(1, ckpt_cfg.get("keep_last_n", 3))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = RUNS_DIR / f"train_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    history: list[dict] = []
    ranker = ToyPoseRanker()

    if resume and ckpt_enabled:
        latest = find_latest_checkpoint(run_dir)
        if latest is not None:
            ckpt = load_checkpoint(latest)
            start_epoch = ckpt["epoch"] + 1
            history = ckpt.get("history", [])
            print(f"Resumed from {latest.name} (epoch {ckpt['epoch']})")

    scenes = [load_scene(p) for p in list_scene_files()]
    if not scenes:
        raise FileNotFoundError("No scene files found under data/scenes")

    for epoch in range(start_epoch, start_epoch + epochs):
        hits = 0
        for scene in scenes:
            expected = {str(x).lower() for x in (scene.get("expected_poses") or [])}
            recs = ranker.recommend(scene, top_k=3)
            top_ids = {str(r["pose_id"]).lower() for r in recs}
            if (expected and (top_ids & expected)) or (
                not expected and recs and recs[0]["score"] > 0
            ):
                hits += 1
        acc = hits / len(scenes)
        history.append({
            "epoch": epoch,
            "hit_rate_at_3": round(acc, 4),
            "n": len(scenes),
        })

        if ckpt_enabled and epoch % save_every == 0:
            save_checkpoint(run_dir, epoch, history, {"seed": seed})
            cleanup_old_checkpoints(run_dir, keep_last_n)

    report = {
        "model": config.get("model", {}).get("name", "unknown"),
        "seed": seed,
        "total_epochs": len(history),
        "history": history,
        "n_poses": len(ranker.poses),
        "n_scenes": len(scenes),
    }
    report_path = run_dir / "train_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"report_path": str(report_path), "run_dir": str(run_dir), **report}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: python -m poseguide.train.trainer [--config PATH] [--epochs N] [--no-resume]"""
    args = argv if argv is not None else sys.argv[1:]
    config_path = "configs/train.yaml"
    override_epochs: int | None = None
    no_resume = False

    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]; i += 2
        elif args[i] == "--epochs" and i + 1 < len(args):
            override_epochs = int(args[i + 1]); i += 2
        elif args[i] == "--no-resume":
            no_resume = True; i += 1
        else:
            i += 1

    cfg = load_config(config_path)
    if override_epochs is not None:
        cfg.setdefault("training", {})["epochs"] = override_epochs

    report = train(cfg, resume=not no_resume)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
