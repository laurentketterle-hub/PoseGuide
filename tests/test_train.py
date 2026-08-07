from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poseguide.train import toy_train as toy_train_mod
from poseguide.train.toy_train import train_toy
from poseguide.train.trainer import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    train,
)


# ── Legacy toy train test ──────────────────────────────────────────────

def test_train_toy_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(toy_train_mod, "RUNS_DIR", tmp_path / "runs")
    report = train_toy(epochs=2)
    assert report["history"][-1]["hit_rate_at_3"] >= 0.5
    assert Path(report["report_path"]).exists()


# ── Config loading ─────────────────────────────────────────────────────

def test_load_yaml_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(
        "model:\n  name: TestModel\ntraining:\n  epochs: 3\nseed: 99\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg["model"]["name"] == "TestModel"
    assert cfg["training"]["epochs"] == 3
    assert cfg["seed"] == 99


# ── Seed reproducibility ────────────────────────────────────────────────

def test_set_seed_reproducible() -> None:
    set_seed(42)
    a = np.random.randn(5).tolist()
    set_seed(42)
    b = np.random.randn(5).tolist()
    assert a == b


def test_train_seed_deterministic(tmp_path: Path, monkeypatch) -> None:
    import poseguide.train.trainer as tmod
    monkeypatch.setattr(tmod, "RUNS_DIR", tmp_path / "runs")
    cfg = {"seed": 7, "training": {"epochs": 2}, "checkpoint": {"enabled": False}}
    r1 = train(cfg, resume=False)
    r2 = train(cfg, resume=False)
    assert r1["history"] == r2["history"]


# ── Checkpoint save / load / resume ─────────────────────────────────────

def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    history = [{"epoch": 1, "hit_rate_at_3": 0.8, "n": 3}]
    ckpt_path = save_checkpoint(run_dir, epoch=1, history=history, state={"seed": 1})
    assert ckpt_path.exists()

    loaded = load_checkpoint(ckpt_path)
    assert loaded["epoch"] == 1
    assert loaded["history"] == history
    assert loaded["state"]["seed"] == 1


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    save_checkpoint(run_dir, epoch=1, history=[], state={})
    save_checkpoint(run_dir, epoch=2, history=[], state={})
    latest = find_latest_checkpoint(run_dir)
    assert latest is not None
    assert "epoch002" in latest.name


def test_cleanup_old_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for ep in range(1, 6):
        save_checkpoint(run_dir, epoch=ep, history=[], state={})
    ckpt_dir = run_dir / "checkpoints"
    assert len(list(ckpt_dir.glob("ckpt_epoch*.json"))) == 5
    cleanup_old_checkpoints(run_dir, keep_last_n=2)
    remaining = sorted(ckpt_dir.glob("ckpt_epoch*.json"))
    assert len(remaining) == 2
    names = [p.name for p in remaining]
    assert any("epoch004" in n for n in names)
    assert any("epoch005" in n for n in names)


def test_find_latest_checkpoint_none(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "nonexistent") is None


def test_train_checkpoint_resume(tmp_path: Path, monkeypatch) -> None:
    import poseguide.train.trainer as tmod
    monkeypatch.setattr(tmod, "RUNS_DIR", tmp_path / "runs")

    cfg = {
        "seed": 42,
        "training": {"epochs": 2},
        "checkpoint": {"enabled": True, "save_every_n_epochs": 1, "keep_last_n": 5},
    }

    r1 = train(cfg, resume=False)
    assert len(r1["history"]) == 2

    cfg["training"]["epochs"] = 2
    r2 = train(cfg, resume=True)
    assert len(r2["history"]) == 4
    assert r2["history"][:2] == r1["history"]


def test_train_creates_metrics_file(tmp_path: Path, monkeypatch) -> None:
    import poseguide.train.trainer as tmod
    monkeypatch.setattr(tmod, "RUNS_DIR", tmp_path / "runs")

    cfg = {"seed": 1, "training": {"epochs": 1}, "checkpoint": {"enabled": False}}
    report = train(cfg, resume=False)

    metrics_path = Path(report["report_path"])
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["total_epochs"] == 1
    assert "hit_rate_at_3" in data["history"][0]
