# Training Pipeline

Production-ready training loop for the PoseGuide ranker with YAML config,
deterministic seeds, and checkpoint resume.

## Quick start

```bash
# Install with training deps
pip install -e ".[dev]" pyyaml

# Run training with default config
python -m poseguide.train.trainer --config configs/train.yaml

# Override epochs for a quick smoke test
python -m poseguide.train.trainer --config configs/train.yaml --epochs 1 --no-resume
```

## Configuration (configs/train.yaml)

| Key | Default | Description |
|-----|---------|-------------|
| `model.name` | ToyPoseRanker | Model identifier |
| `training.epochs` | 5 | Number of training epochs |
| `seed` | 42 | Random seed for reproducibility |
| `checkpoint.enabled` | true | Enable/disable checkpointing |
| `checkpoint.save_every_n_epochs` | 1 | Checkpoint frequency |
| `checkpoint.keep_last_n` | 3 | Max checkpoints to retain |

## Reproducibility

The trainer sets `random`, `numpy`, and `PYTHONHASHSEED` at the start of
every run. Two runs with the same config, data, and seed produce identical
results.

```python
from poseguide.train.trainer import set_seed, train, load_config

cfg = load_config("configs/train.yaml")
cfg["seed"] = 123
set_seed(123)
report = train(cfg, resume=False)
```

## Checkpoint resume

Checkpoints save after every epoch by default. If training is interrupted,
re-run the same command — the trainer auto-detects the latest checkpoint and
resumes from the next epoch.

```python
from poseguide.train.trainer import find_latest_checkpoint, load_checkpoint

latest = find_latest_checkpoint(run_dir)
if latest:
    ckpt = load_checkpoint(latest)
    print(f"Can resume from epoch {ckpt['epoch'] + 1}")
```

Use `--no-resume` to start fresh, ignoring existing checkpoints.

## Output

- `data/runs/train_<timestamp>/checkpoints/` — epoch checkpoints (JSON)
- `data/runs/train_<timestamp>/train_metrics.json` — final report

## Testing

```bash
pytest tests/test_train.py -v
```
