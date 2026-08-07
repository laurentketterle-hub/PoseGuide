"""Training pipeline: YAML config, seeds, checkpoint resume.

Public API:
    from poseguide.train.trainer import train, load_config, set_seed, main
"""

from poseguide.train.trainer import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    load_config,
    main,
    save_checkpoint,
    set_seed,
    train,
)

__all__ = [
    "cleanup_old_checkpoints",
    "find_latest_checkpoint",
    "load_checkpoint",
    "load_config",
    "main",
    "save_checkpoint",
    "set_seed",
    "train",
]
