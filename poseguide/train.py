"""
PoseGuide training pipeline runner.
Loads YAML config, sets seeds, runs training with checkpoint resume.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


class TrainingPipeline:
    """Reproducible training pipeline with checkpoint resume support."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._set_seeds()
        self.device = self._setup_device()
        self.checkpoint_dir = Path(self.config.get("checkpoint", {}).get("save_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, path: str) -> dict:
        """Load YAML or JSON config."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
            return yaml.safe_load(content)
        elif path.suffix == ".json":
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        seeds = self.config.get("seeds", {})
        random.seed(seeds.get("python", 42))
        
        try:
            import numpy as np
            np.random.seed(seeds.get("numpy", 42))
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seeds.get("torch", 42))
            if torch.cuda.is_available() and seeds.get("cuda_deterministic", True):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def _setup_device(self) -> str:
        """Detect and return the best available device."""
        hw = self.config.get("hardware", {})
        preferred = hw.get("device", "cuda")
        
        if preferred == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        elif preferred == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return "cpu"
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint for resume."""
        resume_from = self.config.get("checkpoint", {}).get("resume_from")
        if resume_from:
            p = Path(resume_from)
            if p.exists():
                return p
        
        # Auto-detect latest checkpoint
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            return ckpts[0]
        return None
    
    def run(self, data_dir: str):
        """Execute the training pipeline."""
        print(f"PoseGuide Training Pipeline v{self.config['project']['version']}")
        print(f"Config: {self.config['project']['name']}")
        print(f"Device: {self.device}")
        print(f"Data dir: {data_dir}")
        
        ckpt = self.find_latest_checkpoint()
        if ckpt:
            print(f"Resuming from checkpoint: {ckpt}")
        else:
            print("Starting fresh training")
        
        data_cfg = self.config.get("data", {})
        print(f"Batch size: {data_cfg.get('batch_size', 32)}")
        print(f"Image size: {data_cfg.get('image_size', [256, 256])}")
        print(f"Splits: train={data_cfg.get('train_split', 0.7)}, "
              f"val={data_cfg.get('val_split', 0.15)}, "
              f"test={data_cfg.get('test_split', 0.15)}")
        
        # TODO: Actual training loop with model, optimizer, scheduler
        print("\nTraining pipeline scaffold ready — model integration pending dataset")
        print("Seeds set, device configured, checkpoint resume supported.")


def main():
    parser = argparse.ArgumentParser(description="PoseGuide Training Pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to training config YAML/JSON")
    parser.add_argument("--data-dir", "-d", required=True, help="Path to dataset directory")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(args.config)
    pipeline.run(args.data_dir)


if __name__ == "__main__":
    main()
