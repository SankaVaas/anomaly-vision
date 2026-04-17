"""
Shared utilities — logging, config loading, device setup, reproducibility.
Kept minimal: only helpers used by more than one module live here.
"""

import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


# ─── 1. REPRODUCIBILITY ──────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Fix all random sources so runs are reproducible across restarts.
    Critical for coreset subsampling — random init point affects the bank.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # slightly slower, fully deterministic


# ─── 2. CONFIG ───────────────────────────────────────────────────────────────

def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config. Supports path override from notebook or CLI."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """
    Shallow-merge overrides into cfg at any nesting level.
    Lets the notebook change a single key without rewriting the whole dict.

    Example:
        cfg = override_config(cfg, {"data": {"category": "carpet"},
                                    "patchcore": {"k_neighbors": 5}})
    """
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


# ─── 3. DEVICE ───────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    """
    Resolves device from config, falls back to CPU cleanly.
    Prints a warning if CUDA was requested but unavailable —
    useful on Colab when the T4 runtime hasn't been selected yet.
    """
    requested = cfg["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        print("  ⚠  CUDA requested but not available — falling back to CPU.")
        print("     In Colab: Runtime → Change runtime type → T4 GPU")
        return torch.device("cpu")
    device = torch.device(requested)
    print(f"  Device : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device


# ─── 4. DIRECTORY SETUP ──────────────────────────────────────────────────────

def setup_output_dirs(cfg: dict):
    """Create output directories from config if they don't exist."""
    for key in ["checkpoint_dir", "results_dir"]:
        Path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)


# ─── 5. LOGGING ──────────────────────────────────────────────────────────────

class Logger:
    """
    Minimal file + console logger — no external dependency.
    Writes every print() equivalent to both stdout and a log file.
    """

    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "a")

    def log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def __del__(self):
        try:
            self.log_file.close()
        except Exception:
            pass


# ─── 6. NORMALISATION HELPERS ────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Converts a normalised image tensor [3, H, W] back to
    a uint8 numpy array [H, W, 3] for display / saving.
    """
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t


# ─── 7. SCORE NORMALISATION ──────────────────────────────────────────────────

def normalize_scores(scores: list) -> list:
    """
    Min-max normalise anomaly scores to [0, 1].
    Makes image-level scores comparable across categories
    and thresholds interpretable as percentages.
    """
    arr = np.array(scores, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return [0.0] * len(scores)
    return ((arr - mn) / (mx - mn)).tolist()