"""
Training / fitting entry point.

PatchCore itself is training-free (no gradient updates) —
'training' here means:
  1. One forward pass through all normal images → build memory bank
  2. Run evaluation on test set
  3. Save memory bank + config for inference

This script is also importable so the Colab notebook can call
run() directly without subprocess overhead.
"""

import os
import yaml
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime

from dataset import get_dataloaders
from model   import AnomalyDetector
from evaluate import evaluate, save_anomaly_maps


# ─── 1. CHECKPOINT HELPERS ───────────────────────────────────────────────────

def save_checkpoint(detector: AnomalyDetector, cfg: dict, out_dir: str):
    """
    Saves:
      memory_bank.pkl  — the coreset embedding bank
      config.yaml      — exact config used (reproducibility)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    bank_path = Path(out_dir) / "memory_bank.pkl"
    cfg_path  = Path(out_dir) / "config.yaml"

    with open(bank_path, "wb") as f:
        pickle.dump(detector.memory_bank.bank, f)

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"  Checkpoint saved → {out_dir}")


def load_checkpoint(detector: AnomalyDetector, out_dir: str):
    """Restores memory bank from disk — no retraining needed."""
    bank_path = Path(out_dir) / "memory_bank.pkl"
    if not bank_path.exists():
        raise FileNotFoundError(f"No checkpoint at {bank_path}")

    with open(bank_path, "rb") as f:
        detector.memory_bank.bank = pickle.load(f)

    print(f"  Checkpoint loaded ← {out_dir}")


# ─── 2. COLLECT GT MASKS FROM TEST LOADER ────────────────────────────────────

def _collect_gt(test_loader) -> list:
    """Pull ground-truth masks out of the loader (needed for PRO + pixel AUROC)."""
    masks = []
    for batch in test_loader:
        for i in range(batch["mask"].shape[0]):
            masks.append(batch["mask"][i])
    return masks


def _collect_images(test_loader) -> list:
    """Pull raw images for visualisation."""
    imgs = []
    for batch in test_loader:
        for i in range(batch["image"].shape[0]):
            imgs.append(batch["image"][i])
    return imgs


# ─── 3. MAIN RUN FUNCTION ────────────────────────────────────────────────────

def run(cfg: dict, use_diffusion: bool = True):
    """
    Full pipeline:
      fit → predict → evaluate → save

    Args:
        cfg           : config dict (loaded from default.yaml or overridden in notebook)
        use_diffusion : blend diffusion residual into score map (slower, more precise)

    Returns:
        results dict with AUROC / PRO scores
    """
    category    = cfg["data"]["category"]
    ckpt_dir    = Path(cfg["output"]["checkpoint_dir"]) / category
    results_dir = Path(cfg["output"]["results_dir"])    / category

    print(f"\n{'='*50}")
    print(f"  anomaly-vision  |  category: {category}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # ── Data ──
    train_loader, test_loader = get_dataloaders(cfg)
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Test  samples : {len(test_loader.dataset)}")

    # ── Model ──
    detector = AnomalyDetector(cfg)

    # ── Fit (or load cached bank) ──
    if (ckpt_dir / "memory_bank.pkl").exists():
        print("\n  Found existing checkpoint — skipping fit.")
        load_checkpoint(detector, ckpt_dir)
    else:
        detector.fit(train_loader)
        save_checkpoint(detector, cfg, ckpt_dir)

    # ── Predict ──
    print("\n  Running inference on test set...")
    image_scores, score_maps, labels = detector.predict(
        test_loader, use_diffusion=use_diffusion
    )

    # ── Collect GT masks + images for metrics + viz ──
    gt_masks = _collect_gt(test_loader)
    images   = _collect_images(test_loader)

    # ── Evaluate ──
    results = evaluate(
        image_scores=image_scores,
        score_maps=score_maps,
        gt_masks=gt_masks,
        labels=labels,
        category=category,
        output_dir=str(results_dir),
    )

    # ── Visualise ──
    save_anomaly_maps(
        images=images,
        score_maps=score_maps,
        gt_masks=gt_masks,
        labels=labels,
        output_dir=str(results_dir),
    )

    # ── Persist results ──
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved  → {results_path}")

    return results


# ─── 4. MULTI-CATEGORY SWEEP ─────────────────────────────────────────────────

def run_all_categories(cfg: dict, use_diffusion: bool = True):
    """
    Runs the full pipeline across all 15 MVTec categories.
    Aggregates results into a summary table.
    Useful for benchmarking — shows mean AUROC across the full dataset.
    """
    from model import AnomalyDetector   # re-import cleanly inside function
    all_results = []

    for category in [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]:
        cfg["data"]["category"] = category
        results = run(cfg, use_diffusion=use_diffusion)
        all_results.append(results)

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY — all categories")
    print(f"{'='*50}")
    print(f"  {'Category':<15} {'Img AUROC':>10} {'Pix AUROC':>10} {'PRO':>8}")
    print(f"  {'-'*45}")
    for r in all_results:
        print(f"  {r['category']:<15} {r['image_auroc']:>10.4f} "
              f"{r['pixel_auroc']:>10.4f} {r['pro_score']:>8.4f}")

    mean_img = sum(r["image_auroc"] for r in all_results) / len(all_results)
    mean_pix = sum(r["pixel_auroc"] for r in all_results) / len(all_results)
    mean_pro = sum(r["pro_score"]   for r in all_results) / len(all_results)
    print(f"  {'-'*45}")
    print(f"  {'MEAN':<15} {mean_img:>10.4f} {mean_pix:>10.4f} {mean_pro:>8.4f}")

    # Save summary
    summary_path = Path(cfg["output"]["results_dir"]) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")

    return all_results


# ─── 5. CLI ENTRY POINT ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="anomaly-vision training script")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--category",     default=None, help="Override config category")
    parser.add_argument("--all",          action="store_true", help="Run all 15 categories")
    parser.add_argument("--no-diffusion", action="store_true", help="Skip diffusion refinement")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.category:
        cfg["data"]["category"] = args.category

    use_diff = not args.no_diffusion

    if args.all:
        run_all_categories(cfg, use_diffusion=use_diff)
    else:
        run(cfg, use_diffusion=use_diff)