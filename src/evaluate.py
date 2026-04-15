"""
Evaluation metrics for anomaly detection.

Metrics:
  - Image-level AUROC  : how well we separate normal vs anomalous images
  - Pixel-level AUROC  : how well the score map aligns with ground-truth masks
  - PRO (Per-Region Overlap) : measures localisation quality across defect sizes
                               (small defects are weighted equally to large ones —
                                a metric AUROC alone misses entirely)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.measure import label as connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ─── 1. IMAGE-LEVEL AUROC ────────────────────────────────────────────────────

def image_auroc(scores: list, labels: list) -> float:
    """
    scores : list[float]  — one anomaly score per image
    labels : list[int]    — 0 = normal, 1 = anomaly
    """
    scores = np.array(scores)
    labels = np.array(labels)
    # Guard: need both classes present
    if len(np.unique(labels)) < 2:
        print("  Warning: only one class in labels, AUROC undefined.")
        return float("nan")
    return roc_auc_score(labels, scores)


# ─── 2. PIXEL-LEVEL AUROC ────────────────────────────────────────────────────

def pixel_auroc(score_maps: list, gt_masks: list) -> float:
    """
    score_maps : list[Tensor [1,H,W]]  — anomaly heatmap per image
    gt_masks   : list[Tensor [1,H,W]]  — binary ground-truth mask per image
    """
    all_scores, all_labels = [], []
    for smap, mask in zip(score_maps, gt_masks):
        all_scores.append(smap.squeeze().numpy().ravel())
        all_labels.append(mask.squeeze().numpy().ravel().astype(int))

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    if len(np.unique(all_labels)) < 2:
        print("  Warning: no anomalous pixels in test set.")
        return float("nan")
    return roc_auc_score(all_labels, all_scores)


# ─── 3. PRO SCORE ─────────────────────────────────────────────────────────────

def pro_score(score_maps: list, gt_masks: list, num_thresholds: int = 100) -> float:
    """
    Per-Region Overlap (PRO) — MVTec paper's primary localisation metric.

    For each threshold t:
      - Binarise score map at t
      - For every connected defect region in the GT mask,
        compute overlap = (predicted ∩ region) / |region|
      - Average overlap across all regions  →  one PRO value at t
    Integrate PRO vs FPR curve up to FPR=0.3 (normalised).

    Why this matters: a large defect covering 80% of the image is
    treated equally to a tiny scratch — standard pixel AUROC would
    be dominated by the large defect.
    """
    # Collect all scores + masks as flat arrays (for threshold sweep)
    all_fprs, all_pros = [], []

    thresholds = np.linspace(
        min(s.min().item() for s in score_maps),
        max(s.max().item() for s in score_maps),
        num_thresholds
    )

    for thresh in thresholds:
        fpr_list, pro_list = [], []

        for smap, mask in zip(score_maps, gt_masks):
            pred_bin = (smap.squeeze().numpy() >= thresh).astype(np.uint8)
            gt_bin   = mask.squeeze().numpy().astype(np.uint8)

            # FPR at this threshold for this image
            normal_pixels = (gt_bin == 0).sum()
            fp = ((pred_bin == 1) & (gt_bin == 0)).sum()
            fpr_list.append(fp / (normal_pixels + 1e-8))

            # PRO: iterate over connected defect regions
            regions = connected_components(gt_bin, connectivity=2)
            for region_id in range(1, regions.max() + 1):
                region_mask = (regions == region_id)
                overlap = (pred_bin[region_mask]).sum() / (region_mask.sum() + 1e-8)
                pro_list.append(overlap)

        all_fprs.append(np.mean(fpr_list))
        all_pros.append(np.mean(pro_list) if pro_list else 0.0)

    all_fprs = np.array(all_fprs)
    all_pros = np.array(all_pros)

    # Normalised area under PRO curve up to FPR = 0.3
    mask_fpr = all_fprs <= 0.3
    if mask_fpr.sum() < 2:
        return float("nan")

    sorted_idx = np.argsort(all_fprs[mask_fpr])
    fprs_trim  = all_fprs[mask_fpr][sorted_idx]
    pros_trim  = all_pros[mask_fpr][sorted_idx]
    aupro = np.trapz(pros_trim, fprs_trim) / 0.3   # normalise to [0, 1]
    return float(aupro)


# ─── 4. FULL EVALUATION REPORT ───────────────────────────────────────────────

def evaluate(
    image_scores: list,
    score_maps: list,
    gt_masks: list,
    labels: list,
    category: str,
    output_dir: str = "outputs/results",
) -> dict:
    """
    Runs all three metrics and prints a clean report.
    Returns dict of results (easy to log to W&B / CSV).
    """
    print(f"\n{'─'*45}")
    print(f"  Evaluation — category: {category}")
    print(f"{'─'*45}")

    img_auc  = image_auroc(image_scores, labels)
    pix_auc  = pixel_auroc(score_maps, gt_masks)
    pro      = pro_score(score_maps, gt_masks)

    print(f"  Image AUROC : {img_auc:.4f}")
    print(f"  Pixel AUROC : {pix_auc:.4f}")
    print(f"  PRO Score   : {pro:.4f}")
    print(f"{'─'*45}\n")

    results = {"category": category, "image_auroc": img_auc,
               "pixel_auroc": pix_auc, "pro_score": pro}
    return results


# ─── 5. VISUALISATION ────────────────────────────────────────────────────────

def save_anomaly_maps(
    images: list,
    score_maps: list,
    gt_masks: list,
    labels: list,
    output_dir: str,
    n_samples: int = 8,
    denorm_mean=(0.485, 0.456, 0.406),
    denorm_std=(0.229, 0.224, 0.225),
):
    """
    Saves a grid: original | GT mask | anomaly heatmap
    for the first n_samples anomalous images.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    anomaly_indices = [i for i, l in enumerate(labels) if l == 1][:n_samples]
    if not anomaly_indices:
        print("No anomalous samples to visualise.")
        return

    fig, axes = plt.subplots(len(anomaly_indices), 3,
                             figsize=(9, 3 * len(anomaly_indices)))
    if len(anomaly_indices) == 1:
        axes = [axes]

    mean = np.array(denorm_mean).reshape(3, 1, 1)
    std  = np.array(denorm_std).reshape(3, 1, 1)

    for row, idx in enumerate(anomaly_indices):
        # De-normalise image
        img = images[idx].numpy() * std + mean
        img = np.clip(img.transpose(1, 2, 0), 0, 1)

        smap = score_maps[idx].squeeze().numpy()
        mask = gt_masks[idx].squeeze().numpy()

        # Normalise score map to [0,1] for display
        smap_norm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)

        axes[row][0].imshow(img);           axes[row][0].set_title("Input")
        axes[row][1].imshow(mask, cmap="gray"); axes[row][1].set_title("GT Mask")
        axes[row][2].imshow(img)
        axes[row][2].imshow(smap_norm, cmap="jet", alpha=0.5)
        axes[row][2].set_title("Anomaly Map")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(output_dir) / "anomaly_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved anomaly maps → {out_path}")