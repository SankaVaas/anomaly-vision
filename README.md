# anomaly-vision

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20T4-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD-blue?style=flat-square)

**Zero-shot surface defect detection for industrial quality inspection**  
*No defect labels. No retraining. Deploy on a new product line in minutes.*

[Notebook](notebooks/anomaly_detection.ipynb) · [Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) · [Results](#results)

</div>

---

## The Problem

Industrial quality inspection has a fundamental data problem:

> Defects are rare by design. You may have thousands of normal images and zero labeled defect examples.

Supervised models (YOLO, Mask R-CNN) require hundreds of labeled defect examples per class. In manufacturing that label set rarely exists — and when it does, it is immediately outdated by the next product variant.

**anomaly-vision** solves this by framing defect detection as anomaly detection — the model learns only what *normal* looks like, then flags anything that deviates from it. Zero defect labels required at any stage.

---

## System Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │              anomaly-vision pipeline                 │
                        └─────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────────┐
  │  INPUT   │    │ FEATURE          │    │ MULTI-SCALE    │    │   PATCH     │
  │  IMAGE   │───▶│ EXTRACTOR        │───▶│ FUSION         │───▶│  EMBEDDING  │
  │ 256×256  │    │ WideResNet50     │    │ layer2+layer3  │    │ avg_pool    │
  │   RGB    │    │ (frozen,         │    │ upsample+cat   │    │ 3×3 stride1 │
  └──────────┘    │  ImageNet)       │    │ [B,1536,28,28] │    │ [B, N, C]   │
                  └──────────────────┘    └────────────────┘    └──────┬──────┘
                                                                        │
                        ┌───────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────┐         FIT PHASE (normal images only)
  │      MEMORY BANK            │◀────────────────────────────────────────────
  │  Coreset subsampled         │         • One forward pass, no gradients
  │  vectors from normal imgs   │         • Greedy/random coreset keeps bank
  │  [M, 1536]                  │           tractable (~1000 vectors)
  │  saved: memory_bank.pkl     │         • Cached to disk after first fit
  └──────────────┬──────────────┘
                 │  kNN distance (k=9)
                 ▼
  ┌──────────────────────────┐    ┌──────────────────────────────────────────┐
  │   PATCHCORE SCORE MAP    │    │         DIFFUSION REFINEMENT             │
  │   [B, 1, H, W]           │    │  partial noise at t* → reconstruct       │
  │   patch-level kNN dist   │    │  → residual map [B, 1, H, W]            │
  └──────────────┬───────────┘    │  defects "heal" → pixel-precise signal  │
                 │   0.7 ×        └────────────────────────┬─────────────────┘
                 │                                         │ 0.3 ×
                 └──────────────┬──────────────────────────┘
                                ▼
                  ┌─────────────────────────┐
                  │   FUSED ANOMALY MAP     │
                  │   [B, 1, 256, 256]      │
                  │   upsampled to input    │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │   THRESHOLD             │
                  │   max(μ+3σ, p95)        │
                  │   of normal train scores│
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │   VERDICT               │
                  │   NORMAL ✓              │
                  │   DEFECTIVE ⚠           │
                  └─────────────────────────┘
```

---

## Why This Architecture

| Design Decision                    | Why                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Frozen pretrained backbone         | ImageNet features generalise to industrial textures without any domain training                                 |
| Multi-scale fusion (layer2+layer3) | layer2 captures fine texture, layer3 captures semantic structure — neither alone is sufficient                  |
| Coreset subsampling                | Reduces 5M+ patch vectors to ~1000 without losing coverage — makes kNN tractable on CPU/T4                      |
| Diffusion residual blend           | Adds pixel-precise localisation — PatchCore scores at patch granularity (28×28), diffusion pushes it to 256×256 |
| Per-category threshold (μ+3σ, p95) | PatchCore scores are right-skewed — fixed threshold fails across categories, statistical threshold adapts       |
| Zero trainable parameters          | No overfitting possible, no retraining needed for new categories, fully explainable decision boundary           |

---

## Evaluation Metrics

Three metrics — each measuring something the others miss:

**Image AUROC** — separates normal vs anomalous images globally. Easy to game with large defects.

**Pixel AUROC** — measures score map alignment with ground-truth masks at pixel level. Dominated by large defects.

**PRO Score (Per-Region Overlap)** — iterates over every connected defect region independently. A 10-pixel scratch counts equally to a 500-pixel gouge. The honest metric.

---

## Results — MVTec AD

 | Category   | Img AUROC | Pix AUROC | PRO score |
 | ---------- | --------- | --------- | --------- |
 | bottle     | 0.9937    | 0.9813    | 0.8879    |
 | cable      | 0.9303    | 0.9715    | 0.7494    |
 | capsule    | 0.8412    | 0.9852    | 0.8474    |
 | carpet     | 0.9839    | 0.9910    | 0.8749    |
 | grid       | 0.8580    | 0.9634    | 0.8536    |
 | hazelnut   | 1.0000    | 0.9864    | 0.9348    |
 | leather    | 1.0000    | 0.9926    | 0.9130    |
 | metal_nut  | 0.9868    | 0.9719    | 0.8671    |
 | pill       | 0.9193    | 0.9660    | 0.9103    |
 | screw      | 0.8045    | 0.9760    | 0.8585    |
 | tile       | 0.9982    | 0.9504    | 0.8465    |
 | toothbrush | 0.8722    | 0.9883    | 0.8210    |
 | transistor | 0.9554    | 0.9197    | 0.8105    |
 | wood       | 0.9789    | 0.9268    | 0.8046    |
 | zipper     | 0.9438    | 0.9750    | 0.8807    |
 | -----------|-----------|-----------|-----------|
 | **MEAN**   |**0.9377** |**0.9697** |**0.8573** |
 | -----------|-----------|-----------|-----------|

> Run `train.py --all` to populate the full table across all 15 categories.

**Reference — published PatchCore (WR50):** Image AUROC 0.999 · Pixel AUROC 0.985 · PRO ~0.89  
Our results sit between PaDiM and PatchCore paper — expected given random coreset and sparse bank tradeoffs made for Colab T4 constraints.

---

## Project Structure

```
anomaly-vision/
├── data/
│   ├── raw/                     # downloaded MVTec zip
│   └── processed/               # symlink to extracted dataset
├── src/
│   ├── dataset.py               # MVTec loader, train/test splits, mask loading
│   ├── model.py                 # FeatureExtractor, MemoryBank, DiffusionRefinement, AnomalyDetector
│   ├── train.py                 # fit → predict → evaluate → save, CLI + importable
│   ├── evaluate.py              # Image AUROC, Pixel AUROC, PRO score, visualisation
│   └── utils.py                 # seed, config, device, logging, normalisation
├── configs/
│   └── default.yaml             # all hyperparameters — nothing hardcoded in src/
├── notebooks/
│   └── anomaly_detection.ipynb  # full Colab pipeline
├── outputs/
│   ├── checkpoints/             # memory_bank.pkl per category
│   └── results/                 # AUROC/PRO scores, anomaly maps
├── requirements.txt
└── README.md
```

---

## Quickstart

**Run in Colab (recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SankaVaas/anomaly-vision/blob/main/notebooks/anomaly_detection.ipynb)

1. Open the notebook, select **T4 GPU** runtime
2. Upload your `kaggle.json`
3. Run all cells top to bottom

**Run locally**

```bash
git clone https://github.com/SankaVaas/anomaly-vision.git
cd anomaly-vision
pip install -r requirements.txt

# Fit memory bank and evaluate on bottle category
python src/train.py --category bottle

# Run all 15 categories
python src/train.py --all

# Skip diffusion (faster)
python src/train.py --category bottle --no-diffusion
```

---

## Key Dependencies

| Package                 | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `timm`                  | WideResNet50 pretrained backbone           |
| `diffusers`             | DDPM diffusion prior for reconstruction    |
| `scikit-image`          | Connected component analysis for PRO score |
| `scikit-learn`          | AUROC computation                          |
| `torch` + `torchvision` | Feature extraction, tensor ops             |

---

## What Makes This Different

Most anomaly detection repos train a model and report AUROC. This project:

- **Combines PatchCore + diffusion residual** — two complementary signals fused at inference
- **Per-category adaptive threshold** — statistically grounded, not a hardcoded magic number
- **Multi-category auto-detection** — feed any image, the system identifies the product category and scores it
- **Production-aware design** — zero retraining for new SKUs, memory bank fits in 50MB, runs on edge hardware

---

## Dataset

[MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 industrial categories, 5354 images, pixel-precise ground truth masks.

```
@inproceedings{bergmann2019mvtec,
  title={MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={CVPR},
  year={2019}
}
```

---

## Roadmap

- [ ] Uncertainty maps alongside anomaly scores (epistemic confidence per patch)
- [ ] Adaptive noise scheduling — per-image `t*` based on texture complexity
- [ ] Few-shot category onboarding — meta-memory bank from 10 normal images
- [ ] ONNX export for edge deployment
- [ ] FastAPI inference server

---

<div align="center">
Built for the <strong>AI Tech Lead / Architect</strong> portfolio  
Domain: Industrial Computer Vision · Zero-Shot Anomaly Detection
</div>