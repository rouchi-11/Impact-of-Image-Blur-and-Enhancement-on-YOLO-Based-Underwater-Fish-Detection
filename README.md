# Underwater Fish Detection — YOLO + Image Degradation Study

A three-phase study on how blur and image enhancement affect YOLO-based fish detection on the URPC2020 dataset, and how to build a model robust to it.

---

## What This Study Does

Real underwater cameras produce degraded images — blur, colour shift, low contrast. Models that perform well on clean benchmarks often fail in deployment. This project:
1. **Measures** exactly how much blur degrades detection
2. **Tests** whether CLAHE and Unsharp Masking can fix it 
3. **Solves** it by training YOLOv8m on a mixed dataset of clean + degraded + enhanced images

---

## Results Summary

| Strategy | Best mAP50 | Worst mAP50 | Variance |
|----------|-----------|------------|----------|
| Phase 1 — No enhancement | 0.7633 | 0.6726 | 11.9% |
| Phase 2 — CLAHE / Unsharp Masking | 0.7281 | 0.5181 | 19.4% |
| **Phase 3 — YOLOv8m Mixed Training** | **0.790** | **0.775** | **1.9%** |

Mixed training eliminates the performance gap with **zero inference overhead**.

---

## Repository Structure

```
📦 underwater-fish-detection-yolo
├── 📓 Code1_Phase1_Architecture_Blur_CLAHE.ipynb   ← Train YOLOv8n & YOLOv11n, test 6 blur conditions + CLAHE
├── 📓 Code2_Phase2_Enhancement_Tuning.ipynb        ← CLAHE clip tuning, Unsharp Masking, combined pipeline
├── 📓 Code3_Phase3_Mixed_Training_YOLOv8m.ipynb    ← Build mixed dataset, train YOLOv8m, evaluate + plots
├── 🐍 dataset_builder.py                           ← Creates original / noisy / enhanced image variants
└── 📄 README.md
```

---

## Phases at a Glance

**Phase 1** — Trained YOLOv8n and YOLOv11n on clean URPC2020. Evaluated on Gaussian blur (σ=1,3,5) and Motion blur (k=5,10,15). YOLOv11n was faster and more accurate. Severe blur caused up to −11.23% mAP50. CLAHE at clipLimit=3.0 made things worse.

**Phase 2** — No new training. Tested CLAHE at clipLimit=3.0, Unsharp Masking, and their combination. Finding: higher clip limit = more harm. No enhancement method achieved full recovery. Worst result: −31.62% mAP50.

**Phase 3** — Built a mixed dataset (original + synthetically degraded + CLAHE-enhanced at clip=2.0). Trained YOLOv8m (25.9M params) — chosen over nano models because larger capacity is needed to learn from three distinct image distributions simultaneously. Final mAP50: 0.775–0.790 across all conditions.

---

## Key Takeaway

> Enhancement algorithms designed for human visual perception degrade machine learning performance. The right fix is training diversity, not a preprocessing pipeline.

---

## Dataset

**URPC2020** — 4 marine species (holothurian, echinus, scallop, starfish), ~5,543 images, YOLO-format labels.  
🔗 https://www.kaggle.com/datasets/lywang777/urpc2020

---

## Setup

```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
```

Download URPC2020 from the link above, then run the notebooks in order (Code1 → Code2 → Code3).
