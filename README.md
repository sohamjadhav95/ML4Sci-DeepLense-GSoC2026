# ML4Sci GSoC 2026: Soham Jadhav

Evaluation test submissions for **Google Summer of Code 2026** with [ML4Sci](https://ml4sci.org/).

---

## Submissions Overview

| Project | Task | Model | Key Result |
|---|---|---|---|
| [DeepLense — Lens Finding (DEEPLENSE)](DeepLense/README.md) | Test I: Multi-Class Classification | ResNet18 | Macro AUC 0.9851 · Val Acc 92.56% |
| [DeepLense — Lens Finding (DEEPLENSE)](DeepLense/README.md) | Test V: Lens Finding (Binary) | ResNet18 | AUC 0.9887 · Test Acc 97.01% |
| [PrediCT — CAC Segmentation (PREDICT1)](Predict1/README.md) | Common Task: Preprocessing Pipeline | — | Stratified pipeline, 1,722 train slices |
| [PrediCT — CAC Segmentation (PREDICT1)](Predict1/README.md) | Specific Task: Heart Segmentation | 2D U-Net (7.8M params) | Median Dice **0.9416** · ~63× faster than TotalSegmentator |

---

## DeepLense

**Transfer Learning for Gravitational Lens Classification and Detection** — fine-tuned ResNet18 on simulated and real lensing datasets.

### Test I: Multi-Class Gravitational Lens Classification

**Task:** Classify strong lensing images into 3 dark matter substructure categories: no substructure (`no`), vortex (`vort`), spherical (`sphere`).

**Dataset:** `.npy` files, shape `(1, 150, 150)` — 30,000 train / 7,500 val samples

**Model:** ResNet18 pretrained on ImageNet, FC head replaced → `Linear(512, 3)` | Loss: CrossEntropyLoss | Optimizer: Adam (lr=0.001) | Epochs: 10

**Training Curves:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1  | 1.0260 | 0.4434 | 1.1904 | 0.5103 |
| 4  | 0.3540 | 0.8667 | 0.2869 | 0.8983 |
| 7  | 0.2692 | 0.9039 | 0.2319 | 0.9187 |
| 10 | 0.2277 | 0.9177 | 0.2089 | **0.9256** |

**Results:**
- Val Accuracy: **0.9256** (epoch 10)
- Macro Average AUC: **0.9851**

📓 [Notebook](DeepLense/Test_I_MultiClass_Classification/Deeplens_test_01.ipynb)

---

### Test V: Gravitational Lens Finding (Binary Classification)

**Task:** Binary classification — lensed vs non-lensed galaxies. Severe class imbalance (~5.7% lenses) handled via `WeightedRandomSampler`.

**Dataset:** `.npy` files, shape `(3, 64, 64)` — 30,405 train (1,730 lenses / 28,675 non-lenses) / 19,650 test samples

**Model:** ResNet18 pretrained on ImageNet, FC head replaced → `Linear(512, 1)` | Loss: BCEWithLogitsLoss | Optimizer: Adam (lr=0.001) | Epochs: 10

**Training Curves:**

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1  | 0.2172 | 0.9165 | 0.1158 | 0.9527 |
| 4  | 0.1215 | 0.9562 | 0.0842 | 0.9666 |
| 7  | 0.0911 | 0.9679 | 0.1295 | 0.9500 |
| 10 | 0.0748 | 0.9727 | 0.0848 | **0.9701** |

**Results:**
- Test Accuracy: **0.9701** (epoch 10)
- AUC Score: **0.9887**

📓 [Notebook](DeepLense/Test_V_Lens_Finding/Test_V_Lens_Finding.ipynb)

---

## PREDICT1: CAC Heart Segmentation

**Building and Comparing Segmentation Strategies for Coronary Artery Calcium (CAC)**

**Dataset:** Stanford COCA — 50 gated coronary CT scans, resampled to 0.7×0.7×3.0 mm

### Common Task: Preprocessing Pipeline

| Step | Detail |
|---|---|
| HU Windowing | [−200, 600] HU → normalized [0, 1] |
| Stratified Split | 70/15/15 by calcium burden (Mild/Moderate/Severe) |
| Class Imbalance | `WeightedRandomSampler` — 2× weight for heart-containing slices |
| Slice Counts | 1,722 train / 378 val / 389 test slices |
| Augmentation | Random flip, 90° rotation, brightness jitter ±0.05 (train only) |
| DataLoader | RAM-preloaded, `batch_size=8` |

📓 [Common Task Notebook](Predict1/Common_Task_Preprocessing.ipynb)

### Specific Task: Heart Segmentation Model

**Ground truth:** TotalSegmentator (`--fast`) on all 50 scans — 6 heart structures merged into a single binary mask

**Model:** Lightweight 2D U-Net

```
Input (1×256×256)
  → Encoder: 4× [Conv → BN → ReLU → Conv → BN → ReLU → MaxPool]
  → Bottleneck: DoubleConv (256→512 channels)
  → Decoder: 4× [Upsample → Skip concat → DoubleConv]
  → Output Conv 1×1
Output (1×256×256) — binary heart mask
```

Channel progression: 1 → 32 → 64 → 128 → 256 → 512 | **7,849,025 parameters**

**Training Configuration:**

| Hyperparameter | Value |
|---|---|
| Loss | DiceBCE (α=0.5) |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| LR Scheduler | ReduceLROnPlateau (×0.5, patience=5) |
| Mixed Precision | AMP (autocast) |
| Epochs | 50 (early stopping patience=10, not triggered) |

**Results:**

| Metric | Value |
|---|---|
| **Median Dice (test)** | **0.9416** |
| Mean Dice (test) | 0.8354 ± 0.2797 |
| Mean IoU | 0.7838 |
| % slices ≥ 0.85 Dice | 77.6% |
| Best Val Dice | 0.9395 (epoch 44) |

> **Note on mean vs median:** The high standard deviation reflects a small number of near-empty background slices (Dice → 0) pulling the mean down. The median Dice of 0.9416 substantially exceeds the 0.85 target.

**Inference Time Comparison:**

| Model | Time |
|---|---|
| **U-Net (ours)** | **~600 ms/scan** (11.97 ms/slice × ~50 slices) |
| TotalSegmentator (`--fast`) | 37,800 ms/scan (37.8s avg across 50 scans) |
| **Speedup** | **~63× faster per scan** |

📓 [Specific Task Notebook](Predict1/Specific_Task_Segmentation.ipynb) | 📦 [Model Weights (Google Drive)](https://drive.google.com/your-link-here)

---

## Repository Structure

```
ML4Sci-GSoC-2026/
├── DeepLense/
│   ├── Test_I_MultiClass_Classification/
│   │   ├── Deeplens_test_01.ipynb       # Multi-class classification
│   │   ├── resnet18_test1.pth           # Saved model weights
│   │   └── roc_curve_test1.png          # Per-class ROC curves
│   ├── Test_V_Lens_Finding/
│   │   ├── Test_V_Lens_Finding.ipynb    # Binary lens finding
│   │   └── resnet18_test1.pth           # Saved model weights
│   └── README.md
└── Predict1/
    ├── Common_Task_Preprocessing.ipynb
    ├── Specific_Task_Segmentation.ipynb
    ├── requirements.txt
    └── README.md
```

---

## Environment

| Component | DeepLense | PREDICT1 |
|---|---|---|
| Python | 3.12 | 3.11.9 |
| PyTorch | 2.x (CUDA 12.x) | 2.x (CUDA 11.8) |
| GPU | T4 (Google Colab) | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| OS | Google Colab | Windows 11 |

---

## Contact

**Soham Jadhav** | GSoC 2026 Applicant  
GitHub: [@sohamjadhav95](https://github.com/sohamjadhav95)