# DEEPLENSE: Gravitational Lens Analysis
### GSoC 2026 | ML4Sci | DeepLense Project

**Transfer Learning for Gravitational Lens Classification and Detection**

---

## Overview

This submission implements two evaluation tasks for the DeepLense project:

- **Test I (Common Task):** Multi-class classification of gravitational lensing simulations into three substructure categories using a fine-tuned ResNet18
- **Test V (Specific Task):** Binary lens-finding classifier to distinguish real gravitational lenses from non-lenses using a fine-tuned ResNet18 with class-imbalance handling

---

## Repository Structure

```
DeepLense/
├── Test_I_MultiClass_Classification/
│   ├── Deeplens_test_01.ipynb      # Multi-class classification: no lensing / vort / sphere
│   ├── resnet18_test1.pth          # Saved model weights
│   └── roc_curve_test1.png         # Per-class ROC curves
├── Test_V_Lens_Finding/
│   ├── Test_V_Lens_Finding.ipynb   # Binary lens vs. non-lens classification
│   ├── resnet18_test1.pth          # Saved model weights
│   └── Screenshot 2026-03-11 214313.png
└── README.md
```

---

## Test I: Multi-Class Classification

### Task Description

Classify gravitational lensing simulations into one of three dark matter substructure categories:

| Class | Description |
|---|---|
| `no` | No substructure |
| `vort` | Vortex substructure |
| `sphere` | Spherical substructure |

### Dataset

- **Format:** `.npy` files, shape `(1, 150, 150)`   single-channel simulation images
- **Train samples:** 30,000 | **Val samples:** 7,500
- Loaded via custom `LensDataset` (converts grayscale → RGB for ResNet18 compatibility)

### Data Augmentation (Train Only)

- Random horizontal and vertical flip (p=0.5)
- Random rotation ±30°
- Resize to 224×224
- ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

### Model: ResNet18 (Fine-Tuned)

- **Backbone:** ResNet18 pretrained on ImageNet
- **Head:** Replaced final FC layer → `Linear(512, 3)` for 3-class output
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 10

### Training Curves

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1  | 1.0260 | 0.4434 | 1.1904 | 0.5103 |
| 4  | 0.3540 | 0.8667 | 0.2869 | 0.8983 |
| 7  | 0.2692 | 0.9039 | 0.2319 | 0.9187 |
| 10 | 0.2277 | 0.9177 | 0.2089 | **0.9256** |

### Results

| Metric | Value |
|---|---|
| **Val Accuracy (epoch 10)** | **0.9256** |
| **Macro Average AUC** | **0.9851** |

ROC curves computed per-class (one-vs-rest) on the validation set.

---

## Test V: Gravitational Lens Finding

### Task Description

Binary classification: distinguish real gravitational lenses from non-lenses.

| Class | Label |
|---|---|
| Lens | 1 |
| Non-lens | 0 |

### Dataset

- **Format:** `.npy` files, shape `(3, 64, 64)`   3-channel images
- **Train samples:** 30,405 (lenses: 1,730 | non-lenses: 28,675)
- **Test samples:** 19,650

> **Class Imbalance:** Lenses are heavily underrepresented (~5.7% of training data). Addressed via `WeightedRandomSampler`   each class sampled proportionally to its inverse frequency.

### Data Augmentation (Train Only)

- Random horizontal and vertical flip (p=0.5)
- Random rotation ±30°
- Resize to 224×224
- ImageNet normalization

### Model: ResNet18 (Fine-Tuned)

- **Backbone:** ResNet18 pretrained on ImageNet
- **Head:** Replaced final FC layer → `Linear(512, 1)` for binary output
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 10

### Training Curves

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1  | 0.2172 | 0.9165 | 0.1158 | 0.9527 |
| 4  | 0.1215 | 0.9562 | 0.0842 | 0.9666 |
| 7  | 0.0911 | 0.9679 | 0.1295 | 0.9500 |
| 10 | 0.0748 | 0.9727 | 0.0848 | **0.9701** |

### Results

| Metric | Value |
|---|---|
| **Test Accuracy (epoch 10)** | **0.9701** |
| **AUC Score** | **0.9887** |

---

## Environment

```
Python      3.12
PyTorch     2.x (CUDA 12.x / T4 GPU)
Platform    Google Colab
```

---

## Dependencies

```
torch
torchvision
numpy
matplotlib
Pillow
scikit-learn
```

---

## Reproducibility

All experiments run on Google Colab with a T4 GPU. To reproduce:

```bash
# 1. Download dataset from ML4Sci (Stanford AIMI / shared drive)
# 2. Run Test_I_MultiClass_Classification/Deeplens_test_01.ipynb
# 3. Run Test_V_Lens_Finding/Test_V_Lens_Finding.ipynb
```

---

## References

- [DeepLense Project ML4Sci GSoC 2026](https://ml4sci.org/gsoc/projects/2026/project_DEEPLENSE.html)
- [ResNet: He et al., 2015](https://arxiv.org/abs/1512.03385)
- [Gravitational Lensing Simulations Dataset ML4Sci](https://github.com/ML4SCI/DeepLense)
