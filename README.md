# ML4Sci GSoC 2026: Soham Jadhav

Evaluation test submissions for **Google Summer of Code 2026** with [ML4Sci](https://ml4sci.org/).

---

## Submissions Overview

| Project | Test | Model | Key Result |
|---|---|---|---|
| DeepLense — Lens Finding (DEEPLENSE6) | Test I + Test V | ResNet18 | AUC 0.9887 |
| DeepLense — Data Pipeline LSST (DEEPLENSE7) | Test I + Test V | ResNet18 | AUC 0.9887 |
| PrediCT — CAC Segmentation (PREDICT1) | Common + Specific | 2D U-Net | Median Dice 0.9416 |

---

## DeepLense

### Test I: Multi-Class Gravitational Lens Classification

**Task:** Classify strong lensing images into 3 classes: no substructure, subhalo substructure, vortex substructure.

**Model:** ResNet18 (pretrained ImageNet, fine-tuned) | PyTorch

**Results:**
- Macro AUC: **0.9851**
- Validation Accuracy: ~92.5%

📓 [Notebook](DeepLense/Test_I_MultiClass_Classification/Deeplens_test_01.ipynb)

---

### Test V: Gravitational Lens Finding (Binary Classification)

**Task:** Binary classification: lensed vs non-lensed galaxies. Severe class imbalance handled via WeightedRandomSampler.

**Model:** ResNet18 (pretrained ImageNet, fine-tuned) | PyTorch

**Results:**
- AUC: **0.9887**
- Test Accuracy: ~97%

📓 [Notebook](DeepLense/Test_V_Lens_Finding/Test_V_Lens_Finding.ipynb)

---

## PREDICT1: CAC Heart Segmentation

**Task:** Build a preprocessing pipeline for the Stanford COCA dataset and train a lightweight segmentation model to predict whole-heart masks, benchmarked against TotalSegmentator as ground truth.

**Dataset:** Stanford COCA: 50 gated coronary CT scans, resampled to 0.7×0.7×3.0mm

### Common Task: Preprocessing Pipeline
- HU windowing: [-200, 600] → normalized [0,1]
- Stratified 70/15/15 split by calcium burden category
- WeightedRandomSampler (2× heart slices) for class imbalance
- RAM-preloaded 2D slice dataset: 1,722 train / 378 val / 389 test slices
- Augmentation: random flip, 90° rotation, brightness jitter

📓 [Common Task Notebook](Predict1/Common_Task_Preprocessing.ipynb)

### Specific Task: Heart Segmentation Model
- **Ground truth:** TotalSegmentator (`--fast`) on 50 scans, 6 heart structures merged
- **Model:** Lightweight 2D U-Net, 7.8M parameters, base_features=32
- **Loss:** Combined Dice + BCE (α=0.5)
- **Training:** AdamW + ReduceLROnPlateau + AMP mixed precision, 50 epochs

**Results:**

| Metric | Value |
|---|---|
| Median Dice (test) | **0.9416** |
| Mean Dice (test) | 0.8354 ± 0.2797 |
| Mean IoU | 0.7838 |
| Best Val Dice | 0.9395 (epoch 44) |
| U-Net inference | **11.97 ms/slice** |
| TotalSegmentator | 37,800 ms/scan |
| Speedup | **~3,157×** faster |

📓 [Specific Task Notebook](Predict1/Specific_Task_Segmentation.ipynb) | 📦 [Model Weights (Google Drive)](https://drive.google.com/your-link-here)

---

## Repository Structure

```
ML4Sci-GSoC-2026/
├── DeepLense/
│   ├── Test_I_MultiClass_Classification/
│   │   └── Deeplens_test_01.ipynb
│   └── Test_V_Lens_Finding/
│       └── Test_V_Lens_Finding.ipynb
└── Predict1/
    ├── Common_Task_Preprocessing.ipynb
    ├── Specific_Task_Segmentation.ipynb
    ├── requirements.txt
    └── README.md
```

---

## Environment

```
Python    3.11.9
PyTorch   2.x (CUDA 11.8)
GPU       NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
OS        Windows 11
```

---

## Contact

**Soham Jadhav** | GSoC 2026 Applicant
GitHub: [@sohamjadhav95](https://github.com/sohamjadhav95)