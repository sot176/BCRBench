# Models Overview

## 🧠 Implemented Models

| Model | Venue | Input | Key Idea |
|------|------|------|----------|
| **LMV-Net** | MICCAI 2026 (submitted) | Multi-view, longitudinal |Multi-view longitudinal model with dual-stream attention leveraging both views of one breast. |
| **ImgFeatAlign** | MICCAI 2025 | Single-view, longitudinal | Uses image-based deformation (MammoRegNet) applied in feature space for improved longitudinal comparison. |
| **VMRA-MaR** | MICCAI 2025 | Multi-view, longitudinal | Extends Mirai to longitudinal mammograms using Spatial Asymmetry Detector and Longitudinal Asymmetry Tracker. |
| **OA-BReaCR** | MICCAI 2024 | Single-view, longitudinal | Learns longitudinal changes using feature-based deformation fields for better temporal alignment. |
| **Mirai** | Sci. Transl. Med. 2021 | Multi-view, single-timepoint | Learns breast cancer risk from all four views in a single visit  |


---

## 🔍 Model Comparison  

### Input Representation

- **Single-view models:**
    - OA-BReaCR
    - ImgFeatAlign


- **Multi-view models:**
    - Mirai
    - VMRA-MaR
    - LMV-Net

---

### Temporal Modeling

- **Single-timepoint:**
    - Mirai


- **Longitudinal (multi-timepoint):**
    - VMRA-MaR
    - OA-BReaCR
    - ImgFeatAlign
    - LMV-Net

---

### Aggregation Strategy

- **Patient-wise (all views jointly):**
    - Mirai
    - VMRA-MaR


- **Breast-wise modeling:**
    - LMV-Net