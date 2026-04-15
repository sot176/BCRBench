# Models Overview

## 🧠 Implemented Models

| Model | Venue | Input | Key Idea |
|------|------|------|----------|
| **LMV-Net** | MICCAI 2026 (submitted) | Multi-view, longitudinal | Dual-stream attention leveraging both views of each breast |
| **ImgFeatAlign** | MICCAI 2025 | Single-view, longitudinal | Feature-space deformation for temporal alignment |
| **VMRA-MaR** | MICCAI 2025 | Multi-view, longitudinal | Extends Mirai with spatial and longitudinal asymmetry modeling |
| **OA-BReaCR** | MICCAI 2024 | Single-view, longitudinal | Learns temporal changes via feature deformation fields |
| **Mirai** | Sci. Transl. Med. 2021 | Multi-view, single-timepoint | Baseline risk prediction from a single screening visit |


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