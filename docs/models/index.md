# Models Overview

## 🧠 Implemented Models

| Model | Venue | Paper | Input | Key Idea |
|------|------|------|-----| ----------|
| [LMV-Net](./lmvnet.md) | MICCAI 2026 (submitted) |  - | Multi-view, longitudinal | Multi-view longitudinal model with dual-stream attention leveraging both views of one breast. |
| [ImgFeatAlign](./imgfeatalign.md) | MICCAI 2025 |[Paper](https://doi.org/10.1007/978-3-032-04937-7_47) | Single-view, longitudinal | Uses image-based deformation (MammoRegNet) applied in feature space for improved longitudinal comparison. |
| [VMRA-MaR](./vmramar.md) | MICCAI 2025 | [Paper](https://doi.org/10.1007/978-3-032-05182-0_64) |Multi-view, longitudinal | Extends Mirai to longitudinal mammograms using Spatial Asymmetry Detector and Longitudinal Asymmetry Tracker. |
| [OA-BReaCR](./oabreacr.md) | MICCAI 2024 | [Paper](https://doi.org/10.1007/978-3-031-72378-0_15) |Single-view, longitudinal | Learns longitudinal changes using feature-based deformation fields for better temporal alignment. |
| [Mirai](./mirai.md) | Sci. Transl. Med. 2021 |[Paper](https://doi.org/10.1126/scitranslmed.aba4373) | Multi-view, single-timepoint | Learns breast cancer risk from all four views in a single visit |

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