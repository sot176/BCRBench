# Overview

This page provides a technical overview of BCRBench: A breast cancer risk benchmark, including its scope, objectives, and supported modeling paradigms.

---

## 🔬 Scope

The framework focuses on image-based breast cancer risk prediction from mammography and supports:

- Single-view and multi-view inputs  
- Single-timepoint and longitudinal data  
- Breast-wise and patient-wise modeling  

---

## 🎯 Objectives

The main goals of this benchmark are:

- Standardizing model implementations across methods  
- Enabling fair and reproducible comparisons  
- Providing a unified research framework  
- Accelerating progress in breast cancer risk prediction  

---

## 📊 Supported Tasks

The framework supports:

- End-to-end model training  
- Risk prediction evaluation  
- Consistent benchmarking across architectures  

---

## 🧠 Model Overview

The benchmark includes the following models:

| Model | Venue | Paper | Input | Key Idea |
|------|------|------|-----|----------|
| [LMV-Net](./models/lmvnet.md) | MICCAI 2026 | - | Multi-view, longitudinal | Multi-view longitudinal model with dual-stream attention leveraging both views of one breast. |
| [ImgFeatAlign](./models/imgfeatalign.md) | MICCAI 2025 | [Paper](https://doi.org/10.1007/978-3-032-04937-7_47) | Single-view, longitudinal | Uses image-based deformation (MammoRegNet) applied in feature space for improved longitudinal comparison. |
| [VMRA-MaR](./models/vmramar.md) | MICCAI 2025 | [Paper](https://doi.org/10.1007/978-3-032-05182-0_64) | Multi-view, longitudinal | Extends Mirai to longitudinal mammograms using Spatial Asymmetry Detector and Longitudinal Asymmetry Tracker. |
| [OA-BReaCR](./models/oabreacr.md) | MICCAI 2024 | [Paper](https://doi.org/10.1007/978-3-031-72378-0_15) | Single-view, longitudinal | Learns longitudinal changes using feature-based deformation fields for better temporal alignment. |
| [Mirai](./models/mirai.md) | Sci. Transl. Med. 2021 | [Paper](https://doi.org/10.1126/scitranslmed.aba4373) | Multi-view, single-timepoint | Learns breast cancer risk from all four views in a single visit |

---

## 🏗️ Framework Design

All models in this benchmark share:

- A unified data interface  
- A standardized training pipeline  
- Consistent evaluation metrics  

This ensures fair and reproducible comparison across methods.

---

## 🚀 Extensibility

The framework is designed to be easily extended:

- Add new models with minimal changes  
- Integrate new datasets  
- Extend training and evaluation pipelines  

---

## 📚 Next Steps

- See **Datasets** for data preparation details  
- See **Models** for architecture-level documentation  
- See **Getting Started** to run experiments