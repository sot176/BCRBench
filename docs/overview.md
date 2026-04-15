# Overview

This page provides a technical overview of the Breast Cancer Risk Benchmark, including its scope, objectives, and supported modeling paradigms.

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

| Model | Venue | Input | Key Idea |
|------|------|------|----------|
| **LMV-Net** | MICCAI 2026 (submitted) | Multi-view, longitudinal | Dual-stream attention leveraging both views of each breast |
| **ImgFeatAlign** | MICCAI 2025 | Single-view, longitudinal | Feature-space deformation for temporal alignment |
| **VMRA-MaR** | MICCAI 2025 | Multi-view, longitudinal | Extends Mirai with spatial and longitudinal asymmetry modeling |
| **OA-BReaCR** | MICCAI 2024 | Single-view, longitudinal | Learns temporal changes via feature deformation fields |
| **Mirai** | Sci. Transl. Med. 2021 | Multi-view, single-timepoint | Baseline risk prediction from a single screening visit |

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