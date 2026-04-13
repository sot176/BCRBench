# Overview

This repository provides a unified framework for implementing and benchmarking state-of-the-art breast cancer risk prediction models.

---

## 🔬 Scope

The framework focuses on **image-based risk prediction** from mammography and supports:

- Single-view and multi-view inputs  
- Single-timepoint and longitudinal data  
- Breast-wise and patient-wise modeling  

---

## 🎯 Objectives

The main goals of this repository are:

- To **standardize model implementations**  
- To enable **fair comparison across methods**  
- To provide a **reproducible research framework**  
- To accelerate progress in breast cancer risk prediction  

---

## 📊 Supported Tasks

The framework enables:

- Training models end-to-end  
- Evaluating risk prediction performance  
- Comparing models under consistent settings  

---

## 🧠 Implemented Models

| Model | Venue | Input | Key Idea |
|------|------|------|----------|
| **LMV-Net** | MICCAI 2026 (submitted) | Multi-view, longitudinal | Multi-view longitudinal model with dual-stream attention leveraging both views of one breast. |
| **ImgFeatAlign** | MICCAI 2025 | Single-view, longitudinal | Uses image-based deformation (MammoRegNet) applied in feature space for improved longitudinal comparison. |
| **VMRA-MaR** | MICCAI 2025 | Multi-view, longitudinal | Extends Mirai to longitudinal mammograms using Spatial Asymmetry Detector and Longitudinal Asymmetry Tracker. |
| **OA-BReaCR** | MICCAI 2024 | Single-view, longitudinal | Learns longitudinal changes using feature-based deformation fields for better temporal alignment. |
| **Mirai** | Sci. Transl. Med. 2021 | Multi-view, single-timepoint | Learns breast cancer risk from all four views in a single visit |

---

## 🔍 Model Comparison

### Input Representation

**Single-view models:**  
  - OA-BReaCR  
  - ImgFeatAlign  

<br>

**Multi-view models:**  
  - Mirai  
  - VMRA-MaR  
  - LMV-Net  

---

### Temporal Modeling

**Single-timepoint:**  
  - Mirai  

<br>

**Longitudinal (multi-timepoint):**  
  - VMRA-MaR  
  - OA-BReaCR  
  - ImgFeatAlign  
  - LMV-Net  

---

### Aggregation Strategy

**Patient-wise (all views jointly):**  
  - Mirai  
  - VMRA-MaR  

<br>

**Breast-wise modeling:**  
  - LMV-Net  

---

## 🏗️ Framework Design

All models in this repository:

- Share a **common data interface**  
- Use a **unified training pipeline**  
- Support **consistent evaluation metrics**  

This ensures fair comparison and simplifies experimentation.

---

## 🚀 Extensibility

The framework is designed to be easily extended:

- Add new models with minimal changes  
- Integrate additional datasets  
- Modify training and evaluation pipelines  

---

## 📚 Next Steps

- Explore the **Datasets** section for data preparation  
- Review individual **Model** pages for architecture details  
- Use the **Training** and **Evaluation** guides to run experiments  