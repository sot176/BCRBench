# 🧪 Evaluation

## 📌 Overview

This chapter describes the evaluation pipeline for breast cancer risk prediction models. The pipeline supports multi-GPU inference, stratified metric computation, and results logging.

A unified evaluation framework handles:

- 📂 Data loading for the test split
- 🔍 Model inference and prediction aggregation
- 📊 Metric computation with bootstrapped confidence intervals
- 💾 Results saving and logging

---

## 🤖 Supported Models

| Model | Script |
|---|---|
| **LMV-Net** | `scripts/test_lmv_net.sh` |
| **ImgFeatAlign** | `scripts/test_imgfeatalign.sh` |
| **VMRA-MaR** | `scripts/test_vmra_mar.sh` |
| **OA-BreaCR** | `scripts/test_oa_breacr.sh` |
| **Mirai** | `scripts/test_mirai.sh` |

---

## 🚀 Key Features

**🖥️ Distributed inference**  
Built on Hugging Face Accelerate, supporting multi-GPU evaluation with automatic tensor gathering across processes.

**📦 Flexible model loading**  
Supports loading either the best checkpoint or the final epoch model, selected via the `--best_model` flag.

**📊 Comprehensive metrics**  
All metrics are computed with **bootstrapped 95% confidence intervals**:

- Overall **C-index** and **yearly AUC** (years 1–5)
- Stratified by **breast density**, **cancer type**, and **race** (EMBED dataset)

---

## 🏗️ Code Structure

```text
main_test.py
evaluate/
  test_risk_prediction.py
  test_utils.py
```

- `main_test.py`  
  ▶️ Entry point. Handles argument parsing, initializes the Accelerator, builds the test loader, resolves model checkpoint paths, and launches evaluation.

- `evaluate/test_risk_prediction.py`  
  🔁 Contains the main evaluation logic: inference loop, result aggregation, metric computation, and logging.

- `evaluate/test_utils.py`  
  🧰 Utility functions for model loading (`load_model`) and distributed tensor gathering (`gather_tensor`).

---

## ⚡ Quick Start

To evaluate a model, run the corresponding script:

```bash
bash scripts/test_lmv_net.sh
bash scripts/test_imgfeatalign.sh
bash scripts/test_vmra_mar.sh
bash scripts/test_oa_breacr.sh
bash scripts/test_mirai.sh
```

Each script points to the relevant config under `config/models/<model>.yaml` and sets the required CLI arguments. For a full description of all arguments, see **Evaluating a Model** (`evaluating_a_model.md`).
