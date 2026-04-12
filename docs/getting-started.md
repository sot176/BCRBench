# Getting Started

This guide explains how to set up the repository, prepare your environment, and run training and evaluation for breast cancer risk prediction models.

---

## ⚙️ Installation

Clone the repository and set up a Python environment:

    git clone https://github.com/sot176/BreastCancerRiskBenchmark.git
    cd BreastCancerRiskBenchmark

    conda create -n bc_risk python=3.12
    conda activate bc_risk

    pip install -r requirements.txt

---

## 📁 Configuration

All experiments are controlled through scripts located in the `scripts/` directory.

> ⚠️ **Important:**  
> Before running any script, update the paths inside the scripts to:
> - point to your dataset location  
> - specify output directories for logs and results  

---

## 🧠 Training Models

Each model has a dedicated training script.

Run a model using:

    scripts/train_mirai.sh
    scripts/train_vmra_mar.sh
    scripts/train_oa_breacr.sh
    scripts/train_imgfeatalign.sh
    scripts/train_lmv_net.sh

These scripts handle:
- data loading  
- model initialization  
- training procedure  
- checkpoint saving  

---

## 📊 Evaluation

Each model also provides a corresponding evaluation script:

    scripts/test_mirai.sh
    scripts/test_vmra_mar.sh
    scripts/test_oa_breacr.sh
    scripts/test_imgfeatalign.sh
    scripts/test_lmv_net.sh

Evaluation includes:
- performance metrics (e.g., AUC)  
 
---

## 🧪 Typical Workflow

A standard workflow for running experiments:

1. Prepare dataset in the required CSV format  (See **Datasets** Section of the documentation)
2. Configure dataset paths in the scripts  
3. Train a model  
4. Evaluate the trained model  
5. Compare results across models  

---

## 📚 Available Models

The framework includes implementations of:

- **LMV-Net**  
- **ImgFeatAlign** 
- **VMRA-MaR**
- **OA-BreaCR**  
- **Mirai**  
  
 

See the **Models** section for details on each architecture.

---

## ❗ Notes

- Ensure consistent preprocessing across datasets  
- Verify CSV formatting before training  
- GPU acceleration is recommended for training  

---
 