# Getting Started

This guide walks you through setting up the repository and running training and evaluation for breast cancer risk prediction models.

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

For dataset formatting and preprocessing, see the **Datasets** section.

---

## 🧠 Training

Each model can be trained using its corresponding script:

    scripts/train_mirai.sh
    scripts/train_vmra_mar.sh
    scripts/train_oa_breacr.sh
    scripts/train_imgfeatalign.sh
    scripts/train_lmv_net.sh

---

## 📊 Evaluation

Evaluate trained models using:

    scripts/test_mirai.sh
    scripts/test_vmra_mar.sh
    scripts/test_oa_breacr.sh
    scripts/test_imgfeatalign.sh
    scripts/test_lmv_net.sh

Evaluation scripts compute performance metrics such as AUC.

---

## 🧪 Workflow

Typical usage:

1. Prepare dataset (see **Datasets**)  
2. Update paths in scripts  
3. Train a model  
4. Run evaluation  

---

## ❗ Notes

- Ensure dataset formatting is correct before training  
- Use consistent preprocessing across experiments  
- GPU acceleration is recommended  