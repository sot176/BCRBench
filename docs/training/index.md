# 🧠 Training 

## 📌 Overview

This chapter describes the training pipeline for breast cancer risk prediction models. The pipeline is designed to support multiple model architectures, distributed multi-GPU training, and comprehensive experiment tracking via 📊 Weights & Biases.

A unified training framework handles:

- ⚙️ Model initialization  
- 📂 Data loading and preprocessing  
- 📈 Optimization and learning rate scheduling  
- 🧪 Evaluation and metric computation  
- 💾 Checkpointing and logging  

---

## 🤖 Supported Models

| Model | Script |
|---|---|
| **LMV-Net** | `scripts/train_lmv_net.sh` |
| **ImgFeatAlign** | `scripts/train_imgfeatalign.sh` |
| **VMRA-MaR** | `scripts/train_vmra_mar.sh` |
| **OA-BreaCR** | `scripts/train_oa_breacr.sh` |
| **Mirai** | `scripts/train_mirai.sh` |

Each model is launched via its corresponding shell script in the `scripts/` directory, where model-specific configurations and arguments are defined.

---

## 🚀 Key Features

**🧩 Modular design**  
All models integrate into a shared training loop through a common `BaseRiskModel` interface, enabling easy extensibility and comparison.

**🖥️ Distributed training**  
Built on Hugging Face Accelerate, the pipeline supports multi-GPU and multi-node training using `DistributedDataParallel`. Mixed precision is handled automatically.

**📉 Learning rate scheduling**  
Training begins with a linear warmup phase for a configurable number of steps. This can be followed by an optional `ReduceLROnPlateau` scheduler, which lowers the learning rate when the validation C-index stops improving.

**⚖️ Differential learning rates**  
Pretrained encoder parameters are trained with a lower learning rate (typically 10× smaller) than newly initialized layers, helping to prevent catastrophic forgetting during fine-tuning.

**🖼️ Data augmentation**  
An optional augmentation pipeline based on Kornia applies transformations such as random cropping, resizing, affine transforms, color jitter, and gamma correction during training.

**💾 Checkpointing and resumption**  
Model checkpoints are saved every 10 epochs and whenever a new best validation C-index is achieved. Training can be resumed from any checkpoint, restoring:

- 🧠 Model weights  
- 🔧 Optimizer state  
- 📊 Scheduler state  
- 🔢 Global training step  

**⏹️ Early stopping**  
Training automatically stops if the validation C-index does not improve for a specified number of epochs. The best-performing model is preserved.

**📊 Experiment tracking**  
All training metrics—including loss, C-index, and per-year AUC (years 1–5)—are logged to Weights & Biases. Runs can be resumed using a WandB run ID.

---

## 🏗️ Code Structure

The training pipeline is organized into three main components:

```text
main_train.py
train/
  train_risk_prediction.py
  train_utils.py
```

- `main_train.py`  
  ▶️ Entry point for training. Handles argument parsing, initializes the Accelerator, prepares data loaders, and starts the training process.

- `train/train_risk_prediction.py`  
  🔁 Contains the main training loop, including epoch iteration, checkpointing, and early stopping.

- `train/train_utils.py`  
  🧰 Implements core functionality such as:
  - `train_one_epoch()`
  - `evaluate()`
  - learning rate schedulers
  - checkpoint utilities

The `main_train.py` script delegates execution to `train_val()` in `train_risk_prediction.py`, which in turn calls the lower-level training and evaluation functions.

---

## ⚡ Quick Start

To train a model, run the corresponding script:

```bash
bash scripts/train_lmv_net.sh
bash scripts/train_imgfeatalign.sh
bash scripts/train_vmra_mar.sh
bash scripts/train_oa_breacr.sh
bash scripts/train_mirai.sh
```

For a detailed guide on configuration options, arguments, and the full training workflow, refer to **Training a Model** (`training_a_model.md`).