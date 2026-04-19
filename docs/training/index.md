# Training Overview

## 📌 Overview

This chapter documents the training pipeline for breast cancer risk prediction models. The pipeline supports multiple model architectures, distributed multi-GPU training, and full experiment tracking via Weights & Biases.

Training is handled through a shared pipeline that manages:

- Model initialization  
- Data loading and preprocessing  
- Optimization and scheduling  
- Evaluation and metrics  
- Checkpointing and logging  

---

## Supported Models

| Model | Script |
|---|---|
| **LMV-Net** | `scripts/train_lmv_net.sh` |
| **ImgFeatAlign** | `scripts/train_imgfeatalign.sh` |
| **VMRA-MaR** | `scripts/train_vmra_mar.sh` |
| **OA-BreaCR** | `scripts/train_oa_breacr.sh` |
| **Mirai** | `scripts/train_mirai.sh` |

Each model has its own shell script under `scripts/` that sets the appropriate arguments and launches training.

---

## Key Features

**Modularity:** All models plug into a shared training loop via `BaseRiskModel`  

**Distributed training** — Built on [Hugging Face Accelerate](https://huggingface.co/docs/accelerate), supporting multi-GPU and multi-node training with `DistributedDataParallel`. Mixed precision is handled transparently.

**Learning rate scheduling** — A linear warmup phase runs for a configurable number of steps, after which an optional `ReduceLROnPlateau` scheduler takes over, reducing the learning rate when validation C-index plateaus.

**Differential learning rates** — Pretrained encoder parameters are trained at a lower learning rate (10× smaller by default) than newly added modules, preventing catastrophic forgetting during fine-tuning.

**Data augmentation** — An optional Kornia-based augmentation pipeline applies random cropping, resizing, affine transforms, colour jitter, and gamma correction during training.

**Checkpointing and resumption** — Checkpoints are saved every 10 epochs and whenever a new best validation C-index is achieved. Training can be resumed from any checkpoint, restoring model weights, optimizer state, scheduler state, and global step.

**Early stopping** — Training halts automatically if validation C-index does not improve for a configurable number of epochs, saving the model state at the stopping point.

**Experiment tracking** — All metrics (loss, C-index, per-year AUC for years 1–5) are logged to Weights & Biases per epoch. Runs can be resumed by passing the WandB run ID.

---

## Code Structure

The pipeline is split across three files:

```
main_train.py          # Entry point: argument parsing, Accelerator init, data loading
train/
  train_risk_prediction.py       # Main training loop: epoch loop, checkpointing, early stopping
  train_utils.py # Core functions: train_one_epoch, evaluate, schedulers, checkpoints
```

`main_train.py` is the script you call directly (via the shell scripts). It parses arguments, initialises the Accelerator, creates data loaders, and hands off to `train_val()` in `train/train_risk_prediction.py`. That function runs the epoch loop and delegates per-epoch computation to `train_one_epoch()` and `evaluate()` in `train_utils.py`.

---

## Quick Start

Pick the script for your model and run it:

```bash
bash scripts/train_lmv_net.sh
bash scripts/train_imgfeatalign.sh
bash scripts/train_vmra_mar.sh
bash scripts/train_oa_breacr.sh
bash scripts/train_mirai.sh
```

For a detailed walkthrough of arguments, configuration, and what happens during training, see [Training a Model](training_a_model.md).
