# 🧠 Training a Model

This guide walks you through everything needed to run a training job: prerequisites, configuration, launching training, and understanding what happens at each stage.

---

## ✅ Prerequisites

**🧪 Environment**  
Ensure all dependencies are installed. The pipeline requires:

- PyTorch  
- Hugging Face Accelerate  
- Kornia  
- Weights & Biases (WandB)  

**📂 Data**  
Training expects:

- A CSV file with predefined `train` / `val` split columns  
- A root directory containing the image data  

The exact CSV schema depends on the dataset (`EMBED` or `CSAW`).

**⚙️ Model configuration**  
Each model has a YAML configuration file under `config/models/`. For example, `LMV-Net` uses:

```
config/models/lmv_net.yaml
```

These files define model-specific hyperparameters (e.g. `distance`, `alpha_coeff`, `margin`) that extend the base CLI arguments. If no YAML is found, the pipeline falls back to CLI defaults.

**🧩 Registration model**  
`ImgFeatAlign` and `LMV-Net` require a pretrained `MammoRegNet` registration model. The path is defined in:

```
config/config.py
```

- `paths.csaw_path_saved_reg_model`  
- `paths.embed_path_saved_reg_model`  

(depending on the dataset)

---

## 🚀 Running Training

Each model has a dedicated shell script that sets all required arguments:

```bash
bash scripts/train_lmv_net.sh
bash scripts/train_imgfeatalign.sh
bash scripts/train_vmra_mar.sh
bash scripts/train_oa_breacr.sh
bash scripts/train_mirai.sh
```

👉 Use `accelerate launch` (instead of `python`) to enable multi-GPU training.

---

## 🧾 CLI Arguments

### 🔴 Required

| Argument | Description |
|---|---|
| `--model` | Model name (e.g. `Mirai`, `OA-BreaCR`, `VMRA-MaR`, `ImgFeatAlign`, `LMV-Net`) |
| `--dataset` | Dataset name: `EMBED` or `CSAW` |
| `--csv_file` | Path to CSV file containing data splits |
| `--data_root` | Root directory of image data |
| `--path_out_dir` | Base output directory (timestamped subdirectory is created automatically) |

---

### 🟡 Key Optional

| Argument | Default | Description |
|---|---|---|
| `--num_epochs` | 100 | Number of training epochs |
| `--batch_size` | 12 | Batch size per GPU |
| `--learning_rate` | 5e-5 | Learning rate for newly added modules |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--warmup_steps` | 5000 | Number of warmup steps |
| `--use_scheduler` | True | Enable `ReduceLROnPlateau` |
| `--patience_lr_scheduler` | 5 | Epochs before LR reduction |
| `--lr_decay` | 0.5 | LR reduction factor |
| `--patience` | 15 | Early stopping patience |
| `--augmentations` | True | Enable data augmentation |
| `--resume_from` | None | Path to checkpoint |
| `--wandb_id` | None | WandB run ID for resuming |
| `--seed` | 2023 | Random seed |

---

## ⚙️ Configuration System

Argument loading occurs in two stages:

1. **CLI arguments** are parsed first (including `--model`)
2. **Model YAML config** is loaded from:

```
config/models/<model_name>.yaml
```

YAML values are added as CLI defaults and can always be overridden.

**Example:**
```bash
--dropout 0.3
```
overrides:
```yaml
dropout: 0.1
```

---

## 🔄 Training Pipeline

When training starts, the following steps occur:

### 1️⃣ Argument Parsing & Setup

`main_train.py`:

- Parses CLI arguments  
- Loads YAML config  
- Creates a timestamped output directory:

```
{path_out_dir}_Model_{model}_lr_{lr}_wd_{wd}_epochs_{n}_bs_{bs}_{timestamp}/
```

---

### 2️⃣ Accelerator Initialisation

A Hugging Face `Accelerator` is created:

This enables:

- Multi-GPU training  
- Mixed precision  
- Automatic device placement  
- Gradient synchronisation  

---

### 3️⃣ 🔁 Reproducibility

Seeds are set for:

- `random`  
- `torch`  
- `torch.cuda`  

cuDNN runs in deterministic mode.

---

### 4️⃣ 📦 Data Loading

`get_dataset_and_loader()`:

- Creates training and validation `DataLoader`s  
- Automatically shards training data across GPUs  

👉 `len(train_loader)` already reflects per-GPU size  
👉 No augmentation is applied to validation data  

---

### 5️⃣ 🧠 Model Initialisation

Models are loaded via:

```
models/model_factory.py
```

**Differential learning rates:**

- 🔹 New modules → `learning_rate`  
- 🔹 Encoder → `learning_rate × 0.1`  (for pretrained Mirai encoder)
- 🔹 OA-BreaCR → Encoder trained from scratch (no reduction)  

This prevents destabilising pretrained encoders.

---

### 6️⃣ 📉 Optimiser & Schedulers

**Optimiser:**  
- AdamW over parameter groups  

**Schedulers:**

- 🔥 **Warmup (LambdaLR)**  
    - Linear increase from 0 → base LR  
    - Steps every batch  

- ⏳ **Plateau (ReduceLROnPlateau, optional)**  
    - Reduces LR when validation C-index plateaus  
    - Activates only after warmup  

All components are wrapped with:

```python
accelerator.prepare()
```

---

### 7️⃣ 🔁 Epoch Loop

Each epoch consists of:

**🏋️ Training (`train_one_epoch`)**

- Forward pass  
- Loss computation  
- `accelerator.backward()`  
- Optimiser step  
- Warmup scheduler step (per batch)  

Metrics:

- C-index  
- Per-year AUC  

---

**🧪 Validation (`evaluate`)**

- Runs under `torch.no_grad()`  
- Computes:
  - Loss  
  - C-index  
  - Per-year AUC  

---

**📊 Logging**

- Written to log file  
- Logged to WandB (main process only)

---

**📉 Scheduler Step**

- Plateau scheduler updates after warmup  
- Uses validation C-index  

---

**💾 Checkpointing & Early Stopping**

- Handled automatically (see below)

---

## 📉 Learning Rate Scheduling

Two-phase schedule:

```
Steps 0 → warmup_steps     LR increases linearly (per batch)
Steps > warmup_steps       LR constant or reduced on plateau (per epoch)
```

The plateau scheduler is disabled during warmup to avoid premature LR decay.

---

## 💾 Checkpointing

Checkpoints are saved:

- 📌 Every 10 epochs  
- 🏆 When a new best validation C-index is achieved  

**Files:**
```
checkpoint_{epoch:04d}.pth
best_model_risk_prediction_id-{id}.pth
```

Each checkpoint includes:

- Model state  
- Optimiser state  
- Scheduler states  
- Epoch  
- Global step  
- Best C-index  

---

### 🔄 Resume Training

```bash
accelerate launch main_train.py \
  ... \
  --resume_from /path/to/checkpoint_0050.pth \
  --wandb_id <run-id>
```

👉 `--wandb_id` ensures metrics continue in the same experiment

---

## ⏹️ Early Stopping

Triggered if validation C-index does not improve for `--patience` epochs.

Saved as:

```
early_stopping_risk_prediction_id-{id}.pth
```

👉 Best model is always saved separately

---

## 📁 Output Directory

```
{results_dir}/
  train_risk_prediction_training_id_{id}.log
  checkpoint_0010.pth
  checkpoint_0020.pth
  ...
  best_model_risk_prediction_id-{id}.pth
  early_stopping_risk_prediction_id-{id}.pth
  model_risk_prediction_training_id_{id}_last_epoch.pth
```

---

## 🖼️ Data Augmentation

Enable with:

```bash
--augmentations
```

Applied **only to training data**:

- RandomCrop (1946 × 1581), p=0.2  
- Resize (2048 × 1664)  
- RandomAffine (translation ±10%, scale up to 1.1×), p=0.5  
- ColorJitter (±0.4), p=0.5  
- RandomGamma (0.8–1.2), p=0.5  

👉 No augmentation is applied to validation data

---

## 📊 Experiment Tracking (WandB)

Logged per epoch:

- Training & Validation Loss  
- Training & Validation C-index  
- Year 1–5 AUC (train + validation)  

👉 All metrics use `epoch` as the step  

### 🔄 Resume Logging

```bash
--wandb_id <run-id>
```

This appends metrics to an existing run instead of creating a new one.