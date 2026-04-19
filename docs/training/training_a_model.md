# Training a Model

This page walks through everything you need to run a training job: prerequisites, configuration, launching training, and understanding what happens at each stage.

---

## Prerequisites

**Environment** — Ensure all dependencies are installed. The pipeline requires PyTorch, Hugging Face Accelerate, Kornia, and WandB, among others.

**Data** — Training expects a CSV file with pre-defined `train` / `val` split columns, and a root directory containing the image data. The exact CSV schema depends on the dataset (`EMBED` or `CSAW`).

**Model config** — Each model has a YAML configuration file under `config/models/`. For example, `OA-BreaCR` loads from `config/models/oa_breacr.yaml`. These files define model-specific hyperparameters (e.g. `distance`, `alpha_coeff`, `margin`) that extend the base CLI arguments. If no YAML is found, the pipeline falls back to CLI defaults.

**Registration model** — `ImgFeatAlign` and `LMV-Net` require a pretrained `MammoRegNet` registration model. The path is set in `config/config.py` under `paths.csaw_path_saved_reg_model` or `paths.embed_path_saved_reg_model` depending on the dataset.

---

## Running Training

Each model has a shell script that sets all required arguments:

```bash
bash scripts/train_lmv_net.sh
bash scripts/train_imgfeatalign.sh
bash scripts/train_vmra_mar.sh
bash scripts/train_oa_breacr.sh
bash scripts/train_mirai.sh
```

Use `accelerate launch` (rather than `python`) to enable multi-GPU training. 

---

## CLI Arguments

### Required

| Argument | Description |
|---|---|
| `--model` | Model name, e.g. `Mirai`, `OA-BreaCR`, `VMRA-MaR`, `ImgFeatAlign`, `LMV-Net` |
| `--dataset` | Dataset name: `EMBED` or `CSAW` |
| `--csv_file` | Path to CSV file containing data splits |
| `--data_root` | Root directory of image data |
| `--path_out_dir` | Base output directory (a timestamped subdirectory is created automatically) |

### Key Optional

| Argument | Default | Description |
|---|---|---|
| `--num_epochs` | 100 | Number of training epochs |
| `--batch_size` | 12 | Batch size per GPU |
| `--learning_rate` | 5e-5 | Base learning rate for new modules |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--warmup_steps` | 5000 | Number of linear warmup steps |
| `--use_scheduler` | False | Enable `ReduceLROnPlateau` after warmup |
| `--patience_lr_scheduler` | 5 | Epochs without improvement before LR reduction |
| `--lr_decay` | 0.5 | LR reduction factor for plateau scheduler |
| `--patience` | 15 | Epochs without improvement before early stopping |
| `--augmentations` | False | Enable Kornia data augmentation |
| `--resume_from` | None | Path to checkpoint file to resume from |
| `--wandb_id` | None | WandB run ID for resuming an existing experiment |
| `--seed` | 2023 | Random seed for reproducibility |

---

## Configuration System

Argument loading happens in two stages:

1. **CLI arguments** are parsed first, including the `--model` flag.
2. **Model YAML** is loaded from `config/models/<model_name>.yaml` and its keys are added as additional CLI arguments with their YAML values as defaults.

This means YAML values can always be overridden from the command line. For example, if `oa_breacr.yaml` sets `margin: 1.0`, you can override it with `--margin 2.0` at runtime.

---

## Training Pipeline

When you launch training, the following happens in order:

### 1. Argument Parsing and Setup

`main.py` parses CLI arguments, loads the model YAML config, constructs a timestamped output directory path of the form:

```
{path_out_dir}_Model_{model}_lr_{lr}_wd_{wd}_epochs_{n}_bs_{bs}_{timestamp}/
```

### 2. Accelerator Initialisation

A Hugging Face `Accelerator` is created with `find_unused_parameters=True` to handle models where not all parameters participate in every forward pass. This transparently manages device placement, mixed precision, and gradient synchronisation across GPUs.

### 3. Reproducibility

Seeds are set for Python `random`, `torch`, and `torch.cuda` on all processes. cuDNN is set to deterministic mode.

### 4. Data Loading

`get_dataset_and_loader()` creates training and validation `DataLoader`s. Accelerate shards the training data across GPUs automatically — `len(train_loader)` already reflects the per-GPU length. No augmentation is applied to the validation set.

### 5. Model Initialisation

The model is loaded via `get_model()` from `models/model_factory.py`. Parameter groups are then created with **differential learning rates**:

- **New modules** (non-encoder parameters): trained at `learning_rate`
- **Encoder parameters**: trained at `learning_rate × 0.1` (or `× 1.0` for OA-BreaCR, which is trained from scratch)

This prevents a pretrained encoder from being destabilised early in training.

### 6. Optimiser and Schedulers

An `AdamW` optimiser is created over the parameter groups. Two schedulers are set up:

- **Warmup scheduler** (`LambdaLR`): linearly ramps the LR from 0 to the base LR over `warmup_steps` steps. Steps on every gradient update.
- **Plateau scheduler** (`ReduceLROnPlateau`, optional): reduces LR by `lr_decay` when validation C-index does not improve for `patience_lr_scheduler` epochs. Only activated after warmup completes.

All components are wrapped with `accelerator.prepare()` before training begins.

### 7. Epoch Loop

For each epoch:

**Training** (`train_one_epoch`) — Runs the forward pass, computes loss, calls `accelerator.backward()`, and steps the optimiser. The warmup scheduler steps on every batch while `global_step < warmup_steps`. Predictions and labels are gathered across GPUs, and C-index and per-year AUC are computed on the main process.

**Validation** (`evaluate`) — Runs inference under `torch.no_grad()`. Loss, C-index, and per-year AUC are computed identically to training.

**Logging** — On the main process, metrics are written to the log file and to WandB.

**Scheduler step** — If `use_scheduler` is enabled and `global_step >= warmup_steps`, the plateau scheduler steps on the validation C-index.

**Checkpointing and early stopping** — See the sections below.

---

## Learning Rate Scheduling

The warmup and plateau schedulers work in two non-overlapping phases:

```
Steps 0 → warmup_steps    LR ramps linearly from 0 → base_lr (warmup_scheduler steps every batch)
Steps > warmup_steps       LR held constant unless val C-index plateaus (plateau scheduler steps every epoch)
```

The plateau scheduler is guarded so it cannot reduce the LR during the warmup phase, preventing premature LR decay before the model has stabilised.

---

## Checkpointing

Checkpoints are saved to the output directory in two situations:

- **Every 10 epochs** — `checkpoint_{epoch:04d}.pth`
- **New best validation C-index** — `best_model_risk_prediction_id-{id}.pth`

Each checkpoint contains the model state dict, optimiser state, both scheduler states, current epoch, global step, and best C-index achieved.

To resume training from a checkpoint:

```bash
accelerate launch main_train.py \
  ... \
  --resume_from /path/to/checkpoint_0050.pth \
  --wandb_id <your-wandb-run-id>
```

The `--wandb_id` flag resumes the associated WandB run so metrics are appended to the existing experiment rather than starting a new one.

---

## Early Stopping

If validation C-index does not improve for `--patience` consecutive epochs (default 15), training stops and the model at that point is saved as:

```
early_stopping_risk_prediction_id-{id}.pth
```

The best model checkpoint (saved whenever a new best C-index was achieved) is always preserved separately.

---

## Output Directory Contents

After training completes, the output directory contains:

```
{results_dir}/
  train_risk_prediction_training_id_{id}.log       # Full training log
  checkpoint_0010.pth                              # Periodic checkpoints (every 10 epochs)
  checkpoint_0020.pth
  ...
  best_model_risk_prediction_id-{id}.pth           # Best validation C-index checkpoint
  early_stopping_risk_prediction_id-{id}.pth       # Present only if early stopping triggered
  model_risk_prediction_training_id_{id}_last_epoch.pth  # Final model weights
```

---

## Data Augmentation

Enable augmentation with `--augmentations`. The following transforms are applied **to training data only**, in sequence:

- `RandomCrop` to (1946 × 1581) with p=0.2
- `Resize` to (2048 × 1664)
- `RandomAffine` — translation up to 10%, scale up to 1.1×, with p=0.5
- `ColorJitter` — brightness, contrast, saturation ±0.4, with p=0.5
- `RandomGamma` — gamma in (0.8, 1.2), with p=0.5

Validation loaders never receive augmentation.

---

## Experiment Tracking (WandB)

The following metrics are logged to WandB per epoch:

- `Training Risk Loss`, `Validation Risk Loss`
- `Training C-index`, `Validation C-index`
- `Train Year {1–5} AUC`, `Val Year {1–5} AUC`

All metrics use `epoch` as their step metric. To resume a WandB run, pass `--wandb_id <run-id>` when resuming from a checkpoint.
