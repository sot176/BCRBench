# 🧪 Evaluating a Model

This guide walks you through everything needed to run an evaluation job: prerequisites, configuration, launching evaluation, and understanding what happens at each stage.

---

## ✅ Prerequisites

**🧪 Environment**  
Ensure all dependencies are installed. The pipeline requires:

- PyTorch  
- Hugging Face Accelerate  

**📂 Data**  
Evaluation expects:

- A CSV file with a predefined `test` split column  
- A root directory containing the image data  

The exact CSV schema depends on the dataset (`EMBED` or `CSAW`).

**⚙️ Model configuration**  
Each model has a YAML configuration file under `config/models/`. For example, `LMV-Net` uses:

```
config/models/lmv_net.yaml
```

These files define model-specific hyperparameters that extend the base CLI arguments. If no YAML is found, the pipeline falls back to CLI defaults.

**🧩 Registration model**  
`ImgFeatAlign` and `LMV-Net` require a pretrained `MammoRegNet` registration model. The path is defined in:

```
config/config.py
```

- `paths.csaw_path_saved_reg_model`  
- `paths.embed_path_saved_reg_model`  

(depending on the dataset)

**🏋️ Trained model checkpoint**  
A trained model checkpoint is required. You can use either:

- The **best** checkpoint: `best_model_risk_prediction_id-{id}.pth`  
- The **last epoch** checkpoint: `model_risk_prediction_training_id_{id}_last_epoch.pth`  

Select between them using `--best_model True/False`.

---

## 🚀 Running Evaluation

Each model has a dedicated shell script that sets all required arguments:

```bash
bash scripts/test_lmv_net.sh
bash scripts/test_imgfeatalign.sh
bash scripts/test_vmra_mar.sh
bash scripts/test_oa_breacr.sh
bash scripts/test_mirai.sh
```

👉 Use `accelerate launch` (instead of `python`) to enable multi-GPU evaluation.

---

## 🧾 CLI Arguments

### 🔴 Required

| Argument | Description |
|---|---|
| `--model` | Model name (e.g. `Mirai`, `OA-BreaCR`, `VMRA-MaR`, `ImgFeatAlign`, `LMV-Net`) |
| `--dataset` | Dataset name: `EMBED` or `CSAW` |
| `--csv_file` | Path to CSV file containing the test split |
| `--data_root` | Root directory of image data |
| `--path_out_dir` | Directory where the trained model checkpoint is stored |
| `--path_test_folder` | Output directory for evaluation results and logs |
| `--id_training` | Training run ID used to resolve the checkpoint filename |

### 🟡 Key Optional

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 20 | Batch size per GPU |
| `--num_workers` | 4 | DataLoader worker processes |
| `--best_model` | — | If `True`, loads the best checkpoint; otherwise loads the last epoch |
| `--seed` | — | Random seed for reproducibility |

---

## ⚙️ Configuration System

Argument loading occurs in two stages:

1. **CLI arguments** are parsed first (including `--model`)
2. **Model YAML config** is loaded from:

```
config/models/<model_name>.yaml
```

YAML values are added as CLI defaults and can always be overridden on the command line.

---

## 🔄 Evaluation Pipeline

When evaluation starts, the following steps occur:

### 1️⃣ Argument Parsing & Setup

`main_test.py`:

- Parses CLI arguments  
- Loads YAML config  
- Resolves the checkpoint path based on `--id_training` and `--best_model`:

```
best_model_risk_prediction_id-{id}.pth           # if --best_model True
model_risk_prediction_training_id_{id}_last_epoch.pth  # otherwise
```

---

### 2️⃣ Accelerator Initialisation

A Hugging Face `Accelerator` is created, enabling:

- Multi-GPU inference  
- Automatic device placement  
- Distributed tensor gathering across processes  

---

### 3️⃣ 🔁 Reproducibility

If `--seed` is provided, seeds are set for:

- `random`  
- `torch`  
- `torch.cuda`  

---

### 4️⃣ 📦 Data Loading

`get_dataset_and_loader()` creates a test `DataLoader` for the `test` split.

👉 No augmentation is applied during evaluation.

---

### 5️⃣ 🧠 Model Loading

Handled by `load_model()` in `evaluate/test_utils.py`:

1. Model is instantiated via `models/model_factory.py`
2. Checkpoint is loaded from disk (`map_location="cpu"`)
3. State dict is extracted — supports checkpoints saved as:
   - Raw state dict  
   - Dict with `"model"` or `"state_dict"` key  
4. `module.` prefixes are stripped (from `DataParallel` wrapping)
5. Weights are loaded and model is set to `eval()` mode

The model and test loader are then wrapped with:

```python
accelerator.prepare()
```

---

### 6️⃣ 🔍 Inference Loop

Runs under `torch.no_grad()`. For each batch, the following are collected and gathered across all GPUs:

- `preds` — risk predictions from the primary risk head  
- `event_times` — follow-up times  
- `event_observed` — event indicators  
- `densities` — breast density categories  
- `cancer_types` — cancer type categories  
- `races` — race IDs (EMBED dataset only)  

---

### 7️⃣ 📊 Aggregation & Metric Computation

After inference, the main process:

1. Concatenates all gathered tensors  
2. Computes the **censoring distribution** from event times and indicators  
3. Saves raw predictions and metadata to disk via `save_model_results_to_file()`  
4. Computes all metrics with **bootstrapped 95% confidence intervals**:

| Metric | Stratification |
|---|---|
| C-index | Overall |
| AUC (years 1–5) | Overall |
| AUC | By breast density |
| C-index | By breast density |
| AUC | By cancer type |
| C-index | By cancer type |
| AUC | By race (EMBED only) |
| C-index | By race (EMBED only) |

---

## 📁 Output Directory

```
{path_test_folder}/
  test_risk_prediction_training_id_{id}.log
  c_index_by_density.json
  c_index_by_cancer_type.json
  c_index_by_race.json        # EMBED only
```

All per-subgroup C-index results are saved as JSON files for downstream analysis.

---

## 📋 Final Results

At the end of evaluation, a structured results dict is printed and written to the log file:

```
C-index:              Mean + 95% CI
Yearly AUCs:          Years 1–5, Mean + 95% CI
AUC by density:       Per density category  Mean + 95% CI
C-index by density:   Per density category  Mean + 95% CI
AUC by cancer type:   Per cancer type  Mean + 95% CI
C-index by cancer type: Per cancer type  Mean + 95% CI
AUC by race:          Per race group (EMBED only)  Mean + 95% CI
C-index by race:      Per race group (EMBED only)  Mean + 95% CI
```
