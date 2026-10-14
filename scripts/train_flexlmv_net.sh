#!/bin/bash

set -e

# Repository root

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# User-configurable paths

CSV_FILE=${CSV_FILE:-"/path/to/dataset.csv"}
DATA_ROOT=${DATA_ROOT:-"/path/to/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/outputs"}

# Dataset selection

DATASET="EMBED"  # Options: "CSAW" or "EMBED"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$REPO_ROOT"

accelerate launch main_train.py 
--csv_file "$CSV_FILE" 
--data_root "$DATA_ROOT" 
--path_out_dir "$OUTPUT_DIR" 
--id_training 1 
--use_scheduler 
--batch_size 4 
--augmentations 
--num_workers 7 
--learning_rate 5e-5 
--weight_decay 1e-4 
--model "LMV-Net" 
--lr_decay 0.5 
--patience_lr_scheduler 5 
--num_epochs 30 
--patience 15 
--dataset "$DATASET" 
--finetune_all 
--seed 2023
