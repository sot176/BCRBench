#!/bin/bash

set -e

# Repository root

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# User-configurable paths

CSV_FILE=${CSV_FILE:-"/path/to/metadata.csv"}
DATA_ROOT=${DATA_ROOT:-"/path/to/images"}
TRAINED_MODEL_DIR=${TRAINED_MODEL_DIR:-"/path/to/trained_model"}
TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR:-"$REPO_ROOT/test_outputs"}
CHECKPOINT=${CHECKPOINT:-"/path/to/mirai_checkpoint.p"}

# Dataset selection

DATASET="EMBED"  # Options: "CSAW" or "EMBED"

mkdir -p "$TEST_OUTPUT_DIR"

export PYTHONPATH="$REPO_ROOT"

accelerate launch main_test.py 
--csv_file "$CSV_FILE" 
--data_root "$DATA_ROOT" 
--path_out_dir "$TRAINED_MODEL_DIR" 
--img_encoder_snapshot "$CHECKPOINT" 
--path_test_folder "$TEST_OUTPUT_DIR" 
--model "VMRA-MaR" 
--id_training 1 
--best_model "True" 
--batch_size 1 
--num_workers 7 
--dataset "$DATASET" 
--seed 2023
