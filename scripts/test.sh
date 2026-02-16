#!/bin/bash -l


# Define placeholder variables for paths
CSV_FILE_PATH="PATH_TO_CSV_FILE"
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TEST_FOLDER_PATH="PATH_TO_TEST_FOLDER"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="CSAW"  # or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"


accelerate launch  main_test.py \
            --csv_file "$CSV_FILE_PATH"  \
            --data_root "$DATA_ROOT_PATH"  \
            --path_out_dir "$OUTPUT_DIR_PATH" \
            --path_test_folder "$TEST_FOLDER_PATH" \
            --id_training "$TRAINING_ID" \
            --batch_size 8 \
            --num_workers 7 \
            --dataset "$DATASET" \
            --dropout_perceiver 0.3 \
            --seed 2023

