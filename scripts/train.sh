#!/bin/bash -l

# Define placeholder variables for paths
CSV_FILE_PATH="PATH_TO_CSV_FILE"
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="CSAW"  # or "EMBED"

accelerate launch  main_train.py \
            --cvs_file "$CSV_FILE_PATH"  \
            --data_root "$DATA_ROOT_PATH"  \
            --path_out_dir "$OUTPUT_DIR_PATH" \
             --id_training "$TRAINING_ID" \
            --use_scheduler "True" \
            --batch_size 4 \
            --augmentations "True" \
            --num_workers 7 \
            --model "LMV-Net" \
            --learning_rate 5e-5 \
            --weight_decay 1e-4 \
            --lr_decay 0.5 \
            --patience_lr_scheduler 5 \
            --patience 15 \
            --num_epochs 30 \
            --dataset "$DATASET" \
            --finetune_all \
            --seed 2023




