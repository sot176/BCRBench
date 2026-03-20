#!/bin/bash -l

#SBATCH --job-name=Train_Risk_VMRA_MaR         # Job name
#SBATCH --output=/scratch/project_465002309/thrunsol/LMV_Risk_prediction_training_results_1664_2048_test_unified_github/error_output_files/embed/%x-%j.out  # Output file with job name and ID
#SBATCH --error=/scratch/project_465002309/thrunsol/LMV_Risk_prediction_training_results_1664_2048_test_unified_github/error_output_files/embed/%x-%j.err   # Error file with job name and ID
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=10:59:00               # Run time (hh:mm:ss)
#SBATCH --account=project_465002309     # Project for billing


# Directories for data and output
export WORKING_DIR=/scratch/project_465002309/thrunsol/BreastCancerRiskBenchmark/src
export TORCH_HOME=/scratch/${SLURM_JOB_ACCOUNT}/thrunsol/torch-cache
export HF_HOME=/flash/${SLURM_JOB_ACCOUNT}/thrunsol/hf-cache
mkdir -p $TORCH_HOME $HF_HOME
# Load modules
module purge

# Define paths and Singularity container
export PATH="/projappl/project_465002309/thrunsol/my_env/MyPytorchEnvRiskPred/bin:$PATH"

# Update the repository
cd $WORKING_DIR
git reset --hard
git pull || { echo "Git pull failed!"; exit 1; }
echo "Repository updated successfully. Starting training..."

# Set the WANDB API Key and login
export WANDB_API_KEY=0101399a4b7125a822a5262b07f7de3cbaf647d9
wandb login

# Echo job details for debugging
echo "Starting job $SLURM_JOB_NAME-$SLURM_JOB_ID with ${SLURM_GPUS_PER_NODE} GPUs per noe"
set -xv  # Print commands for debugging

# Run the training script using Singularity
export PYTHONPATH=$WORKING_DIR
accelerate launch --mixed_precision=bf16  main_train.py \
            --csv_file /scratch/project_465002309/thrunsol/embed_datasets/combined_cases_with_followup_race.csv \
            --data_root /scratch/project_465002309/thrunsol/embed_datasets/risk_dataset_1664_2048 \
            --path_out_dir /scratch/project_465002309/thrunsol/LMV_Risk_prediction_training_results_1664_2048_test_unified_github/embed/$SLURM_JOB_NAME-$SLURM_JOB_ID \
            --img_encoder_snapshot /scratch/project_465002309/thrunsol/mirai_pretrained_backbone/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p \
            --id_training 1 \
            --use_scheduler "True" \
            --batch_size 2 \
            --augmentations "True" \
            --num_workers 7 \
            --learning_rate 1e-4 \
            --weight_decay 1e-4 \
            --warmup_steps 0 \
            --model "VMRA-MaR" \
            --use_asymmetry \
            --lr_decay 0.5 \
            --patience_lr_scheduler 5 \
            --num_epochs 30 \
            --patience 15 \
            --dataset "EMBED" \
            --seed 2023



