#!/bin/bash -l

#SBATCH --job-name=Test_Mirai_EMBED         # Job name
#SBATCH --output=/scratch/project_465002309/thrunsol/VMaR_Risk_prediction_test_results_1664_2048/error_output_files/embed/%x-%j.out  # Output file with job name and ID
#SBATCH --error=/scratch/project_465002309/thrunsol/VMaR_Risk_prediction_test_results_1664_2048/error_output_files/embed/%x-%j.err   # Error file with job name and ID
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:59:00               # Run time (hh:mm:ss)
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

# Run the training script using Singularity
export PYTHONPATH=$WORKING_DIR
mkdir -p  /scratch/project_465002309/thrunsol/VMaR_Risk_prediction_test_results_1664_2048/embed/

# Run the training script using Singularity
export PYTHONPATH=$WORKING_DIR


accelerate launch  main_test.py \
  --csv_file /scratch/project_465002309/thrunsol/embed_datasets/combined_cases_with_followup_race.csv \
  --data_root /scratch/project_465002309/thrunsol/embed_datasets/risk_dataset_1664_2048 \
  --path_out_dir /scratch/project_465002309/thrunsol/LMV_Risk_prediction_training_results_1664_2048_test_unified_github/embed/Train_Risk_VMRA_MaR-16747214_Model_VMRA-MaR_lr_5e-05_wd_0.0001_epochs_30_bs_2_2026-03-15-19-48 \
  --img_encoder_snapshot /scratch/project_465002309/thrunsol/mirai_pretrained_backbone/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p \
  --path_test_folder  /scratch/project_465002309/thrunsol/VMaR_Risk_prediction_test_results_1664_2048/embed \
  --model "VMRA-MaR" \
  --use_asymmetry \
  --use_sad_bias \
  --use_lat_bn \
  --batch_size 1 \
  --num_workers 7 \
  --best_model "True" \
  --dataset "EMBED" \
  --seed 2023 \







