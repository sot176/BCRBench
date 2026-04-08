import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import yaml

from evaluate import test_risk
from datasets import get_dataset_and_loader



def parse_test_args():
    parser = argparse.ArgumentParser(description="Test-time arguments for risk prediction models")

    # ---------------- Required paths ----------------
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--path_test_folder", type=str, required=True)

    # ---------------- Basic dataset & model options ----------------
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name (mirai, ImgFeatAlign, VMRA-MaR, OA-BreaCR, LMV-Net, etc.)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (EMBED, CSAW-CC, etc.)")

    # ---------------- Temporary parse ----------------
    # Only parse known args to detect the model
    temp_args, _ = parser.parse_known_args()

    # ---------------- Model-specific YAML ----------------
    yaml_path = f"configs/model/{temp_args.model.lower()}.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Merge YAML defaults (CLI args take precedence)
        for k, v in model_config.items():
            if not hasattr(temp_args, k):
                parser.add_argument(f"--{k}", default=v)
    else:
        print(f"[WARNING] YAML config not found for model '{temp_args.model}'. Using CLI/default values.")

    # ---------------- Final parse ----------------
    args = parser.parse_args()

    return args


def main():
    args = parse_test_args()
    accelerator = Accelerator()

    # Set seed for reproducibility on all processes
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    test_loader = get_dataset_and_loader(
        dataset_name=args.dataset,
        model_name=args.model,
        split="test",
        csv_file=args.csv_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        transforms=None
    )

    # --- Model Path Logic ---
    if args.best_model == "True":
        model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"
    else:
        model_filename = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"


    path_model_risk = os.path.join(args.path_out_dir, model_filename)
    logg_filename = f"test_risk_prediction_training_id_{args.id_training}.log"
    path_logger = os.path.join(args.path_test_folder, logg_filename)

    if accelerator.is_main_process:
        os.makedirs(args.path_test_folder, exist_ok=True)
        print("Model path:", path_model_risk)
        print("Logger path:", path_logger)

    # --- Run Evaluation ---
    test_risk(
        args=args,
        test_loader=test_loader,
        path_model=path_model_risk,
        out_dir= args.path_out_dir,
        path_logger=path_logger,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
