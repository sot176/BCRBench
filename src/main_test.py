import argparse
from email import parser
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import yaml

from evaluate import test_risk
from datasets import get_dataset_and_loader


def parse_test_args():

    parser = argparse.ArgumentParser()

    # ---------------- General CLI args ----------------
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
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name (mirai, ImgFeatAlign, VMRA-MaR, OA-BreaCR, LMV-Net, etc.)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (EMBED, CSAW-CC, etc.)")

    # ---------------- Temporary parse ----------------
    temp_args, _ = parser.parse_known_args()

    # ---------------- Model-specific YAML ----------------
    try:
        yaml_path = f"configs/model/{temp_args.model.lower()}.yaml"
        with open(yaml_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Dynamically add YAML args to the parser
        for k, v in model_config.items():
            # Determine argument type from the YAML value
            arg_type = type(v) if not isinstance(v, bool) else None
            if isinstance(v, bool):
                # For booleans, create store_true/store_false flags
                if v:
                    parser.add_argument(f"--{k}", action="store_true")
                else:
                    parser.add_argument(f"--{k}", action="store_false")
            else:
                parser.add_argument(f"--{k}", type=arg_type, default=v)
    except FileNotFoundError:
        print(f"[WARNING] YAML config for {temp_args.model} not found. Using CLI/default values.")

    # ---------------- Final parse ----------------
    args = parser.parse_args()

    # ---------------- Results directory ----------------
    args.results_dir = (
        f"{args.path_out_dir}_Model_{args.model}_lr_{args.learning_rate}_wd_{args.weight_decay}"
        f"_epochs_{args.num_epochs}_bs_{args.batch_size}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}/"
    )

    return args


def main():
    args = parse_test_args()
    print("Arguments", args)
    
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
