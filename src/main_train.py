import argparse
from html import parser
import os
import random
import torch
import logging
import time
import kornia.augmentation as K_A
from kornia.constants import Resample
from datetime import datetime
import kornia.augmentation.container as K_C
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import argparse
from datetime import datetime
import yaml
from argparse import Namespace

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from train import train_val
from datasets import get_dataset_and_loader

# function to log the details
def setup_logging(path_logger, is_main_process):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Setup handlers only on the main process to avoid duplicate logs
    if is_main_process:
        # Clear existing handlers to prevent duplicate logging
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler (writes to log file)
        file_handler = logging.FileHandler(path_logger, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Console handler (prints to stdout)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
    return logger

 
def parse_cli_args():
    import argparse
    import yaml
    from argparse import Namespace
    from datetime import datetime

    parser = argparse.ArgumentParser()

    # ---------------- General CLI args ----------------
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--patience_lr_scheduler", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--finetune_all", action="store_true")
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--wandb_id", type=str)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--id_training", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--pin_memory", type=bool, default=True)

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
    args = parse_cli_args()

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Define datasets and dataloader
    if args.augmentations:  ### For newest and oldest mammograms
        train_transform = K_C.AugmentationSequential(
            K_A.RandomCrop(size=(1946, 1581), p=0.2),
            K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
            K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.1), degrees=0, shear=0, p=0.5),
            K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
            K_A.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.05), p=0.5),
        )
        if accelerator.is_main_process:
            print("Train augmentations :", train_transform)
    else:
        train_transform = None


    train_loader = get_dataset_and_loader(
        dataset_name=args.dataset,
        model_name=args.model,
        split="train",
        csv_file=args.csv_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        transforms=train_transform,
    )
    validation_loader = get_dataset_and_loader(
        dataset_name=args.dataset,
        model_name=args.model,
        split="val",
        csv_file=args.csv_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        transforms=train_transform,
    )

    # Setup the path to save the model and the logger.
    model_path = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    logg_path = f"train_risk_prediction_training_id_{args.id_training}.log"

    path_out_model = os.path.join(args.results_dir, model_path)
    path_logger = os.path.join(args.results_dir, logg_path)

    # Ensure the directory exists on the main process
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)

    # call the logging
    logger = setup_logging(path_logger, accelerator.is_main_process)

    start_time = time.time()

    if accelerator.is_main_process:
        logger.info("Training started...")
    train_val(args,
                      train_loader,
                      validation_loader,
                      path_logger,
                      path_out_model,
                      accelerator  # Pass the accelerator object
                      )

    end_time = time.time()
    if accelerator.is_main_process:
        logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
        logger.info(f"Saving model to: {path_out_model}")


if __name__ == '__main__':
    main()

