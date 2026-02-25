import argparse
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

from accelerate import Accelerator
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # -------------------
    # Common arguments for all models
    # -------------------
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--augmentations", type=str, required=True)
    parser.add_argument("--use_scheduler", type=str, required=True)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--finetune_all", action="store_true")
    parser.add_argument("--patience_lr_scheduler", default=5, type=int)
    parser.add_argument("--patience", default=15, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--schuffle", default=True, type=bool)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (mirai, ImgFeatAlign, VMRA-MaR, OA-BreaCR, LMV-Net, etc.)")

    # -------------------
    # Parse first to check the model
    # -------------------
    temp_args, _ = parser.parse_known_args()  # Only parse known args for now

    # -------------------
    # OA-BreaCR-specific arguments
    # -------------------
    if temp_args.model == "OA-BreaCR":
        parser.add_argument('--num_output_neurons', type=int, default=6,
                            help='Number of output neurons, should be max_followup+1')
        parser.add_argument('--start_label', type=int, default=0,
                            help='Start label for ordinal learning')
        parser.add_argument('--max-t', type=int, default=50,
                            help='Number of samples during stochastic sampling')
        parser.add_argument('--no-sto', action='store_true',
                            help='Disable stochastic sampling')
        parser.add_argument('--distance', type=str, default='JDistance',
                            help='Distance metric between two Gaussian distributions')
        parser.add_argument('--alpha-coeff', type=float, default=1e-5)
        parser.add_argument('--beta-coeff', type=float, default=1e-4)
        parser.add_argument('--margin', type=float, default=2)
        parser.add_argument('--use_poe', action='store_true',
                            help='Enable POE functionality')

    # -------------------
    # Parse final args including OA-BreaCR args if any
    # -------------------
    args = parser.parse_args()

    # Add a results dir for logging/output
    args.results_dir = (
        f"{args.path_out_dir}_Model_{args.model}_lr_{args.learning_rate}_wd_{args.weight_decay}"
        f"_epochs_{args.num_epochs}_bs_{args.batch_size}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}/"
    )

    return args



def main():

    args = parse_arguments()
    accelerator = Accelerator()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Define datasets and dataloader
    if args.augmentations == "True":  ### For newest and oldest mammograms
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
        shuffle=args.schuffle,
        pin_memory=args.pin_memory,
        transforms=train_transform
    )
    validation_loader = get_dataset_and_loader(
        dataset_name=args.dataset,
        model_name=args.model,
        split="val",
        csv_file=args.csv_file,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.schuffle,
        pin_memory=args.pin_memory,
        transforms=train_transform
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

