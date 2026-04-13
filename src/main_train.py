import argparse
import os
import random
import logging
import time
from datetime import datetime
from typing import Optional, Any
import warnings

import torch
import yaml
import kornia.augmentation as K_A
import kornia.augmentation.container as K_C
from kornia.constants import Resample
from torch.serialization import SourceChangeWarning
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

warnings.filterwarnings("ignore", category=SourceChangeWarning)

from train import train_val
from datasets import get_dataset_and_loader


def setup_logging(path_logger: str, is_main_process: bool) -> Optional[logging.Logger]:
    """
    Configure logging for distributed training.

    Sets up file and console logging handlers only on the main process to avoid duplicate logs
    in distributed training scenarios.

    Args:
        path_logger: Path to save log file.
        is_main_process: Whether this is the main process in distributed training.

    Returns:
        Logger instance configured with file and console handlers if is_main_process=True, else None.

    Raises:
        IOError: If unable to create log directory or file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Setup handlers only on the main process to avoid duplicate logs
    if is_main_process:
        # Clear existing handlers to prevent duplicate logging
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        try:
            # File handler (writes to log file)
            file_handler = logging.FileHandler(path_logger, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

            # Console handler (prints to stdout)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(console_handler)
        except IOError as e:
            raise IOError(f"Failed to setup logging: {e}")
    return logger

def load_model_config(model_name: str, logger: logging.Logger) -> dict:
    """
    Load model-specific configuration from YAML file.

    Attempts to load configuration for a given model. Logs a warning if config is not found.

    Args:
        model_name: Name of the model (will be normalized for file lookup).
        logger: Logger instance for warnings.

    Returns:
        Dictionary containing model configuration, or empty dict if file not found.
    """
    yaml_path = f"config/models/{model_name.lower().replace('-', '_')}.yaml"
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except FileNotFoundError:
        logger.warning(
            f"YAML config for {model_name} not found at {yaml_path}. Using CLI/default values."
        )
        return {}


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments and load model-specific configuration from YAML.

    Performs two-stage parsing: first parses CLI args to determine the model, then loads
    model-specific hyperparameters from a YAML config file and adds them dynamically.

    Returns:
        Namespace object containing all parsed arguments including dynamic model-specific parameters.
        Also sets results_dir attribute with timestamped output directory path.

    Raises:
        SystemExit: If required arguments are missing (handled by argparse).
    """
    parser = argparse.ArgumentParser(
        description="Train breast cancer risk prediction models with distributed training support."
    )

    # ========== Required arguments ==========
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with data splits")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing image data")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Base output directory for results")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'Mirai', 'VMRA-MaR')")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'EMBED', 'CSAW')")

    # ========== Optional hyperparameters ==========
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--patience_lr_scheduler", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)

    # ========== Optional flags ==========
    parser.add_argument("--augmentations", action="store_true", help="Enable data augmentation")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB run ID for resuming experiment")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility")
    parser.add_argument("--id_training", type=int, default=1, help="Training run identifier")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle training data")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Pin memory in DataLoader")

    # First parse to get model name
    temp_args, _ = parser.parse_known_args()

    # Load model-specific config from YAML
    logger = logging.getLogger(__name__)
    model_config = load_model_config(temp_args.model, logger)

    # Add model-specific arguments dynamically
    for key, value in model_config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    # Final parse with all arguments
    args = parser.parse_args()

    # Construct results directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.results_dir = (
        f"{args.path_out_dir}_Model_{args.model}_lr_{args.learning_rate}_wd_{args.weight_decay}"
        f"_epochs_{args.num_epochs}_bs_{args.batch_size}_{timestamp}/"
    )

    return args

def set_reproducibility_seeds(seed: Optional[int]) -> None:
    """
    Set random seeds for reproducible training across all libraries.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU and CUDA),
    and configures cuDNN for deterministic behavior.

    Args:
        seed: Seed value. If None, skips seed setting.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def setup_augmentations() -> K_C.AugmentationSequential:
    """
    Create augmentation pipeline for training data.

    Applies a sequence of augmentations: random cropping, resizing, affine transforms,
    color jittering, and gamma correction.

    Returns:
        AugmentationSequential instance with configured augmentations.
    """
    transform = K_C.AugmentationSequential(
        K_A.RandomCrop(size=(1946, 1581), p=0.2),
        K_A.Resize((2048, 1664), resample=Resample.NEAREST.name),
        K_A.RandomAffine(
            translate=(0.0, 0.1),
            scale=(1.0, 1.1),
            degrees=0,
            shear=0,
            p=0.5,
        ),
        K_A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.0,
            p=0.5,
        ),
        K_A.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.05), p=0.5),
    )
    return transform


def main() -> None:
    """
    Main entry point for training script.

    Orchestrates the entire training pipeline:
    1. Parses CLI arguments and loads model config
    2. Initializes distributed training (Accelerator)
    3. Sets random seeds for reproducibility
    4. Creates data loaders with optional augmentations
    5. Sets up logging
    6. Launches training loop
    7. Logs training completion and timings

    Raises:
        RuntimeError: If training initialization or execution fails.
    """
    # Parse arguments
    args = parse_cli_args()

    # Initialize distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set random seeds for reproducibility
    set_reproducibility_seeds(args.seed)

    # Setup logging on main process
    logger: Optional[logging.Logger] = None
    if accelerator.is_main_process:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info(f"Training arguments: {args}")

    # Create augmentation pipeline if enabled
    train_transform: Optional[K_C.AugmentationSequential] = None
    if args.augmentations:
        train_transform = setup_augmentations()
        if accelerator.is_main_process and logger:
            logger.info(f"Data augmentation enabled")

    # Create data loaders
    try:
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
            transforms=None,  # No augmentation for validation
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create data loaders: {e}")

    # Setup output directories and paths
    model_filename = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    log_filename = f"train_risk_prediction_training_id_{args.id_training}.log"

    path_out_model = os.path.join(args.results_dir, model_filename)
    path_logger_file = os.path.join(args.results_dir, log_filename)

    # Create output directory on main process
    if accelerator.is_main_process:
        try:
            os.makedirs(args.results_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create output directory {args.results_dir}: {e}")

    # Setup file logging
    if accelerator.is_main_process:
        logger = setup_logging(path_logger_file, accelerator.is_main_process)
        logger.info(f"Output directory: {args.results_dir}")

    # Run training
    start_time = time.time()
    if accelerator.is_main_process and logger:
        logger.info("Starting training...")

    try:
        train_val(
            args,
            train_loader,
            validation_loader,
            path_logger_file,
            path_out_model,
            accelerator,
        )
    except Exception as e:
        if accelerator.is_main_process and logger:
            logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    # Log completion
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    if accelerator.is_main_process and logger:
        logger.info(f"Training completed successfully in {elapsed_minutes:.2f} minutes")
        logger.info(f"Model saved to: {path_out_model}")


if __name__ == "__main__":
    main()

