import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from evaluate import test_risk
from datasets import get_dataset_and_loader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--path_test_folder", type=str, required=True)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--dataset", type=str)

    # Model architecture flags
    parser.add_argument("--early_stop", type=str, default="False")
    parser.add_argument("--best_model", type=str, default="False")
    parser.add_argument("--use_checkppoint", type=str, default="False")

    # Dataloader args
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--shuffle", default=False, type=bool)  # Corrected spelling
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (mirai, ImgFeatAlign, VMRA-MaR, OA-BreaCR, LMV-Net, etc.)")
    return parser.parse_args()



def main():
    args = parse_arguments()
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
        shuffle=args.schuffle,
        pin_memory=args.pin_memory,
        transforms=None
    )

    # --- Model Path Logic ---
    model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"

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
