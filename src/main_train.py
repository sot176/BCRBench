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

def parse_block_layout(raw_block_layout):
    """
    Convert CLI strings like ['BasicBlock,2', 'BasicBlock,2', ...]
    into a proper nested list of tuples:
    [
        [('BasicBlock', 2)],
        [('BasicBlock', 2)],
        [('BasicBlock', 2)],
        [('BasicBlock', 2)]
    ]
    Each stage is a list of (block_name, num_repeats) tuples.
    """
    block_layout = []
    for stage_str in raw_block_layout:
        stage_blocks = []
        for block_spec in stage_str.split('-'):
            name, repeats = block_spec.split(',')
            stage_blocks.append((name, int(repeats)))
        block_layout.append(stage_blocks)
    return block_layout


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
        parser.add_argument('-a', '--arch', default='resnet18',
                    help='resnet18, resnet50, densenet121, densenet169, vgg16, vgg19,'
                            'convnext_tiny, convnext_small, vit_b_16, regnet_x_8gf')
        parser.add_argument('--img_size', type=int, nargs='+', default=[2048, 1664],
                    help='Height and width of image in pixels. [default: [2048,1664]]')
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
        parser.add_argument('--use_poe', action='store_true', default=True,
                            help='Enable POE functionality')
        parser.add_argument('--use_sto', action='store_true', default=True,
                            help='Enable stochastic sampling in POE')

    # -------------------
    # Mirai-specific arguments
    # -------------------
    if temp_args.model == "Mirai":
        # Snapshots / Pretrained weights
        parser.add_argument('--img_encoder_snapshot', type=str, default=None,
                            help='Filename of image feature extractor snapshot for mirai_full models')
        parser.add_argument('--transformer_snapshot', type=str, default=None,
                            help='Filename of transformer snapshot for mirai_full models')
        parser.add_argument('--state_dict_path', type=str, default=None, help='filename of model snapshot to load[default: None]')
        parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
        parser.add_argument('--calibrator_snapshot', type=str, default=None, help='filename of calibrator. Produced for a single model on development set using Platt Scaling')
        parser.add_argument('--patch_snapshot', type=str, default=None, help='filename of patch model snapshot to load. Only used for aggregator type models [default: None]')
    
        # Training / Fine-tuning options
        parser.add_argument('--freeze_image_encoder', action='store_true', default=True,
                            help='Whether to freeze image encoder during training')

        # Annotation / Auxiliary supervision
        parser.add_argument('--use_region_annotation', action='store_true', default=False,
                            help='Include cancer region annotation loss')
        
        # Model Architecture / Hyperparameters
        parser.add_argument('--transfomer_hidden_dim', type=int, default=512, help='start hidden dim for transformer')
        parser.add_argument('--use_precomputed_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model.')
        parser.add_argument('--input_dim', type=int, default=512, help='Input dim for 2stage models. [default:512]')
        parser.add_argument('--precomputed_hidden_dim', type=int, default=512,
                            help='Input dimension for transformer projection layer')
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=8, help='Num heads for transformer')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')
        parser.add_argument('--img_only_dim', type=int, default=512,
                    help='Input dimension for image-only features in the image encoder')
        # resnet-specific
        parser.add_argument('--model_name', type=str, default='mirai_full', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
        parser.add_argument('--block_layout', type=str, nargs='+', default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"], help='Layout of blocks for a ResNet model. Must be a list of length 4. Each of the 4 elements is a string of form "block_name,num_repeats-block_name,num_repeats-...". [default: resnet18 layout]')
        parser.add_argument('--block_widening_factor', type=int, default=1, help='Factor by which to widen blocks.')
        parser.add_argument('--num_groups', type=int, default=1, help='Num groups per conv in Resnet blocks.')
        parser.add_argument('--pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism')
        parser.add_argument('--deep_risk_factor_pool', action='store_true', default=False, help='make risk factor pool use several layers to fuse image and rf info')
        parser.add_argument('--replace_snapshot_pool', action='store_true', default=False, help='Use detached models')
        parser.add_argument('--pretrained_on_imagenet', action='store_true', default=False, help='Pretrain the model on imagenet. Only relevant for default models like VGG, resnet etc')
        parser.add_argument('--pretrained_imagenet_model_name', type=str, default='resnet18', help='Name of pretrained model to load for custom resnets.')
        parser.add_argument('--make_fc', action='store_true', default=False, help='Replace last linear layer with convolutional layer')
        parser.add_argument('--replace_bn_with_gn', action='store_true', default=False, help='Use group normalization instead of batch norm.')

        # risk factors
        parser.add_argument('--use_risk_factors', action='store_true', default=False, help='Whether to feed risk factors into last FC of model.') #
        parser.add_argument('--pred_risk_factors', action='store_true', default=False, help='Whether to predict value of all RF from image.') #
        parser.add_argument('--pred_risk_factors_lambda',  type=float, default=0.25,  help='lambda to weigh the risk factor prediction.')
        parser.add_argument('--use_pred_risk_factors_at_test', action='store_true', default=False, help='Whether to use predicted risk factor values at test time.') #
        parser.add_argument('--use_pred_risk_factors_if_unk', action='store_true', default=False, help='Whether to use predicted risk factor values at test time only if rf is unk.') #
        parser.add_argument('--risk_factor_keys', nargs='*', default=['density', 'binary_family_history', 'binary_biopsy_benign', 'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia', 'age', 'menarche_age', 'menopause_age', 'first_pregnancy_age', 'prior_hist', 'race', 'parous', 'menopausal_status', 'weight','height', 'ovarian_cancer', 'ovarian_cancer_age', 'ashkenazi', 'brca', 'mom_bc_cancer_history', 'm_aunt_bc_cancer_history', 'p_aunt_bc_cancer_history', 'm_grandmother_bc_cancer_history', 'p_grantmother_bc_cancer_history', 'sister_bc_cancer_history', 'mom_oc_cancer_history', 'm_aunt_oc_cancer_history', 'p_aunt_oc_cancer_history', 'm_grandmother_oc_cancer_history', 'p_grantmother_oc_cancer_history', 'sister_oc_cancer_history', 'hrt_type', 'hrt_duration', 'hrt_years_ago_stopped'], help='List of risk factors to include in risk factor vector.')
        parser.add_argument('--risk_factor_metadata_path', type=str, default='/home/administrator/Mounts/Isilon/metadata/risk_factors_jul22_2018_mammo_and_mri.json', help='Path to risk factor metadata file.')
        parser.add_argument('--pred_both_sides', action='store_true', default=False, help='Simulatenously pred both sides for multi-img model')
        parser.add_argument('--predict_birads', action='store_true', default=False, help='Wether to predict birads label for negative mammos in risk dataset objects. Note, preds, probs, and labels converted to binary (cancer vs negative) after prediction for logging purposes')
        parser.add_argument('--pred_missing_mammos', action='store_true', default=False, help='Whether to predict missing images when doing image dropout.') #
        parser.add_argument('--also_pred_given_mammos', action='store_true', default=False, help='Whether to predict given images.') #

        #survival analysis setup
        parser.add_argument('--survival_analysis_setup', action='store_true', default=True, help='Whether to modify model, eval and training for survival analysis.') #
        parser.add_argument('--make_probs_indep', action='store_true', default=False, help='Make surival model produce indepedent probablities.') #
        parser.add_argument('--mask_mechanism', default='default', help='How to mask for survival objective. options [default, indep, slice, linear].') #
        parser.add_argument('--eval_survival_on_risk', action='store_true', default=False, help='Port over survival model to risk model.') #
        parser.add_argument('--eval_risk_survival', action='store_true', default=False, help='Port over risk model to survival model.') #
        
        # device
        parser.add_argument('--is_ccds_server', action='store_true', default=False, help='Change all paths accordingly.')
        parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
        parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use in data_parallel.')
        parser.add_argument('--num_shards', type=int, default=1, help='Num GPUs to shard a single model.')
        parser.add_argument('--data_parallel', action='store_true', default=False, help='spread batch size across all available gpus. Set to false when using model parallelism. The combo of model and data parallelism may result in unexpected behavior')
        parser.add_argument('--model_parallel', action='store_true', default=False, help='spread single model across num_shards. Note must have num_shards > 1 to take effect and only support in specific models. So far supported in all models that extend Resnet-base, i.e resnet-[n], nonlocal-resnet[n], custom-resnet models')

        # Other Optional Configs
        parser.add_argument('--num_images', type=int, default=1,
                        help='In multi image setting, the number of images per single sample.')
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--max_followup', type=int, default=5,
                            help='Only used for survival analysis / cumulative probability layer')

    # -------------------
    # Parse final args
    # -------------------
    args = parser.parse_args()

    # Add a results dir for logging/output
    args.results_dir = (
        f"{args.path_out_dir}_Model_{args.model}_lr_{args.learning_rate}_wd_{args.weight_decay}"
        f"_epochs_{args.num_epochs}_bs_{args.batch_size}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}/"
    )
     # --- Mirai-specific post-processing ---
    if args.model == "Mirai":
        # Convert block layout strings into nested tuples/lists
        args.block_layout = parse_block_layout(args.block_layout)

    return args



def main():

    args = parse_arguments()
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

