import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from evaluate import test_risk
from datasets import get_dataset_and_loader


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
    parser.add_argument("--finetune_all", action="store_true")

    # Dataloader args
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--shuffle", default=False, type=bool)  
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (mirai, ImgFeatAlign, VMRA-MaR, OA-BreaCR, LMV-Net, etc.)")
    
    temp_args, _ = parser.parse_known_args()  # Only parse known args for now

    if temp_args.model == "OA-BreaCR":
        parser.add_argument('-a', '--arch', default='resnet18',
                    help='resnet18, resnet50')
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
        parser.add_argument('--distance', type=str, default='Bhattacharyya',
                            help='Distance metric between two Gaussian distributions')
        parser.add_argument('--alpha-coeff', type=float, default=1e-5)
        parser.add_argument('--beta-coeff', type=float, default=1e-4)
        parser.add_argument('--margin', type=float, default=2)
        parser.add_argument('--use_poe', action='store_true', help='Enable POE functionality')
        parser.add_argument('--no_poe', action='store_false', dest='use_poe')
        parser.add_argument('--use_sto', action='store_true', help='Enable stochastic sampling in POE')
        parser.add_argument('--no_sto', action='store_false', dest='use_sto')
        

    if temp_args.model == "Mirai" or temp_args.model == "VMRA-MaR":
        # Snapshots / Pretrained weights
        parser.add_argument('--img_encoder_snapshot', type=str, default=None,
                            help='Filename of image feature extractor snapshot for mirai_full models')
        parser.add_argument('--transformer_snapshot', type=str, default=None,
                            help='Filename of transformer snapshot for mirai_full models')
        parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')

        # Training / Fine-tuning options
        parser.add_argument('--freeze_image_encoder',   action='store_true',
                            help='Whether to freeze image encoder during training')

        # Transformer architecture

        parser.add_argument('--transfomer_hidden_dim', type=int, default=512, help='start hidden dim for transformer')
        parser.add_argument('--use_precomputed_hiddens', action='store_true', default=False, help='Whether to only use hiddens from a pretrained model.')
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=8, help='Num heads for transformer')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')
        parser.add_argument('--multi_image',type=bool, default=True, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')  

        # resnet-specific
        parser.add_argument('--model_name', type=str, default='mirai_full', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
        parser.add_argument('--block_layout', type=str, nargs='+', default=["BasicBlock,2", "BasicBlock,2", "BasicBlock,2", "BasicBlock,2"], help='Layout of blocks for a ResNet model. Must be a list of length 4. Each of the 4 elements is a string of form "block_name,num_repeats-block_name,num_repeats-...". [default: resnet18 layout]')
        parser.add_argument('--block_widening_factor', type=int, default=1, help='Factor by which to widen blocks.')
        parser.add_argument('--num_groups', type=int, default=1, help='Num groups per conv in Resnet blocks.')
        parser.add_argument('--pool_name', type=str, default='GlobalMaxPool', help='Pooling mechanism')
        parser.add_argument('--deep_risk_factor_pool', action='store_true',  help='make risk factor pool use several layers to fuse image and rf info')
        parser.add_argument('--replace_snapshot_pool', type=bool, default=True,  help='Use detached models')
        parser.add_argument('--pretrained_on_imagenet', action='store_true',  help='Pretrain the model on imagenet. Only relevant for default models like VGG, resnet etc')
        parser.add_argument('--pretrained_imagenet_model_name', type=str, default='resnet18', help='Name of pretrained model to load for custom resnets.')
        parser.add_argument('--make_fc', action='store_true',  help='Replace last linear layer with convolutional layer')
        parser.add_argument('--replace_bn_with_gn', action='store_true', help='Use group normalization instead of batch norm.')

        # Risk factors
        parser.add_argument('--use_risk_factors',type=bool, default=False, help='Whether to feed risk factors into last FC of model.') 
        parser.add_argument('--pred_risk_factors', type=bool,default=False, help='Whether to predict value of all RF from image.') 
        parser.add_argument('--pred_both_sides', type=bool,default=False, help='Simulatenously pred both sides for multi-img model')
        parser.add_argument('--predict_birads',  type=bool,default=False, help='Wether to predict birads label for negative mammos in risk dataset objects. Note, preds, probs, and labels converted to binary (cancer vs negative) after prediction for logging purposes')
        parser.add_argument('--pred_missing_mammos',type=bool, default=False, help='Whether to predict missing images when doing image dropout.') 
        parser.add_argument('--also_pred_given_mammos',type=bool, default=False, help='Whether to predict given images.') 
        
        # regularization
        parser.add_argument('--use_region_annotation', action='store_true', default=False, help='Wether to add a loss factoring in the collected cancer region annotations .')

        #survival analysis setup
        parser.add_argument('--survival_analysis_setup', action='store_true',  help='Whether to modify model, eval and training for survival analysis.') 
        parser.add_argument('--max_followup', type=int, default=5, help='Max followup to predict over')
        parser.add_argument('--state_dict_path', type=str, default=None,
                        help='filename of model snapshot to load[default: None]')
        # Other Optional Configs
        parser.add_argument('--num_images', type=int, default=4,
                        help='In multi image setting, the number of images per single sample.')
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
        parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use in data_parallel.')
        parser.add_argument('--num_shards', type=int, default=1, help='Num GPUs to shard a single model.')
        parser.add_argument('--data_parallel', action='store_true', default=False,
                            help='spread batch size across all available gpus. Set to false when using model parallelism. The combo of model and data parallelism may result in unexpected behavior')
        parser.add_argument('--model_parallel', action='store_true', default=False,
                            help='spread single model across num_shards. Note must have num_shards > 1 to take effect and only support in specific models. So far supported in all models that extend Resnet-base, i.e resnet-[n], nonlocal-resnet[n], custom-resnet models')
        parser.add_argument('--wrap_model', action='store_true', default=False,
                            help='Whether to strip last layer of model, and add layers to fit to new task.')
        # VMRNN architecture parameters
        parser.add_argument('--depths_downsample', nargs='+', type=int,
                            default=[2, 2, 6, 2],
                            help='Depths for downsample blocks')
        parser.add_argument('--depths_upsample', nargs='+', type=int,
                            default=[2, 2, 6, 2],
                            help='Depths for upsample blocks')
        parser.add_argument('--embed_dim', type=int, default=512,
                            help='Embedding dimension')

        # Asymmetry module parameters
        parser.add_argument('--use_asymmetry', action='store_true',
                            help='Enable asymmetry module')
        parser.add_argument('--latent_h', type=int, default=52)
        parser.add_argument('--latent_w', type=int, default=64)
        parser.add_argument('--use_sad_bias', action='store_true')
        parser.add_argument('--use_lat_bn', action='store_true')   
        parser.add_argument('--use_sad_bn', action='store_true')
        parser.add_argument('--lat_dropout', type=float, default=0.0)
        parser.add_argument('--initial_asym_mean', type=float, default=2000)
        parser.add_argument('--initial_asym_std', type=float, default=300)
        parser.add_argument("--asym_dim", type=int, default=0, help="Dimension of asymmetry features ")
    
    args = parser.parse_args()

    # --- Mirai-specific post-processing ---
    if args.model == "Mirai" or  args.model == "VMRA-MaR":
        # Convert block layout strings into nested tuples/lists
        args.block_layout = parse_block_layout(args.block_layout)

    return args



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
