from models.model_factory import get_model
from config.config import cfg
import torch


def load_model(args, path_model):
    """Load model and checkpoint."""
    path_saved_reg_model = (
        cfg["paths"]["csaw_path_saved_reg_model"]
        if args.dataset == "CSAW"
        else cfg["paths"]["embed_path_saved_reg_model"]
    )

    model = get_model(
        args.model,
        args=args,
        path_saved_reg_model=path_saved_reg_model,
        max_followup=5,
        finetune_all=args.finetune_all,
    )

    checkpoint = torch.load(path_model, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model.eval()


def gather_tensor(accelerator, tensor):
    """Helper for gathering tensors across processes."""
    return accelerator.gather(tensor).cpu()
