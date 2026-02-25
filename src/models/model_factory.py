import inspect
from models.MammoRegNet import MammoRegNet
import torch


def _build_model(model_class, args=None, **kwargs):
    sig = inspect.signature(model_class.__init__)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    # If the model explicitly requires 'args' and we have it, pass it
    if 'args' in valid_params and args is not None:
        filtered_kwargs['args'] = args

    return model_class(**filtered_kwargs)


def build_mammo_reg_net(path_saved_reg_model):
    checkpoint = torch.load(path_saved_reg_model, map_location="cpu", weights_only=True)
    new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model_reg = MammoRegNet()
    model_reg.load_state_dict(new_checkpoint)
    model_reg.eval()

    return model_reg

def get_model(model_name: str, args=None, path_saved_reg_model=None, **kwargs):
    mammo_reg_net = None

    if model_name in {"ImgFeatAlign", "LMV-Net"}:
        if path_saved_reg_model is None:
            raise ValueError(f"{model_name} requires MammoRegNet checkpoint.")
        mammo_reg_net = build_mammo_reg_net(path_saved_reg_model)

    if model_name == "Mirai":
        from models.Mirai.model import MiraiModel
        return _build_model(MiraiModel, args=args, **kwargs)

    elif model_name == "ImgFeatAlign":
        from models.ImgFeatAlign.model import ImgFeatAlign
        return _build_model(ImgFeatAlign, mammo_reg_net=mammo_reg_net, **kwargs)

    elif model_name == "LMV-Net":
        from models.LMVNet.model import LMVNet
        return _build_model(LMVNet, mammo_reg_net=mammo_reg_net, **kwargs)

    elif model_name == "VMRA-MaR":
        from models.VMRAMAR.model import VMRAMaR
        return _build_model(VMRAMaR, args=args, **kwargs)

    elif model_name == "OA-BreaCR":
        from models.OABreaCR.model import OA_BreaCR
        return _build_model(OA_BreaCR, args=args, **kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}")