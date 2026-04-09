import inspect
from models.MammoRegNet import MammoRegNet
import torch
from models.common_parts import BaseRiskModel


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
from typing import Optional
import torch.nn as nn


def get_model(model_name: str, args=None,
              path_saved_reg_model: Optional[str] = None,
              **kwargs) -> BaseRiskModel:
    """
    Build and return a risk prediction model by name.

    Args:
        model_name:            Name of the model to build.
        args:                  Runtime arguments.
        path_saved_reg_model:  Path to MammoRegNet checkpoint (required for
                               registration-based models).
    Returns:
        Instantiated BaseRiskModel.
    """

    # ── Models that require a registration network ────────────────
    REGISTRATION_MODELS = {"ImgFeatAlign", "LMV-Net"}

    # ── Lazy imports — only load what is needed ───────────────────
    def _build_mirai():
        from models.Mirai.model import Mirai
        return _build_model(Mirai, args=args, **kwargs)

    def _build_imgfeatalign():
        from models.ImgFeatAlign.model import ImgFeatAlign
        return _build_model(ImgFeatAlign,
                            mammo_reg_net=mammo_reg_net,
                            args=args, **kwargs)

    def _build_lmvnet():
        from models.LMVNet.model import LMVNet
        return _build_model(LMVNet,
                            mammo_reg_net=mammo_reg_net,
                            args=args, **kwargs)

    def _build_vmramar():
        from models.VMRAMAR.model import VMRAMaR
        return _build_model(VMRAMaR, args=args, **kwargs)

    def _build_oa_breacr():
        from models.OABreaCR.model import OA_BreaCR
        return _build_model(OA_BreaCR, args=args, **kwargs)

    # ── Registry ──────────────────────────────────────────────────
    MODEL_REGISTRY = {
        "Mirai":        _build_mirai,
        "ImgFeatAlign": _build_imgfeatalign,
        "LMV-Net":      _build_lmvnet,
        "VMRA-MaR":     _build_vmramar,
        "OA-BreaCR":    _build_oa_breacr,
    }

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    # ── Build registration network if needed ──────────────────────
    mammo_reg_net = None
    if model_name in REGISTRATION_MODELS:
        if path_saved_reg_model is None:
            raise ValueError(
                f"'{model_name}' requires a MammoRegNet checkpoint. "
                f"Pass path_saved_reg_model=<path>."
            )
        mammo_reg_net = build_mammo_reg_net(path_saved_reg_model)

    return MODEL_REGISTRY[model_name]()