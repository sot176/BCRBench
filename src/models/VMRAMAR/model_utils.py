from __future__ import annotations

import torch
from torch import nn
import sys



MAX_FOLLOWUP = 5
FORMAL_VIEW_SEQUENCE = (
    ("RCC", 0, 0),
    ("RMLO", 1, 0),
    ("LCC", 0, 1),
    ("LMLO", 1, 1),
)


def register_onconet_alias(onconet_module) -> None:
    sys.modules.setdefault("onconet", onconet_module)

    for key in list(sys.modules.keys()):
        if key.startswith("models.Mirai.onconet"):
            sys.modules.setdefault(
                key.replace("models.Mirai.onconet", "onconet"),
                sys.modules[key],
            )


def freeze_encoder(encoder: nn.Module) -> None:
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()


def model_args(model):
    if hasattr(model, "_model"):
        return model._model.args
    return model.args


def get_img_repr_dim(image_encoder) -> int:
    if hasattr(image_encoder, "_model"):
        return image_encoder._model.args.img_only_dim
    return image_encoder.args.img_only_dim


def resolve_module(root: nn.Module, name: str) -> nn.Module:
    modules = dict(root.named_modules())
    if name not in modules:
        matches = [key for key in modules if name in key]
        raise ValueError(
            f"Feature map layer {name!r} not found. "
            f"Close matches: {matches[:20]}"
        )
    return modules[name]


def setup_feature_map_hook(model, args) -> None:
    model.feature_map_layer_name = getattr(args, "feature_map_layer", "layer4_1")
    model._captured_feature_map = None

    encoder_model = (
        model.image_encoder._model
        if hasattr(model.image_encoder, "_model")
        else model.image_encoder
    )

    model.feature_map_layer = resolve_module(
        encoder_model,
        model.feature_map_layer_name,
    )

    def capture_feature_map(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        model._captured_feature_map = output

    model.feature_map_hook = model.feature_map_layer.register_forward_hook(
        capture_feature_map
    )


def zero_risk_factors_for_args(
    args,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if not bool(getattr(args, "use_risk_factors", False)):
        return None

    key_to_dim = getattr(args, "risk_factor_key_to_num_class", None)
    risk_factor_keys = list(getattr(args, "risk_factor_keys", []) or [])

    if (not key_to_dim) and risk_factor_keys:
        from models.Mirai.onconet.utils.risk_factors import RiskFactorVectorizer

        RiskFactorVectorizer(args)
        key_to_dim = args.risk_factor_key_to_num_class

    if key_to_dim and risk_factor_keys:
        return [
            torch.zeros(batch_size, int(key_to_dim[key]), device=device, dtype=dtype)
            for key in risk_factor_keys
        ]

    rf_dim = int(getattr(args, "rf_dim", 0) or 0)
    if rf_dim > 0:
        return [torch.zeros(batch_size, rf_dim, device=device, dtype=dtype)]

    return None


def expand_risk_factors_per_img(risk_factors, num_imgs: int):
    if risk_factors is None:
        return None

    expanded = []
    for factor in risk_factors:
        factor = factor.unsqueeze(1).expand(-1, num_imgs, -1)
        factor = factor.contiguous().view(-1, factor.size(-1))
        expanded.append(factor)

    return expanded


def make_transformer_batch(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    view_seq = torch.tensor(
        [view for _, view, _ in FORMAL_VIEW_SEQUENCE],
        device=device,
        dtype=torch.long,
    )
    side_seq = torch.tensor(
        [side for _, _, side in FORMAL_VIEW_SEQUENCE],
        device=device,
        dtype=torch.long,
    )

    return {
        "time_seq": torch.zeros(batch_size, len(FORMAL_VIEW_SEQUENCE), device=device, dtype=torch.long),
        "view_seq": view_seq.unsqueeze(0).expand(batch_size, -1),
        "side_seq": side_seq.unsqueeze(0).expand(batch_size, -1),
    }


def compute_asymmetry_feature(sad, lat, feat_maps, view_mask, exam_mask):
    asymmetry_scores, coords, coord_valid = sad(feat_maps, view_mask)

    window_size = max(int(sad.latent_h), int(sad.latent_w))

    r_aa = lat(
        asymmetry_scores,
        coords,
        coord_valid,
        exam_mask,
        window_size=window_size,
    )

    if r_aa.dim() == 1:
        r_aa = r_aa.unsqueeze(-1)
    elif r_aa.dim() != 2:
        raise ValueError(f"Unexpected asymmetry feature shape: {r_aa.shape}")

    return r_aa, asymmetry_scores, coords, coord_valid
