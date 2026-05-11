from __future__ import annotations

import sys

import torch
from torch import nn


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
        from onconet.utils.risk_factors import RiskFactorVectorizer

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

