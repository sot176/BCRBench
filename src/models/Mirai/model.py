import sys
import torch
import torch.nn as nn


from . import onconet as _onconet
sys.modules.setdefault("onconet", _onconet)

for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key],
        )

from .onconet.models.factory import get_model_by_name, load_model


class Mirai(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot,
                args,
                do_wrap_model=False,
            )
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        if getattr(args, "freeze_image_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self._get_img_repr_dim()

        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot,
                args,
                do_wrap_model=False,
            )
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name("transformer", False, args)

        if hasattr(self.transformer, "args"):
            self.args.img_only_dim = getattr(
                self.transformer.args,
                "transformer_hidden_dim",
                getattr(self.transformer.args, "transfomer_hidden_dim", self.image_repr_dim),
            )

    def _zero_risk_factors_for_args(
        self,
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

    def _zero_risk_factors(self, batch_size, device, dtype):
        return self._zero_risk_factors_for_args(self.args, batch_size, device, dtype)

    def _get_img_repr_dim(self):
        if hasattr(self.image_encoder, "_model"):
            return self.image_encoder._model.args.img_only_dim
        return self.image_encoder.args.img_only_dim

    def _expand_risk_factors_per_img(self, risk_factors, num_imgs):
        if risk_factors is None:
            return None

        expanded = []
        for factor in risk_factors:
            factor = factor.unsqueeze(1).expand(-1, num_imgs, -1)
            factor = factor.contiguous().view(-1, factor.size(-1))
            expanded.append(factor)
        return expanded

    def forward(self, batch):
        x = batch["images"]  # expected: (B, N, C, H, W)

        bsz, num_imgs, channels, height, width = x.size()
        risk_factors = self._zero_risk_factors(bsz, x.device, x.dtype)
        risk_factors_per_img = self._expand_risk_factors_per_img(risk_factors, num_imgs)

        x = x.contiguous().view(bsz * num_imgs, channels, height, width)

        _, img_x, _ = self.image_encoder(x, risk_factors, batch)
        img_x = img_x.view(bsz, num_imgs, -1)
        img_x = img_x[:, :, :self.image_repr_dim]

        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict

    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs
        return {
            "logit_output": (
                logit,
                batch["target"],
                batch["y_mask"],
            )
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return torch.sigmoid(logit)
