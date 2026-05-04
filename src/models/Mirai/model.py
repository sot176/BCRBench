import torch
import torch.nn as nn

from . import onconet as _onconet
from .model_utils import (
    expand_risk_factors_per_img,
    freeze_encoder,
    get_img_repr_dim,
    model_args,
    register_onconet_alias,
    zero_risk_factors_for_args,
)

register_onconet_alias(_onconet)

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
            freeze_encoder(self.image_encoder)

        self.image_repr_dim = get_img_repr_dim(self.image_encoder)

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
                getattr(
                    self.transformer.args,
                    "transfomer_hidden_dim",
                    self.image_repr_dim,
                ),
            )

    def forward(self, batch):
        x = batch["images"]  # (B, N, C, H, W)
        bsz, num_imgs, channels, height, width = x.size()

        image_encoder_args = model_args(self.image_encoder)

        image_risk_factors = zero_risk_factors_for_args(
            image_encoder_args,
            bsz,
            x.device,
            x.dtype,
        )
        image_risk_factors_per_img = expand_risk_factors_per_img(
            image_risk_factors,
            num_imgs,
        )

        x = x.contiguous().view(bsz * num_imgs, channels, height, width)

        _, img_x, _ = self.image_encoder(
            x,
            image_risk_factors_per_img,
            batch,
        )

        img_x = img_x.view(bsz, num_imgs, -1)
        img_x = img_x[:, :, :self.image_repr_dim]

        transformer_risk_factors = zero_risk_factors_for_args(
            self.args,
            bsz,
            img_x.device,
            img_x.dtype,
        )

        logit, transformer_hidden, activ_dict = self.transformer(
            img_x,
            transformer_risk_factors,
            batch,
        )

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
