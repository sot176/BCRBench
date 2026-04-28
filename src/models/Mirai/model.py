

import torch
import torch.nn as nn
import sys


# ----------------------------------------------------
# Register OncoNet aliases
# ----------------------------------------------------
from . import onconet as _onconet
sys.modules.setdefault("onconet", _onconet)

for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
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

        args.img_only_dim = self.transformer.args.transformer_hidden_dim
    
    def _get_img_repr_dim(self):
        if hasattr(self.image_encoder, "_model"):
            return self.image_encoder._model.args.img_only_dim
        return self.image_encoder.args.img_only_dim

    def forward(self, batch):
        """
        x: (B, C, N, H, W)
        """
        x = batch["images"]        # (B,4,C,H,W)

        bsz, channels, num_imgs, height, width = x.size()

        x = x.transpose(1, 2).contiguous().view(bsz * num_imgs, channels, height, width)

        _, img_x, _ = self.image_encoder(x, None, batch)
        img_x = img_x.view(bsz, num_imgs, -1)
        img_x = img_x[:, :, :self.image_repr_dim]

        logit, transformer_hidden, activ_dict = self.transformer(img_x, None, batch)
        return logit, transformer_hidden, activ_dict


    # =================================================
    # Heads
    # =================================================
    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs

        return {
            "logit_output": (
                logit,
                batch["target"],      # (B,5)
                batch["y_mask"]
            )
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return torch.sigmoid(logit)