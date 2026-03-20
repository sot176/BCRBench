import torch
import torch.nn as nn
import sys

# ── Register onconet aliases BEFORE any import from factory ──────────
from . import onconet as _onconet
sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

from .onconet.models.factory import get_model_by_name, load_model, RegisterModel
from .onconet.models.hiddens_transfomer import AllImageTransformer
from models.common_parts import extract_mirai_backbone_full, extract_mirai_backbone
from config.config import cfg
from models.common_parts  import  CumulativeProbabilityLayer


@RegisterModel("mirai_full")
class Mirai(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot, args, do_wrap_model=False
            )
            # Replace RiskFactorPool with GlobalMaxPool — matches their replace_snapshot_pool logic
            if getattr(args, "replace_snapshot_pool", True):
                non_trained_encoder = get_model_by_name("custom_resnet", False, args)
                self.image_encoder._model.pool = non_trained_encoder._model.pool
                self.image_encoder._model.args = non_trained_encoder._model.args
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        if getattr(args, "freeze_image_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim

        # ── Transformer ───────────────────────────────────────────────
        args.precomputed_hidden_dim = self.image_repr_dim

        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot, args, do_wrap_model=False
            )
        else:
            self.transformer = get_model_by_name("transformer", False, args)

        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, batch):
        x = batch["images"]                              # (B, C, N, H, W)
        B, C, N, H, W = x.size()

        # Flatten views for encoder
        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        # Encode
        _, img_x, _ = self.image_encoder(x, None, batch)
        img_x = img_x.view(B, N, -1)[:, :, :self.image_repr_dim]

        # Transformer
        logit, transformer_hidden, activ_dict = self.transformer(
            img_x, None, batch
        )
        return logit, transformer_hidden, activ_dict

    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs
        return {
            "logit_output": (logit, batch["target"], batch["y_mask"])
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return logit
