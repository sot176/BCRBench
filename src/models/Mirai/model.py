import torch
import torch.nn as nn

from . import onconet as _onconet
from .model_utils import register_onconet_alias, zero_risk_factors_for_args
from models.common_parts import BaseRiskModel, extract_mirai_backbone
from config.config import cfg

register_onconet_alias(_onconet)

from .onconet.models.factory import get_model_by_name, load_model


class Mirai(BaseRiskModel):
    """
    Mirai image-based longitudinal risk prediction model.
    """

    def __init__(self, args):
        super().__init__(args)

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])
        self._set_encoder_trainable(not getattr(args, "freeze_image_encoder", False))
        self.image_repr_dim = int(getattr(args, "image_repr_dim"))
        self.spatial_pool = nn.AdaptiveMaxPool2d((1, 1))

        # -------------------------
        # Transformer
        # -------------------------
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name("transformer", False, args)

        if hasattr(self.transformer, "args"):
            self.args.img_only_dim = getattr(
                self.transformer.args,
                "transformer_hidden_dim",
                getattr(self.transformer.args, "transfomer_hidden_dim", self.image_repr_dim),
            )

    # -------------------------
    # Helpers
    # -------------------------

    def _set_encoder_trainable(self, finetune: bool):
        for p in self.encoder.parameters():
            p.requires_grad = finetune
        self.encoder.train(mode=finetune)
        print("Finetuning image encoder" if finetune else "Freezing image encoder")

    @staticmethod
    def _to_3ch(x):
        """Convert (B,N,1,H,W) -> (B,N,3,H,W)."""
        return x.expand(-1, -1, 3, -1, -1)

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, batch):
        x = self._to_3ch(batch["images"])  # (B,N,3,H,W)
        bsz, num_imgs, channels, height, width = x.size()

        x = x.contiguous().view(bsz * num_imgs, channels, height, width)
        feat_map = self.encoder(x)

        if feat_map.dim() == 4:
            img_x = self.spatial_pool(feat_map)
        else:
            img_x = feat_map

        img_x = img_x.view(bsz, num_imgs, -1)
        img_x = img_x[:, :, :self.image_repr_dim]

        transformer_risk_factors = zero_risk_factors_for_args(
            self.args, bsz, img_x.device, img_x.dtype
        )

        logit, transformer_hidden, activ_dict = self.transformer(
            img_x, transformer_risk_factors, batch
        )

        return {
            "risk_prediction": {"logit": logit},
            "transformer_hidden": transformer_hidden,
            "activations": activ_dict,
            "image_features": img_x,
        }

    # -------------------------
    # Risk heads
    # -------------------------

    def get_risk_heads(self, outputs, batch):
        logit = outputs["risk_prediction"]["logit"]

        return {
            "logit_output": (logit, batch["target"], batch["y_mask"])
        }

    def get_primary_risk_head(self, outputs):
        return torch.sigmoid(outputs["risk_prediction"]["logit"])