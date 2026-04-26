import torch
import torch.nn as nn
import sys
from models.common_parts import BaseRiskModel


# ── Register onconet aliases before any import from factory ──────────
from . import onconet as _onconet
sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

from .onconet.models.factory import get_model_by_name, load_model


class Mirai(BaseRiskModel):
    """
    Full Mirai model combining image encoder + transformer for longitudinal risk prediction.
    """
    def __init__(self, args):
        super().__init__(args)

        # -------------------------
        # Image Encoder
        # -------------------------
        self.image_encoder = self._init_image_encoder(self.args)

        # Freeze encoder if requested
        if getattr(self.args, "freeze_image_encoder", True):
            self._freeze_encoder(self.image_encoder)

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim

        # -------------------------
        # Transformer
        # -------------------------
        self.args.precomputed_hidden_dim = self.image_repr_dim
        self.transformer = self._init_transformer(self.args)

        # Update transformer output dim
        self.args.img_only_dim = self.transformer.args.transformer_hidden_dim

    # -------------------------
    # Helper methods
    # -------------------------
    def _init_image_encoder(self, args):
        """Initialize the image encoder, optionally loading a snapshot."""
        if getattr(args, "img_encoder_snapshot", None):
            encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            encoder = get_model_by_name("custom_resnet", False, args)
        return encoder

    @staticmethod
    def _freeze_encoder(encoder):
        """Freeze all parameters of the encoder."""
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("[INFO] Image encoder frozen.")

    def _init_transformer(self, args):
        """Initialize the transformer, optionally loading a snapshot."""
        if getattr(args, "transformer_snapshot", None):
            transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            transformer = get_model_by_name("transformer",False, args)
        return transformer

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, batch):
        images = batch["images"]   # (B, C, N, H, W)
        risk_factors = batch.get("risk_factors", None)
        B, C, N, H, W = images.size()
        x = images.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        
        return logit, transformer_hidden, activ_dict


    # -------------------------
    # Risk head helpers
    # -------------------------
    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs
        return {
            "logit_output": (logit, batch["target"], batch["y_mask"])
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return torch.sigmoid(logit)