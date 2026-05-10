import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from config.config import cfg
from models.common_parts import extract_mirai_backbone
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from .vmrnn import VMRNN

from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.factory import get_model_by_name, load_model
from .model_utils import (
    freeze_encoder,
    get_img_repr_dim,
    register_onconet_alias,zero_risk_factors_for_args, expand_risk_factors_per_img, model_args
)
 
register_onconet_alias(_onconet)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def parse_params(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def parse_int_list(value, default):
    if value is None:
        value = default

    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") or value.startswith("("):
            value = value.strip("[]()")
        value = [v.strip() for v in value.split(",") if v.strip()]

    if isinstance(value, int):
        value = [value]

    if isinstance(value, tuple):
        value = list(value)

    return [int(v) for v in value]


class VMRAMaR(BaseRiskModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        vmrnn_params = parse_params(getattr(args, "vmrnn_params", None))
        asymmetry_params = parse_params(getattr(args, "asymmetry_params", None))

        # ----------------------------
        # Image encoder
        # ----------------------------
        self.image_encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])


        if str2bool(getattr(args, "freeze_image_encoder", False)):
            freeze_encoder(self.image_encoder)

        self.image_repr_dim = int(
            getattr(args, "image_repr_dim", get_img_repr_dim(self.image_encoder))
        )

        # ----------------------------
        # VMRNN
        # ----------------------------
        self.vmrnn_embed_dim = int(
            vmrnn_params.get("embed_dim", getattr(args, "embed_dim", 512))
        )

        depths_downsample = parse_int_list(
            vmrnn_params.get(
                "depths_downsample",
                getattr(args, "depths_downsample", None),
            ),
            [1, 2],
        )

        depths_upsample = parse_int_list(
            vmrnn_params.get(
                "depths_upsample",
                getattr(args, "depths_upsample", None),
            ),
            [2, 1],
        )

        feature_resolution = parse_int_list(
            vmrnn_params.get(
                "feature_resolution",
                getattr(args, "feature_resolution", None),
            ),
            [1, 2],
        )

        self.vmrnn = VMRNN(
            embed_dim=self.vmrnn_embed_dim,
            depths_downsample=depths_downsample,
            depths_upsample=depths_upsample,
            feature_resolution=feature_resolution,
        )

        self.temporal_projection = nn.Identity()
        if self.image_repr_dim != self.vmrnn_embed_dim:
            self.temporal_projection = nn.Sequential(
                nn.LayerNorm(self.image_repr_dim),
                nn.Linear(self.image_repr_dim, self.vmrnn_embed_dim),
                nn.Dropout(float(getattr(args, "vmrnn_projection_dropout", 0.1))),
            )

        # ----------------------------
        # Asymmetry modules
        # ----------------------------
        self.use_asymmetry = str2bool(
            asymmetry_params.get(
                "use_asymmetry",
                getattr(args, "use_asymmetry", False),
            )
        )

        self.sad = None
        self.lat = None
        self.asym_dim = 0

        if self.use_asymmetry:
            sad_args = SimpleNamespace(
                latent_h=int(
                    asymmetry_params.get(
                        "latent_h",
                        getattr(args, "latent_h", 5),
                    )
                ),
                latent_w=int(
                    asymmetry_params.get(
                        "latent_w",
                        getattr(args, "latent_w", 5),
                    )
                ),
                lat_dropout=float(
                    asymmetry_params.get(
                        "lat_dropout",
                        getattr(args, "lat_dropout", 0.1),
                    )
                ),
                use_sad_bias=str2bool(
                    asymmetry_params.get(
                        "use_sad_bias",
                        getattr(args, "use_sad_bias", True),
                    )
                ),
                use_sad_bn=str2bool(
                    asymmetry_params.get(
                        "use_sad_bn",
                        getattr(args, "use_sad_bn", True),
                    )
                ),
            )

            self.sad = SpatialAsymmetryDetector(sad_args)
            self.lat = LongitudinalAsymmetryTracker(asymmetry_params)

            self.asym_dim = int(
                asymmetry_params.get(
                    "asym_dim",
                    getattr(args, "asym_dim", 1),
                )
            )
            if self.asym_dim <= 0:
                self.asym_dim = 1

        # ----------------------------
        # Risk head
        # ----------------------------
        final_dim = self.vmrnn_embed_dim + self.asym_dim

        self.ahl = CumulativeProbabilityLayer(
            final_dim,
            max_followup=int(getattr(args, "max_followup", 5)),
        )

    def forward(self, batch):
        images = batch["images"]

        if images.dim() != 6:
            raise ValueError(f"Expected images with 6 dims, got shape {images.shape}")

        # Accept both:
        # (B, T, V, C, H, W)
        # (B, T, C, V, H, W)
        if images.shape[3] in {1, 3}:
            B, T, V, C, H, W = images.shape
            x = images
        else:
            B, T, C, V, H, W = images.shape
            x = images.permute(0, 1, 3, 2, 4, 5).contiguous()

        x_flat = x.view(B * T * V, C, H, W)

        image_encoder_args = model_args(self.image_encoder)

        image_risk_factors = zero_risk_factors_for_args(
            image_encoder_args,
            B * T,
            x_flat.device,
            x_flat.dtype,
        )

        risk_factors_per_img = expand_risk_factors_per_img(
            image_risk_factors,
            V,
        )

        _, img_feats, _ = self.image_encoder(
            x_flat,
            risk_factors_per_img,
            batch,
        )

        img_feats = img_feats.view(B, T, V, -1)
        img_feats = img_feats[:, :, :, : self.image_repr_dim]

        fused_feats = img_feats.mean(dim=2)
        fused_feats = self.temporal_projection(fused_feats)

        vmrnn_outputs = self.vmrnn(fused_feats, None, batch)

        if isinstance(vmrnn_outputs, tuple):
            temporal_output = vmrnn_outputs[0]
            hidden_states = vmrnn_outputs[1] if len(vmrnn_outputs) > 1 else None
        else:
            temporal_output = vmrnn_outputs
            hidden_states = None

        if temporal_output.dim() == 3:
            temporal_feature = temporal_output.mean(dim=1)
        else:
            temporal_feature = temporal_output

        features = [temporal_feature]

        r_aa = None
        if self.use_asymmetry:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]

            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            r_aa = self.lat(asym_feats)

            if r_aa.dim() == 1:
                r_aa = r_aa.unsqueeze(-1)

            features.append(r_aa)

        combined_feats = torch.cat(features, dim=1)

        logits = self.ahl(combined_feats)
        probs = torch.sigmoid(logits)

        return {
            "logit": logits,
            "probs": probs,
            "temporal_feature": temporal_feature,
            "temporal_output": temporal_output,
            "hidden_states": hidden_states,
            "r_aa": r_aa,
        }

    def get_risk_heads(self, outputs, batch):
        return {
            "logit_output": (
                outputs["logit"],
                batch["target"],
                batch["y_mask"],
            )
        }

    def get_primary_risk_head(self, outputs):
        return outputs["probs"]
