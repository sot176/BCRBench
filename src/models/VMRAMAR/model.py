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

        if getattr(args, "freeze_image_encoder", False):
            print("Freezing image encoder parameters.")
            freeze_encoder(self.image_encoder)

        self.image_repr_dim = int(
            getattr(args, "image_repr_dim")
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

        if V < 4 and self.use_asymmetry:
            raise ValueError(
                f"Asymmetry expects 4 views ordered [R_CC, R_MLO, L_CC, L_MLO], got V={V}."
            )

        x_flat = x.reshape(B * T * V, C, H, W)

        img_feats = self.image_encoder(x_flat)

        # Encoder can return either spatial maps (B*T*V, C_feat, H_feat, W_feat)
        # or pooled vectors (B*T*V, D).
        if img_feats.dim() == 4:
            C_feat, H_feat, W_feat = img_feats.shape[1:]
            img_maps = img_feats.reshape(B, T, V, C_feat, H_feat, W_feat)

            # Pooled image vectors for VMRNN temporal branch.
            img_vecs = img_maps.mean(dim=(-1, -2))  # (B, T, V, C_feat)

            if C_feat != self.image_repr_dim:
                raise RuntimeError(
                    f"Encoder returned {C_feat} channels, but image_repr_dim={self.image_repr_dim}."
                )
        else:
            img_maps = None
            img_vecs = img_feats.reshape(B, T, V, -1)
            img_vecs = img_vecs[:, :, :, : self.image_repr_dim]

        # ----------------------------
        # Temporal branch
        # ----------------------------
        fused_feats = img_vecs.mean(dim=2)  # (B, T, D)
        fused_feats = self.temporal_projection(fused_feats)

        vmrnn_outputs = self.vmrnn(fused_feats)

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

        # ----------------------------
        # Spatial asymmetry branch
        # View order: [R_CC, R_MLO, L_CC, L_MLO]
        # ----------------------------
        r_aa = None
        asymmetry_scores = None
        asymmetry_coords = None
        asymmetry_heatmap = None

        if self.use_asymmetry:
            if img_maps is None:
                raise RuntimeError(
                    "Asymmetry requires spatial feature maps, but encoder returned vectors."
                )

            right_cc = img_maps[:, :, 0]   # (B, T, C, H, W)
            right_mlo = img_maps[:, :, 1]  # (B, T, C, H, W)
            left_cc = img_maps[:, :, 2]    # (B, T, C, H, W)
            left_mlo = img_maps[:, :, 3]   # (B, T, C, H, W)

            sad_cc = self.sad(left_cc, right_cc)
            sad_mlo = self.sad(left_mlo, right_mlo)

            asymmetry_scores = 0.5 * (
                sad_cc["asymmetry_values"] + sad_mlo["asymmetry_values"]
            )  # (B, T)

            asymmetry_coords = torch.stack(
                [sad_cc["asymmetry_coords"], sad_mlo["asymmetry_coords"]],
                dim=2,
            )  # (B, T, 2 views, 2 coords)

            asymmetry_heatmap = torch.stack(
                [sad_cc["heatmap"], sad_mlo["heatmap"]],
                dim=2,
            )  # (B, T, 2 views, H, W)

            exam_mask = batch.get(
                "exam_mask",
                torch.ones(B, T, device=images.device, dtype=torch.bool),
            )

            try:
                r_aa = self.lat(asymmetry_scores, exam_mask)
            except TypeError:
                r_aa = self.lat(asymmetry_scores)

            if isinstance(r_aa, tuple):
                r_aa = r_aa[0]

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
            "asymmetry_scores": asymmetry_scores,
            "asymmetry_coords": asymmetry_coords,
            "asymmetry_heatmap": asymmetry_heatmap,
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
