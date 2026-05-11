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
            self.image_encoder.eval()

        self.image_repr_dim = int(getattr(args, "image_repr_dim", 512))
        self.spatial_pool = nn.AdaptiveMaxPool2d((1, 1))

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

        if self.image_repr_dim != self.vmrnn_embed_dim:
            self.temporal_projection = nn.Sequential(
                nn.LayerNorm(self.image_repr_dim),
                nn.Linear(self.image_repr_dim, self.vmrnn_embed_dim),
                nn.Dropout(float(getattr(args, "vmrnn_projection_dropout", 0.1))),
            )
        else:
            self.temporal_projection = nn.Identity()

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
            print("Using asymmetry modules")

            self.sad = SpatialAsymmetryDetector(
                latent_h=int(
                    asymmetry_params.get("latent_h", getattr(args, "latent_h", 5))
                ),
                latent_w=int(
                    asymmetry_params.get("latent_w", getattr(args, "latent_w", 5))
                ),
                flexible=str2bool(
                    asymmetry_params.get(
                        "flexible_asymmetry",
                        getattr(args, "flexible_asymmetry", False),
                    )
                ),
                embedding_channel=self.image_repr_dim,
                initial_asym_mean=float(
                    asymmetry_params.get(
                        "initial_asym_mean",
                        getattr(args, "initial_asym_mean", 8_000_000.0),
                    )
                ),
                initial_asym_std=float(
                    asymmetry_params.get(
                        "initial_asym_std",
                        getattr(args, "initial_asym_std", 1_520_381.0),
                    )
                ),
            )

            self.lat = LongitudinalAsymmetryTracker(
                threshold_ratio=float(
                    asymmetry_params.get(
                        "threshold_ratio",
                        getattr(args, "threshold_ratio", 0.4),
                    )
                ),
                persistent_weight=float(
                    asymmetry_params.get(
                        "persistent_weight",
                        getattr(args, "persistent_weight", 1.0),
                    )
                ),
            )

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

        # Encoder can return either:
        # spatial maps: (B*T*V, C_feat, H_feat, W_feat)
        # vectors:      (B*T*V, D)
        if img_feats.dim() == 4:
            C_feat, H_feat, W_feat = img_feats.shape[1:]

            if C_feat != self.image_repr_dim:
                raise RuntimeError(
                    f"Encoder returned {C_feat} channels, but image_repr_dim={self.image_repr_dim}."
                )

            img_maps = img_feats.reshape(B, T, V, C_feat, H_feat, W_feat)

            img_vecs = self.spatial_pool(img_feats).flatten(1)
            img_vecs = img_vecs.reshape(B, T, V, C_feat)
        else:
            img_maps = None
            img_vecs = img_feats.reshape(B, T, V, -1)
            img_vecs = img_vecs[:, :, :, : self.image_repr_dim]

        # ----------------------------
        # Temporal branch
        # ----------------------------
        fused_feats = img_vecs.mean(dim=2)  # (B, T, D)
        fused_feats = self.temporal_projection(fused_feats)

        reconstructed_output, states_down, states_up, latent_tokens = self.vmrnn(
            fused_feats
        )

        if latent_tokens.dim() == 3:
            temporal_feature = latent_tokens.mean(dim=1)
        elif latent_tokens.dim() == 4:
            temporal_feature = latent_tokens.mean(dim=(1, 2))
        else:
            temporal_feature = latent_tokens

        if temporal_feature.dim() != 2:
            temporal_feature = temporal_feature.reshape(B, -1)

        if temporal_feature.shape[1] != self.vmrnn_embed_dim:
            raise RuntimeError(
                f"Temporal feature has dim {temporal_feature.shape[1]}, "
                f"but risk head expects vmrnn_embed_dim={self.vmrnn_embed_dim}."
            )

        features = [temporal_feature]

        # ----------------------------
        # Asymmetry-aware risk factor branch
        # ----------------------------
        r_aa = None
        asymmetry_scores = None
        asymmetry_coords = None
        coord_valid = None

        if self.use_asymmetry:
            if img_maps is None:
                raise RuntimeError(
                    "Asymmetry requires spatial feature maps, but encoder returned vectors."
                )

            view_mask = batch.get(
                "view_mask",
                torch.ones(B, T, V, device=images.device, dtype=torch.bool),
            )

            exam_mask = batch.get(
                "exam_mask",
                torch.ones(B, T, device=images.device, dtype=torch.bool),
            )

            asymmetry_scores, asymmetry_coords, coord_valid = self.sad(
                img_maps,
                view_mask,
            )

            r_aa = self.lat(
                asymmetry_scores,
                asymmetry_coords,
                coord_valid,
                exam_mask,
                window_size=max(H_feat, W_feat),
            )

            r_aa = r_aa.unsqueeze(-1)  # (B, 1)
            features.append(r_aa)

        combined_feats = torch.cat(features, dim=1)

        logits = self.ahl(combined_feats)

        # Keep this only if CumulativeProbabilityLayer returns logits.
        # If it already returns cumulative probabilities, remove sigmoid.
        probs = torch.sigmoid(logits)

        return {
            "logit": logits,
            "probs": probs,
            "temporal_feature": temporal_feature,
            "temporal_output": reconstructed_output,
            "latent_tokens": latent_tokens,
            "states_down": states_down,
            "states_up": states_up,
            "r_aa": r_aa,
            "asymmetry_scores": asymmetry_scores,
            "asymmetry_coords": asymmetry_coords,
            "coord_valid": coord_valid,
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
