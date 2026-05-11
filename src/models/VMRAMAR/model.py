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
            sad_args = SimpleNamespace(
                latent_h=int(
                    asymmetry_params.get("latent_h", getattr(args, "latent_h", 5))
                ),
                latent_w=int(
                    asymmetry_params.get("latent_w", getattr(args, "latent_w", 5))
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
                asymmetry_params.get("asym_dim", getattr(args, "asym_dim", 1))
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

    def train(self, mode=True):
        super().train(mode)

        if getattr(self.args, "freeze_image_encoder", False):
            self.image_encoder.eval()

        return self

    def _run_sad_pair(self, left, right, B, T, C, H, W):
        left_flat = left.reshape(B * T, C, H, W)
        right_flat = right.reshape(B * T, C, H, W)

        out = self.sad(left_flat, right_flat)

        if "asymmetry_values" in out:
            values = out["asymmetry_values"]
            out["asymmetry_values"] = values.reshape(B, T, *values.shape[1:])

        if "asymmetry_coords" in out:
            coords = out["asymmetry_coords"]
            out["asymmetry_coords"] = coords.reshape(B, T, *coords.shape[1:])

        if "heatmap" in out:
            heatmap = out["heatmap"]
            out["heatmap"] = heatmap.reshape(B, T, *heatmap.shape[1:])

        return out

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

            sad_cc = self._run_sad_pair(
                left_cc,
                right_cc,
                B,
                T,
                C_feat,
                H_feat,
                W_feat,
            )
            sad_mlo = self._run_sad_pair(
                left_mlo,
                right_mlo,
                B,
                T,
                C_feat,
                H_feat,
                W_feat,
            )

            asymmetry_scores = 0.5 * (
                sad_cc["asymmetry_values"] + sad_mlo["asymmetry_values"]
            )

            if "asymmetry_coords" in sad_cc and "asymmetry_coords" in sad_mlo:
                asymmetry_coords = torch.stack(
                    [sad_cc["asymmetry_coords"], sad_mlo["asymmetry_coords"]],
                    dim=2,
                )

            if "heatmap" in sad_cc and "heatmap" in sad_mlo:
                asymmetry_heatmap = torch.stack(
                    [sad_cc["heatmap"], sad_mlo["heatmap"]],
                    dim=2,
                )

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

            if r_aa.dim() > 2:
                r_aa = r_aa.reshape(B, -1)

            if r_aa.shape[1] != self.asym_dim:
                raise RuntimeError(
                    f"Asymmetry feature has dim {r_aa.shape[1]}, "
                    f"but risk head expects asym_dim={self.asym_dim}."
                )

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
