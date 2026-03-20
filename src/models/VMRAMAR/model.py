import sys
import torch
import torch.nn as nn

from .vmrnn import VMRNN
from .image_aggregator import ImageAggregator
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from config.config import cfg
from models.Mirai.onconet.models.factory import get_model_by_name, load_model


class VMRAMaR(nn.Module):
    """
    VMRNN-Asymmetry Mammogram Risk model.

    Pipeline:
        1. Image encoder  — shared Mirai ResNet backbone
        2. Image aggregator — fuses CC/MLO views per visit → T_t
        3. VMRNN — temporal modeling across visits
        4. (optional) Asymmetry branch — SAD + LAT
        5. AHL — additive hazard layer → 5-year risk
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ── 1. Image encoder ──────────────────────────────────────────
        if args.img_encoder_snapshot is not None:
                self.image_encoder = load_model(
                    args.img_encoder_snapshot, args, do_wrap_model=False
                )
                if getattr(args, "replace_snapshot_pool", True):
                    non_trained_encoder = get_model_by_name("custom_resnet", False, args)
                    # Replace pool, fc, and prob_of_failure_layer — all depend on hidden dim
                    self.image_encoder._model.pool               = non_trained_encoder._model.pool
                    self.image_encoder._model.fc                 = non_trained_encoder._model.fc
                    self.image_encoder._model.prob_of_failure_layer = non_trained_encoder._model.prob_of_failure_layer
                    self.image_encoder._model.args               = non_trained_encoder._model.args
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        # ── 2. Image aggregator ───────────────────────────────────────
        num_views = getattr(args, "num_images", 4)
        self.image_aggregator = ImageAggregator(args.embed_dim, num_views=num_views)

        # ── 3. VMRNN ──────────────────────────────────────────────────
        self.vmrnn = VMRNN(
            embed_dim=args.embed_dim,
            depths_downsample=args.depths_downsample,
            depths_upsample=args.depths_upsample,
            feature_resolution=(1, 1),          # temporal mode
        )

        # ── 4. Asymmetry modules ──────────────────────────────────────
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad      = SpatialAsymmetryDetector(args)
            self.lat      = LongitudinalAsymmetryTracker(args)
            latent_h      = getattr(args, "latent_h", 52)
            latent_w      = getattr(args, "latent_w", 64)
            self.asym_proj = nn.Linear(latent_h * latent_w, 512)

        # ── 5. Additive Hazard Layer ───────────────────────────────────
        ahl_input_dim = args.embed_dim
        if self.use_asymmetry:
            ahl_input_dim += 512
        self.ahl = CumulativeProbabilityLayer(ahl_input_dim, max_followup=5)

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, data, risk_factors=None):
        x = data["images"]                              # (B, T, C, V, H, W)
        B, T, C, V, H, W = x.shape

        # ── Encode all images ─────────────────────────────────────────
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, T, V, C, H, W)
        x = x.view(B * T * V, C, H, W)
        feats = self.image_encoder(x)                  # (B·T·V, C_feat, Hf, Wf)
        _, C_feat, Hf, Wf = feats.shape
        feats = feats.view(B, T, V, C_feat, Hf, Wf)   # (B, T, V, C, Hf, Wf)

        # ── Aggregate views ───────────────────────────────────────────
        feats_pooled     = feats.mean(dim=(-2, -1))    # (B, T, V, C)
        visit_embeddings = self.image_aggregator(feats_pooled)  # (B, T, C)

        # ── Temporal modeling ─────────────────────────────────────────
        out, _, _ = self.vmrnn(visit_embeddings)       # (B, T, C)
        temporal_feature = out.mean(dim=1) # (B, C)

        # ── Asymmetry features ────────────────────────────────────────
        features = [temporal_feature]
        if self.use_asymmetry and V >= 2:
            # Views: 0=left CC, 1=right CC, 2=left MLO, 3=right MLO
            left  = feats[:, :, [0, 2]].mean(dim=2)   # (B, T, C, Hf, Wf)
            right = feats[:, :, [1, 3]].mean(dim=2)   # (B, T, C, Hf, Wf)

            asym     = self.sad(left, right)
            heatmaps = asym["heatmap"]
            if heatmaps.dim() == 3:
                _, H_a, W_a = heatmaps.shape
                heatmaps = heatmaps.view(B, T, H_a, W_a)

            coords = asym["asymmetry_coords"]
            if coords.dim() == 2:
                coords = coords.view(B, T, 2)

            B_a, T_a, H_a, W_a = heatmaps.shape
            asym_features = self.asym_proj(
                heatmaps.view(B_a, T_a, H_a * W_a)    # (B, T, 25)
            )                                           # (B, T, 512)
            asym_feature = self.lat(
                asym_features,   # (B, T, 512)
                coords,          # (B, T, 2)
                heatmaps,        # (B, T, H_a, W_a)
            )                    # (B, 512)
            features.append(asym_feature)

        holistic_embedding = torch.cat(features, dim=1)

        # ── Risk prediction ───────────────────────────────────────────
        risk = self.ahl(holistic_embedding)
        return {"logit": risk}

    # ── loss helpers ──────────────────────────────────────────────────

    def get_risk_heads(self, outputs, batch):
        return {
            "logit_output": (
                outputs["logit"],
                batch["target"],
                batch["y_mask"],
            )
        }

    def get_primary_risk_head(self, outputs):
        return outputs["logit"]