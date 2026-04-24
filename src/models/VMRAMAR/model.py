import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmrnn import VMRNN
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from .visit_aggregator import VisitAggregator
from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.factory import get_model_by_name, load_model


sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )


def _disable_inplace_relu(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


class IdentityPool(nn.Module):
    def forward(self, x):
        return None, x

    def replaces_fc(self):
        return False


class VMRAMaR(BaseRiskModel):
    def __init__(self, args):
        super().__init__(args)

        self.image_encoder = self._init_image_encoder(self.args)

        if getattr(self.args, "freeze_image_encoder", True):
            self._freeze_encoder(self.image_encoder)

        self.image_encoder._model.pool = IdentityPool()
        self.image_encoder._model.fc = nn.Identity()
        self.image_encoder._model.prob_of_failure_layer = nn.Identity()
        _disable_inplace_relu(self.image_encoder)

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        self.embed_dim = self.args.embed_dim

        # Per-view-type projections: CC and MLO stay separate first
        self.cc_fc = nn.Linear(self.image_repr_dim, self.embed_dim)
        self.mlo_fc = nn.Linear(self.image_repr_dim, self.embed_dim)

        # Aggregate left/right inside each view type
        self.cc_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.mlo_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)

        # Fuse CC and MLO into one exam embedding
        self.exam_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(getattr(self.args, "dropout", 0.0)),
        )

        self.temporal_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.visit_aggregator = VisitAggregator(self.args)

        self.vmrnn = VMRNN(
            embed_dim=self.embed_dim,
            depths_downsample=self.args.depths_downsample,
            depths_upsample=self.args.depths_upsample,
            feature_resolution=(1, 1),
        )

        self.use_asymmetry = getattr(self.args, "use_asymmetry", True)
        if self.use_asymmetry:
            self.sad = SpatialAsymmetryDetector(
                latent_h=int(self.args.latent_h),
                latent_w=int(self.args.latent_w),
            )
            self.lat = LongitudinalAsymmetryTracker(
                threshold_ratio=float(getattr(self.args, "threshold_ratio", 0.4)),
                persistent_weight=float(getattr(self.args, "persistent_weight", 1.0)),
            )

        final_dim = self.embed_dim + (1 if self.use_asymmetry else 0)
        self.fusion_norm = nn.LayerNorm(final_dim)
        self.ahl = CumulativeProbabilityLayer(final_dim, max_followup=5)

        # Assumed view grouping: ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']
        self.cc_indices = getattr(self.args, "cc_indices", [0, 2])
        self.mlo_indices = getattr(self.args, "mlo_indices", [1, 3])

    def _init_image_encoder(self, args):
        if getattr(args, "img_encoder_snapshot", None):
            encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
            if getattr(args, "replace_snapshot_pool", True):
                new_encoder = get_model_by_name("custom_resnet", False, args)
                encoder._model.pool = new_encoder._model.pool
                encoder._model.fc = new_encoder._model.fc
                encoder._model.prob_of_failure_layer = new_encoder._model.prob_of_failure_layer
                encoder._model.args = new_encoder._model.args
        else:
            encoder = get_model_by_name("custom_resnet", False, args)
        return encoder

    @staticmethod
    def _freeze_encoder(encoder):
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("[INFO] Image encoder frozen.")

    def _aggregate_view_type(self, feats, proj, attn):
        # feats: (B, T, N_viewtype, C)
        B, T, Nv, C = feats.shape
        feats = proj(feats)                    # (B, T, Nv, D)
        feats = feats.view(B * T, Nv, -1)     # (B*T, Nv, D)
        attn_out, _ = attn(feats, feats, feats)
        pooled = attn_out.mean(dim=1)         # (B*T, D)
        return pooled.view(B, T, -1)          # (B, T, D)

    def forward(self, batch):
        x = batch["images"]          # (B, T, V, C, H, W)
        exam_mask = batch["exam_mask"]
        view_mask = batch["view_mask"]

        B, T, V, C, H, W = x.size()

        x_flat = x.view(B * T * V, C, H, W)
        _, feat_maps, _ = self.image_encoder(x_flat, None, batch)

        C_feat, Hf, Wf = feat_maps.shape[1:]
        feat_maps = feat_maps.view(B, T, V, C_feat, Hf, Wf)

        pooled_feats = feat_maps.mean(dim=(-2, -1))  # (B, T, V, C_feat)

        cc_feats = pooled_feats[:, :, self.cc_indices, :]
        mlo_feats = pooled_feats[:, :, self.mlo_indices, :]

        cc_exam = self._aggregate_view_type(cc_feats, self.cc_fc, self.cc_attn)     # (B, T, D)
        mlo_exam = self._aggregate_view_type(mlo_feats, self.mlo_fc, self.mlo_attn) # (B, T, D)

        exam_embeddings = self.exam_fusion(torch.cat([cc_exam, mlo_exam], dim=-1))   # (B, T, D)
        exam_embeddings = self.temporal_projection(exam_embeddings)
        exam_embeddings = self.visit_aggregator(exam_embeddings, exam_mask)

        history_embedding, states, reconstructions = self.vmrnn(exam_embeddings)

        lengths = exam_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        temporal_feature = history_embedding[
            torch.arange(B, device=history_embedding.device),
            lengths
        ]  # (B, D)

        features = [temporal_feature]

        if self.use_asymmetry and V >= 2:
            asymmetry_scores, coords, coord_valid = self.sad(feat_maps, view_mask)

            window_size = max(
                int(getattr(self.sad, "latent_h")),
                int(getattr(self.sad, "latent_w")),
            )

            r_aa = self.lat(
                asymmetry_scores,
                coords,
                coord_valid,
                exam_mask,
                window_size=window_size,
            )

            if r_aa.dim() == 1:
                r_aa = r_aa.unsqueeze(-1)
            elif r_aa.dim() != 2:
                raise ValueError(f"Unexpected asymmetry feature shape: {r_aa.shape}")

            features.append(r_aa)

        for i, f in enumerate(features):
            if f.dim() != 2 or f.size(0) != B:
                raise ValueError(
                    f"Feature {i} has invalid shape {f.shape}, expected (B, D)"
                )

        combined = torch.cat(features, dim=-1)
        combined = self.fusion_norm(combined)
        risk = self.ahl(combined)

        return {"logit": risk}

    def get_risk_heads(self, outputs, batch):
        return {"logit_output": (outputs["logit"], batch["target"], batch["y_mask"])}

    def get_primary_risk_head(self, outputs):
        return torch.sigmoid(outputs["logit"])
