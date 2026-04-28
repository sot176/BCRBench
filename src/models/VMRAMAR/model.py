import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_parts import extract_mirai_backbone
from config.config import cfg

from .vmrnn import VMRNNEncoder
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from .visit_aggregator import VisitAggregator
from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.pools import Simple_AttentionPool


sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

class VMRAMaR(BaseRiskModel):
    def __init__(self, args):
        super().__init__(args)

        # -------------------------
        # Frozen image encoder
        # -------------------------
        self.image_encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])

        if getattr(self.args, "freeze_image_encoder", True):
            self._freeze_encoder(self.image_encoder)
        self.pool = Simple_AttentionPool(self.args, self.args.transformer_hidden_dim)

        # -------------------------
        # Multi-view exam encoder
        # -------------------------
        self.cc_indices = getattr(self.args, "cc_indices", [0, 2])
        self.mlo_indices = getattr(self.args, "mlo_indices", [1, 3])

        self.cc_fc = nn.Linear(self.args.image_repr_dim, self.args.embed_dim)
        self.mlo_fc = nn.Linear(self.args.image_repr_dim, self.args.embed_dim)

        self.exam_fusion = nn.Sequential(
            nn.Linear(2 * self.args.embed_dim, self.args.embed_dim),
            nn.GELU(),
            nn.Dropout(getattr(self.args, "dropout", 0.1)),
        )

        self.temporal_projection = nn.Linear(self.args.embed_dim, self.args.embed_dim)
        self.visit_aggregator = VisitAggregator(self.args)

        # -------------------------
        # Longitudinal encoder
        # -------------------------
        # Exam embeddings are already pooled to one vector per exam, so the
        # VMRNN must operate on a single-token representation.
        self.vmrnn = VMRNNEncoder(
            input_dim=self.args.embed_dim,
            hidden_dim=128,
            spatial_resolution=(4, 4),
            downsample_depths=self.args.depths_downsample,
            upsample_depths=self.args.depths_upsample,
            dropout=0.1,
            vss_backend="auto",   # falls back to torch if VMamba is unavailable
        )

        # -------------------------
        # Asymmetry branch
        # -------------------------
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

        # -------------------------
        # Risk head
        # -------------------------
        final_dim = 128 + (1 if self.use_asymmetry else 0)
        self.fusion_norm = nn.LayerNorm(final_dim)
        self.ahl = CumulativeProbabilityLayer(final_dim, max_followup=5)

    @staticmethod
    def _freeze_encoder(encoder):
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("[INFO] Image encoder frozen.")

    def _masked_mean(self, x, mask):
        """
        x:    (B, T, V, D)
        mask: (B, T, V)
        """
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp(min=1.0)
        return summed / denom

    def _encode_exams(self, pooled_feats, view_mask, exam_mask):
        """
        pooled_feats: (B, T, V, C_feat)
        view_mask:    (B, T, V)
        exam_mask:    (B, T)
        returns:
            exam_embeddings: (B, T, D)
        """
        cc_feats = pooled_feats[:, :, self.cc_indices, :]
        mlo_feats = pooled_feats[:, :, self.mlo_indices, :]

        cc_mask = view_mask[:, :, self.cc_indices]
        mlo_mask = view_mask[:, :, self.mlo_indices]

        cc_tokens = self.cc_fc(cc_feats)
        mlo_tokens = self.mlo_fc(mlo_feats)

        cc_exam = self._masked_mean(cc_tokens, cc_mask)
        mlo_exam = self._masked_mean(mlo_tokens, mlo_mask)

        exam_embeddings = self.exam_fusion(torch.cat([cc_exam, mlo_exam], dim=-1))
        exam_embeddings = self.temporal_projection(exam_embeddings)
        exam_embeddings = self.visit_aggregator(exam_embeddings, exam_mask)

        return exam_embeddings

    def _compute_asymmetry_feature(self, feat_maps, view_mask, exam_mask):
        """
        returns:
            r_aa: (B, 1)
            asymmetry_scores, coords, coord_valid
        """
        asymmetry_scores, coords, coord_valid = self.sad(feat_maps, view_mask)

        window_size = max(
            int(self.sad.latent_h),
            int(self.sad.latent_w),
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

        return r_aa, asymmetry_scores, coords, coord_valid

    def forward(self, batch):
        images = batch["images"]       # (B, T, V, C, H, W)
        exam_mask = batch["exam_mask"] # (B, T)
        view_mask = batch["view_mask"] # (B, T, V)

        B, T, V, C, H, W = images.size()

        # 1. Image encoding
        x_flat = images.view(B * T * V, C, H, W)
        feat_maps = self.image_encoder(x_flat)  # (BTV, C_feat, Hf, Wf)

        _, pooled_feats = self.pool(feat_maps)  # (BTV, C_feat)
        pooled_feats = pooled_feats.view(B, T, V, -1)

        # 2. Exam encoding
        exam_embeddings = self._encode_exams(pooled_feats, view_mask, exam_mask)

        # 3. Longitudinal encoding
        history_embedding, states, reconstructions = self.vmrnn(exam_embeddings, exam_mask)

        features = [history_embedding]

        # 4. Asymmetry feature
        asymmetry_scores = None
        coords = None
        coord_valid = None
        r_aa = None

        if self.use_asymmetry and V >= 2:
            c_feat, hf, wf = feat_maps.shape[1:]
            feat_maps = feat_maps.view(B, T, V, c_feat, hf, wf)
            r_aa, asymmetry_scores, coords, coord_valid = self._compute_asymmetry_feature(
                feat_maps,
                view_mask,
                exam_mask,
            )
            features.append(r_aa)

        for i, f in enumerate(features):
            if f.dim() != 2 or f.size(0) != B:
                raise ValueError(
                    f"Feature {i} has invalid shape {f.shape}, expected (B, D)"
                )

        # 5. Risk prediction
        combined = torch.cat(features, dim=-1)
        combined = self.fusion_norm(combined)

        logits = self.ahl(combined)
        probs = torch.sigmoid(logits)

        return {
            "logit": logits,
            "probs": probs,
            "history_embedding": history_embedding,
            "exam_embeddings": exam_embeddings,
            "states": states,
            "vmrnn_reconstruction": reconstructions,
            "exam_asymmetry": asymmetry_scores,
            "coords": coords,
            "coord_valid": coord_valid,
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

