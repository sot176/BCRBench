from pyexpat import features
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmrnn import TransformerVMRNNEncoder
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
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
    """Disable inplace ReLU to avoid autograd issues."""
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


class IdentityPool(nn.Module):
    """Dummy pool layer for snapshot encoders."""

    def forward(self, x):
        return None, x  # logit=None, hidden=feature_map

    def replaces_fc(self):
        return False


class VMRAMaR(BaseRiskModel):
    """
    Stable reimplementation of VMRNN-Asymmetry Mammogram Risk model.

    Args:
        args: configuration object
    """

    def __init__(self, args):
        super().__init__(args)

        # -------------------------
        # 1. Image Encoder
        # -------------------------
        self.image_encoder = self._init_image_encoder(self.args)

        if getattr(self.args, "freeze_image_encoder", True):
            self._freeze_encoder(self.image_encoder)

        self.image_encoder._model.pool = IdentityPool()
        self.image_encoder._model.fc = nn.Identity()
        self.image_encoder._model.prob_of_failure_layer = nn.Identity()
        _disable_inplace_relu(self.image_encoder)

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim

        # -------------------------
        # 2. View → Exam Aggregation 
        # -------------------------
        self.view_fc = nn.Linear(self.image_repr_dim, self.args.embed_dim)
        self.view_attn = nn.MultiheadAttention(
            self.args.embed_dim, num_heads=4, batch_first=True
        )

        # -------------------------
        # 3. Temporal Projection + VMRNN
        # -------------------------
        self.temporal_projection = nn.Linear(self.args.embed_dim, self.args.embed_dim)

        self.vmrnn = TransformerVMRNNEncoder( input_dim=self.args.embed_dim, hidden_dim=self.args.embed_dim )

        # -------------------------
        # 4. Asymmetry Branch (fixed)
        # -------------------------
        self.use_asymmetry = getattr(self.args, "use_asymmetry", True)

        if self.use_asymmetry:
            self.sad = SpatialAsymmetryDetector( latent_h=int(self.args.latent_h), latent_w=int(self.args.latent_w),)
            self.lat = LongitudinalAsymmetryTracker(
            threshold_ratio=float(getattr(self.args, "threshold_ratio", 0.4)),
            persistent_weight=float(getattr(self.args, "persistent_weight", 1.0)),
        )

        # -------------------------
        # 5. Final Prediction
        # -------------------------
        final_dim = self.args.embed_dim + (1 if self.use_asymmetry else 0)
        self.fusion_norm = nn.LayerNorm(final_dim)
        self.ahl = CumulativeProbabilityLayer(final_dim, max_followup=5)


    def _init_image_encoder(self, args):
        """Initialize the image encoder, optionally loading a snapshot."""
        if getattr(args, "img_encoder_snapshot", None):
            encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
            if getattr(args, "replace_snapshot_pool", True):
                new_encoder = get_model_by_name("custom_resnet", False, args)
                # Replace pool, fc, prob_of_failure_layer, and args
                encoder._model.pool = new_encoder._model.pool
                encoder._model.fc = new_encoder._model.fc
                encoder._model.prob_of_failure_layer = new_encoder._model.prob_of_failure_layer
                encoder._model.args = new_encoder._model.args
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

    def forward(self, batch):
        x = batch["images"]          # (B, T, V, C, H, W)
        exam_mask = batch["exam_mask"]  # (B, T)
        view_mask = batch["view_mask"]  # (B, T, V)

        B, T, V, C, H, W = x.size()

        # -------------------------
        # Encode images
        # -------------------------
        x_flat = x.view(B * T * V, C, H, W)
        _, feat_maps, _ = self.image_encoder(x_flat, None, batch)

        C_feat, Hf, Wf = feat_maps.shape[1:]
        feat_maps = feat_maps.view(B, T, V, C_feat, Hf, Wf)

        # -------------------------
        # View → exam embedding
        # -------------------------
        feats = feat_maps.mean(dim=(-2, -1))   # (B, T, V, C)
        feats = self.view_fc(feats)            # (B, T, V, D)

        feats_bt = feats.view(B * T, V, -1)
        attn_out, _ = self.view_attn(feats_bt, feats_bt, feats_bt)

        exam_embeddings = attn_out.mean(dim=1).view(B, T, -1)  # (B, T, D)

        # -------------------------
        # Temporal modeling  
        # -------------------------
        exam_embeddings = self.temporal_projection(exam_embeddings)

        history_embedding, _, _ = self.vmrnn(exam_embeddings, exam_mask)

        # Use LAST valid timestep  
        last_idx = exam_mask.sum(dim=1) - 1
        temporal_feature = history_embedding[torch.arange(B), last_idx]

        features = [temporal_feature]

        # -------------------------
        # Asymmetry branch (FIXED)
        # -------------------------
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

            features.append(r_aa.unsqueeze(-1))

        # -------------------------
        # Final prediction
        # -------------------------
        combined = torch.cat(features, dim=-1)
        combined = self.fusion_norm(combined)

        risk = self.ahl(combined)

        return {"logit": risk}

    # ── Risk helpers ─────────────────────────────

    def get_risk_heads(self, outputs, batch):
        return {"logit_output": (outputs["logit"], batch["target"], batch["y_mask"])}

    def get_primary_risk_head(self, outputs):
        return torch.sigmoid(outputs["logit"])