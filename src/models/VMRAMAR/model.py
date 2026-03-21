import sys
import torch
import torch.nn as nn

from .vmrnn import VMRNN
from .image_aggregator import ImageAggregator
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from config.config import cfg
from models.Mirai import onconet as _onconet

sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

from models.Mirai.onconet.models.factory import get_model_by_name, load_model

class IdentityPool(nn.Module):
    def forward(self, x):
        # Return dummy "logit" and hidden to match the old interface
        return None, x  # logit=None, hidden=feature_map
    def replaces_fc(self):
        return False

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

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim    
        self.image_encoder._model.pool = IdentityPool()  # removes global pooling
        self.image_encoder._model.fc   = nn.Identity()  # removes fully connected
        self.image_encoder._model.prob_of_failure_layer = nn.Identity()  # optional
        

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

    def forward(self, batch):
        x = batch["images"]                              # (B, C, N, H, W)
        B, T, C, V, H, W = x.shape
        x = x.view(B * T * V, C, H, W)

        # ── Encode all images ─────────────────────────────────────────
        _, img_feats, _ = self.image_encoder(x, None, batch) # img feat shape [B*T*V, C_feat, H_feat, W_feat]
        C_feat, H_feat, W_feat = img_feats.shape[1:]
        
        img_feats = img_feats.view(B, T, V, C_feat, H_feat, W_feat)
        
        # ── Fuse features across views (left/right) ──────────────────
        fused_feats = img_feats.mean(dim=2)  # average over V → [B, T, C, H, W]

         # ── Asymmetry features ───────────────────────────────────────
        if self.use_asymmetry:
            # Extract left/right breasts
            left_feats = img_feats[:, :, 0, :, :, :]   # [B, T, C, H, W]
            right_feats = img_feats[:, :, 1, :, :, :]  # [B, T, C, H, W]

            # Merge batch and temporal dims for SAD module
            left_feats = left_feats.view(B * T, C_feat, H_feat, W_feat)
            right_feats = right_feats.view(B * T, C_feat, H_feat, W_feat)

            # Apply SAD alignment
            sad_out = self.sad(left_feats, right_feats)
            asym_feats = sad_out["asymmetry_values"]  # shape (B*T,)
            asym_feature = self.lat(asym_feats.view(B, T, -1))

            # Temporal pooling
            temporal_feature = fused_feats.view(B, T, -1).mean(dim=1)

            # Combine features
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            # Only temporal pooling
            combined_feats = fused_feats.view(B, T, -1).mean(dim=1)

        # ── Risk prediction ───────────────────────────────────────────
        risk = self.ahl(combined_feats)
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