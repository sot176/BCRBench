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


def _disable_inplace_relu(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

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
        if getattr(args, "freeze_image_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim    
        self.image_encoder._model.pool = IdentityPool()  # removes global pooling
        self.image_encoder._model.fc   = nn.Identity()  # removes fully connected
        self.image_encoder._model.prob_of_failure_layer = nn.Identity()  # optional
        _disable_inplace_relu(self.image_encoder)


        # ── 2. VMRNN ──────────────────────────────────────────────────
        self.vmrnn = VMRNN(
            embed_dim=args.embed_dim,
            depths_downsample=args.depths_downsample,
            depths_upsample=args.depths_upsample,
            feature_resolution=(1, 1),          # temporal mode
        )

        # ── 3. Asymmetry modules ──────────────────────────────────────
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad      = SpatialAsymmetryDetector(args)
            self.lat      = LongitudinalAsymmetryTracker(args)
            latent_h      = getattr(args, "latent_h", 5)
            latent_w      = getattr(args, "latent_w", 5)
            self.asym_proj = nn.Linear(latent_h * latent_w, 512)

        # ── 4. Additive Hazard Layer ───────────────────────────────────
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
        
        # ── Pool spatial dims, fuse views ────────────────────────────
        feats_pooled     = img_feats.mean(dim=(-2, -1))     # (B, T, V, C)
        visit_embeddings = feats_pooled.mean(dim=2)          # (B, T, C) — mean over views

        # ── VMRNN temporal modeling ───────────────────────────────────
        out, _, _ = self.vmrnn(visit_embeddings)             # (B, T, C)
        temporal_feature = out.mean(dim=1)                   # (B, C)

         # ── Asymmetry features ────────────────────────────────────────
        features = [temporal_feature]
        if self.use_asymmetry and V >= 2:
            left_feats  = img_feats[:, :, 0]                # (B, T, C, H, W)
            right_feats = img_feats[:, :, 1]                # (B, T, C, H, W)

            sad_out     = self.sad(left_feats, right_feats)
            asym_coords = sad_out["asymmetry_coords"]        # (B, T, 2)
            asym_maps   = sad_out["heatmap"]                 # (B, T, H_lat, W_lat)

            if asym_maps.dim() == 3:
                _, H_lat, W_lat = asym_maps.shape
                asym_maps = asym_maps.view(B, T, H_lat, W_lat)

            if asym_coords.dim() == 2:
                asym_coords = asym_coords.view(B, T, 2)

            B_a, T_a, H_lat, W_lat = asym_maps.shape
            asym_feature = self.asym_proj(
                asym_maps.view(B_a, T_a, H_lat * W_lat)    # (B, T, H*W)
            )                                                # (B, T, 512)

            asym_feature = self.lat(
                asym_feature, asym_coords, asym_maps
            )                                                # (B, 512)
            features.append(asym_feature)

        combined_feats = torch.cat(features, dim=1)         # (B, C) or (B, C+512)

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
        logit =  outputs["logit"]
        pred_risk = torch.sigmoid(logit)
        return pred_risk
    
 