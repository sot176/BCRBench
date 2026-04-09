import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmrnn import VMRNN
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from config.config import cfg
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel

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


from models.Mirai.onconet.models.factory import get_model_by_name, load_model


class IdentityPool(nn.Module):
    """Dummy pool layer for snapshot encoders."""

    def forward(self, x: torch.Tensor):
        return None, x  # logit=None, hidden=feature_map

    def replaces_fc(self) -> bool:
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
        # Image Encoder
        # -------------------------
        self.image_encoder = self._init_image_encoder(self.args)

        # Freeze encoder if requested
        if getattr(self.args, "freeze_image_encoder", True):
            self._freeze_encoder(self.image_encoder)

        # Replace pooling / fc layers with identity to extract features
        self.image_encoder._model.pool = IdentityPool()
        self.image_encoder._model.fc = nn.Identity()
        self.image_encoder._model.prob_of_failure_layer = nn.Identity()
        _disable_inplace_relu(self.image_encoder)

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim

        # ── 2. View Aggregation (CC / MLO) ───────────────────────────
        self.view_fc = nn.Linear(self.image_repr_dim, self.args.embed_dim)
        self.view_attn = nn.MultiheadAttention(
            self.args.embed_dim, num_heads=4, batch_first=True
        )

        # ── 3. Temporal Projection + VMRNN ───────────────────────────
        self.vmrnn_out_proj = nn.Sequential(
            nn.Linear(self.args.embed_dim, self.args.embed_dim),
            nn.LayerNorm(self.args.embed_dim),
        )
        self.vmrnn = VMRNN(
            embed_dim=self.args.embed_dim,
            depths_downsample=self.args.depths_downsample,
            depths_upsample=self.args.depths_upsample,
            feature_resolution=(1, 1),
        )

        # ── 4. Asymmetry Branch ──────────────────────────────────────
        self.use_asymmetry = getattr(self.args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = SpatialAsymmetryDetector(self.args)
            self.lat = LongitudinalAsymmetryTracker(self.args)
            latent_h = getattr(self.args, "latent_h", 5)
            latent_w = getattr(self.args, "latent_w", 5)
            self.asym_proj = nn.Sequential(
                nn.Linear(latent_h * latent_w, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
            )

        # ── 5. Final Fusion + AHL ────────────────────────────────────
        final_dim = self.args.embed_dim + (512 if self.use_asymmetry else 0)
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

    def forward(self, batch: dict) -> dict:
        """
        Forward pass.

        Args:
            batch: dict containing keys:
                images: (B, T, V, C, H, W)

        Returns:
            dict with key "logit" (B, max_followup)
        """
        x = batch["images"]  # (B, T, V, C, H, W)
        B, T, C, V, H, W = x.size()
        x = x.view(B * T * V, C, H, W)

        # ── Encode images ─────────────────────────────
        _, img_feats, _ = self.image_encoder(x, None, batch)
        C_feat, H_feat, W_feat = img_feats.shape[1:]
        img_feats = img_feats.view(B, T, V, C_feat, H_feat, W_feat)

        # ── Spatial pooling ───────────────────────────
        feats = img_feats.mean(dim=(-2, -1))  # (B, T, V, C_feat)

        # ── View aggregation ──────────────────────────
        feats = self.view_fc(feats)             # (B, T, V, D)
        feats_bt = feats.view(B * T, V, -1)
        attn_out, _ = self.view_attn(feats_bt, feats_bt, feats_bt)
        visit_embeddings = attn_out.mean(dim=1).view(B, T, -1)  # (B, T, D)

        # ── VMRNN temporal processing ───────────────
        out, _, _ = self.vmrnn(visit_embeddings)  # (B, T, D)
        temporal_feature = self.vmrnn_out_proj(out.mean(dim=1))  # (B, D)

        features = [temporal_feature]

        # ── Asymmetry branch ─────────────────────────
        if self.use_asymmetry and V >= 2:
            left_feats = img_feats[:, :, 0]   # (B, T, C, Hf, Wf)
            right_feats = img_feats[:, :, 1]  # (B, T, C, Hf, Wf)

            sad_out = self.sad(left_feats, right_feats)
            asym_maps = sad_out["heatmap"]        # (B, T, H_lat, W_lat)
            asym_coords = sad_out["asymmetry_coords"]

            # Project to 512-dim features
            B_a, T_a, H_lat, W_lat = asym_maps.shape
            asym_feature = self.asym_proj(asym_maps.view(B_a, T_a, H_lat * W_lat))  # (B, T, 512)

            # Longitudinal aggregation
            asym_feature = self.lat(asym_feature, asym_coords, asym_maps)  # (B, 512)
            features.append(asym_feature)

        # ── Fusion + risk prediction ────────────────
        combined = self.fusion_norm(torch.cat(features, dim=1))
        risk = self.ahl(combined)

        return {"logit": risk}

    # ── Risk helpers ─────────────────────────────

    def get_risk_heads(self, outputs: dict, batch: dict) -> dict:
        return {"logit_output": (outputs["logit"], batch["target"], batch["y_mask"])}

    def get_primary_risk_head(self, outputs: dict) -> torch.Tensor:
        return torch.sigmoid(outputs["logit"])