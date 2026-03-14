from models.Mirai.onconet.models.factory import load_model, RegisterModel
import torch
import torch.nn as nn
import sys

from models.common_parts  import  CumulativeProbabilityLayer
from models.common_parts import extract_mirai_backbone
from config.config import cfg
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from .vmrnn import VMRNN

@RegisterModel("vmra_mar")
class VMRAMaR(nn.Module):
    def __init__(self, args, image_encoder=None, vmrnn=None, sad_module=None, lat_module=None):
        super(VMRAMaR, self).__init__()
        self.args = args
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.image_encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if vmrnn is not None:
            self.vmrnn = vmrnn
        elif getattr(args, "vmrnn_snapshot", None) is not None:
            self.vmrnn = load_model(args.vmrnn_snapshot, args, do_wrap_model=False)
        else:
            self.vmrnn = VMRNN(args.embed_dim, args.depths_downsample, args.depths_upsample, args.feature_resolution)
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = sad_module or SpatialAsymmetryDetector(args)
            self.lat = lat_module or LongitudinalAsymmetryTracker(args)
        self.ahl = CumulativeProbabilityLayer(512, max_followup=5)

    def forward(self, data, risk_factors=None, batch=None):
        x = data['images']  # shape: (B, T, C, V, H, W)
        B, T, C, V, H, W = x.size()

        # Flatten batch, time, and views for encoder
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, T, V, C, H, W)
        x = x.view(B * T * V, C, H, W)

        # Encode images
        img_feats = self.image_encoder(x)  # (B*T*V, C_feat, Hf, Wf)
        C_feat, Hf, Wf = img_feats.shape[1:]

        # Reshape to keep views separate
        img_feats = img_feats.view(B, T, V, C_feat * Hf * Wf)  # (B, T, V, L_raw)

        # Ensure fixed feature length for VMRNN
        L_expected = 64 * 52
        if img_feats.shape[-1] > L_expected:
            img_feats = img_feats[:, :, :, :L_expected]
        elif img_feats.shape[-1] < L_expected:
            pad_size = L_expected - img_feats.shape[-1]
            padding = torch.zeros(B, T, V, pad_size, device=img_feats.device)
            img_feats = torch.cat([img_feats, padding], dim=-1)

        # Fuse multiple views (e.g., left/right breast)
        fused_feats = img_feats.mean(dim=2)  # shape: (B, T, L_expected)

        # Pass through VMRNN
        temporal_output, states_down, states_up = self.vmrnn(fused_feats, risk_factors, batch)
        temporal_feature = temporal_output.mean(dim=1)  # mean over time

        # Optional asymmetry features
        if self.use_asymmetry and V >= 2:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]
            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            asym_feature = self.lat(asym_feats)
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            combined_feats = temporal_feature

        # Risk prediction
        risk_pred = self.ahl(combined_feats)
        return {'logit': risk_pred}

    def get_risk_heads(self, outputs, batch):
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "logit_output": (outputs["logit"], target, mask) }
    
    def get_primary_risk_head(self, outputs):
        return outputs["logit"]