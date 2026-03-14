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
        x = data['images']
        B, T, C, V, H, W = x.size()
        x = x.view(B * T * V, C, H, W)

        img_feats = self.image_encoder(x)
        C_feat, Hf, Wf = img_feats.shape[1:]

        img_feats = img_feats.view(B, T, V, C_feat, Hf, Wf)
        fused_feats = img_feats.mean(dim=2)  # fuse left/right views

        fused_feats = fused_feats.view(B*T, C_feat, Hf, Wf)
        fused_feats = fused_feats.flatten(2).transpose(1,2)  # (B*T, Hf*Wf, C)

        temporal_output, _, _ = self.vmrnn(fused_feats)
        temporal_feature = temporal_output.view(B, T, -1).mean(dim=1)

        if self.use_asymmetry:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]
            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            asym_feature = self.lat(asym_feats)
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            combined_feats = temporal_feature
        
        risk_pred = self.ahl(combined_feats)
        return {'logit': risk_pred}
    
    def get_risk_heads(self, outputs, batch):
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "logit_output": (outputs["logit"], target, mask) }
    
    def get_primary_risk_head(self, outputs):
        return outputs["logit"]