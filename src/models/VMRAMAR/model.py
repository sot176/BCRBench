from models.Mirai.onconet.models.factory import load_model, RegisterModel, get_model_by_name
import torch
import torch.nn as nn
import sys

from models.common_parts  import  CumulativeProbabilityLayer
from models.common_parts import extract_mirai_backbone
from config.config import cfg


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
            self.vmrnn = get_model_by_name('vmrnn', False, args)
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = sad_module or get_model_by_name('sad', False, args)
            self.lat = lat_module or get_model_by_name('lat', False, args)
        self.ahl = CumulativeProbabilityLayer(512, max_followup=5)

    def forward(self, x, risk_factors=None, batch=None):
        B, T, C, V, H, W = x.size()
        x = x.view(B * T * V, C, H, W)

        img_feats= self.image_encoder(x)
        img_feats = img_feats.view(B, T, V, -1)
        #img_feats = img_feats[:, :, :, :self.image_repr_dim]
        print("img feats shape", img_feats.shape)
        fused_feats = img_feats.mean(dim=2)
          
        temporal_output, hidden_states = self.vmrnn(fused_feats, risk_factors, batch)
        if self.use_asymmetry:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]
            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            asym_feature = self.lat(asym_feats)
            temporal_feature = temporal_output.mean(dim=1)
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            combined_feats = temporal_output.mean(dim=1)
        risk_pred = self.ahl(combined_feats)
        
        # If asymmetry is used, compute left-right features
        if self.use_asymmetry:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]
            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            asym_feature = self.lat(asym_feats)
            # Temporal pooling (mean over time)
            temporal_feature = fused_feats.mean(dim=1)
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            # Just average over time dimension
            combined_feats = fused_feats.mean(dim=1)

        # Predict risk
        risk_pred = self.ahl(combined_feats)
        return risk_pred
