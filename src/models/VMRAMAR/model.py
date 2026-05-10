import torch
import torch.nn as nn
from .vmrnn import VMRNNEncoder
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.factory import get_model_by_name, load_model
from .model_utils import (
    FORMAL_VIEW_SEQUENCE,
    compute_asymmetry_feature,
    expand_risk_factors_per_img,
    register_onconet_alias,
    get_img_repr_dim,
    model_args,
    setup_feature_map_hook,
    zero_risk_factors_for_args,freeze_encoder
)

register_onconet_alias(_onconet)


class VMRAMaR(BaseRiskModel):
    def __init__(self, args, image_encoder=None, vmrnn=None, sad_module=None, lat_module=None):
        super(VMRAMaR, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot,
                args,
                do_wrap_model=False,
            )
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        if getattr(args, "freeze_image_encoder", False):
            freeze_encoder(self.image_encoder) 
        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if vmrnn is not None:
            self.vmrnn = vmrnn
        elif getattr(args, "vmrnn_snapshot", None) is not None:
            self.vmrnn = load_model(args.vmrnn_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.vmrnn = get_model_by_name('vmrnn', False, args)
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = sad_module or get_model_by_name('sad', False, args)
            self.lat = lat_module or get_model_by_name('lat', False, args)
        self.ahl = CumulativeProbabilityLayer(512, args, max_followup=5)

    def forward(self, x, risk_factors=None, batch=None):
        B, T, C, V, H, W = x.size()
        x = x.view(B * T * V, C, H, W)

        if risk_factors is not None:
            risk_factors_per_img = [
                factor.expand([V, *factor.size()]).contiguous().view(-1, factor.size(-1))
                for factor in risk_factors
            ]
        else:
            risk_factors_per_img = None
        _, img_feats, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_feats = img_feats.view(B, T, V, -1)
        img_feats = img_feats[:, :, :, :self.image_repr_dim]

        fused_feats = img_feats.mean(dim=2)
        """  
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
        """
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

