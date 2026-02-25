import torch
import torch.nn as nn

from .model_utils import CrossAttentionBlock, LongitudinalFeatureProcessor
from models.common_parts  import  CumulativeProbabilityLayer
from utils import get_risk_loss_BCE


class LMVNet(nn.Module):
    def __init__(self, mammo_reg_net: nn.Module, max_followup: int = 5,
                 num_attn_blocks: int = 1, feature_dim: int = 1536,  finetune_all: bool = False):
        super().__init__()

        self.longitudinal_feat_processor = LongitudinalFeatureProcessor(mammo_reg_net=mammo_reg_net, finetune_all=finetune_all)

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(in_channels=feature_dim, reduced_channels=feature_dim,
                                heads=4, dropout=0.3, drop_path=0.2, ffn_expansion_factor=2)
            for _ in range(num_attn_blocks)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion layer for multi-view pooled vector
        self.view_fc = nn.Linear(feature_dim * 2, feature_dim)

        # Separate cumulative probability layers
        self.cumulative_risk = CumulativeProbabilityLayer(num_features=feature_dim, max_followup=max_followup)

    def forward(self, batch):

        img_cur_cc = batch["current_image_cc"]
        img_pri_cc = batch["previous_image_cc"]
        img_cur_mlo = batch["current_image_mlo"]
        img_pri_mlo = batch["previous_image_mlo"]
        time_gap = batch["time_gap"]

        # Step 1: Longitudinal features
        longitudinal_output = self.longitudinal_feat_processor(
            img_cur_cc, img_pri_cc, img_cur_mlo, img_pri_mlo, time_gap
        )

        f_cc = longitudinal_output['f_cc_long']  # [B, 1536, H, W]
        f_mlo = longitudinal_output['f_mlo_long']  # [B, 1536, H, W]

        # Step 2: Cross-attention blocks
        for blk in self.cross_attn_blocks:
            f_cc, f_mlo = blk(f_cc, f_mlo)

        # Step 3: Multi-view fusion by concatenation
        # Pool each view separately
        pooled_cc = self.global_avg_pool(f_cc).flatten(1)  # [B, 1536]
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1) # [B, 1536]

        # Concatenate and reduce dimensionality
        pooled_multi = torch.cat([pooled_cc, pooled_mlo], dim=1)  # [B, 1536*2]
        pooled_multi = self.view_fc(pooled_multi)  # [B, 1536]

        # Step 5: Separate risk predictions
        risk_multi = self.cumulative_risk(pooled_multi)
        risk_cc = self.cumulative_risk(pooled_cc)
        risk_mlo = self.cumulative_risk(pooled_mlo)

        return {'risk_multi': risk_multi, 'risk_cc': risk_cc, 'risk_mlo': risk_mlo}

    def get_risk_heads(self, outputs, batch):
        """
        Returns a dictionary of:
        {
            head_name: (logits, target, mask)
        }
        """

        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "multi": (outputs["risk_multi"], target, mask),
            "cc": (outputs["risk_cc"], target, mask),
            "mlo": (outputs["risk_mlo"], target, mask),
        }

    def get_primary_risk_head(self, outputs):
        return outputs["risk_multi"]
    
    def compute_total_loss(self, outputs, batch):
        risk_heads = self.get_risk_heads(outputs, batch)

        return sum(
            get_risk_loss_BCE(logits, target, mask)
            for logits, target, mask in risk_heads.values()
        )