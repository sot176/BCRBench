import torch
import torch.nn as nn
from .model_utils import CrossAttentionBlock, LongitudinalFeatureProcessor
from models.common_parts import CumulativeProbabilityLayer
from typing import Dict, Any
from models.common_parts import BaseRiskModel


class LMVNet(BaseRiskModel):
    """
    Longitudinal Multi-View Network (LMVNet) for risk prediction.

    Processes current and previous mammography images (CC and MLO views),
    aligns features longitudinally, applies cross-attention, and outputs
    cumulative risk predictions.
    """
    def __init__(
        self,
        mammo_reg_net: nn.Module,
        args
    ):
        super().__init__(args)
       
        # Longitudinal feature processor
        self.longitudinal_feat_processor = LongitudinalFeatureProcessor(
            mammo_reg_net=mammo_reg_net,
            args=self.args
        )

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                in_channels=self.args.feature_dim,
                reduced_channels=self.args.feature_dim,
                heads=self.args.num_heads,
                dropout=self.args.dropout,
                drop_path=self.args.drop_path,
                ffn_expansion_factor=self.args.ffn_expansion_factor
            )
            for _ in range(self.args.num_attn_blocks)
        ])

        # Adaptive pooling for each view
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion layer for multi-view concatenated features
        self.view_fc = nn.Linear(self.args.feature_dim * 2, self.args.feature_dim)

        # Cumulative probability layers for risk prediction
        self.cumulative_risk = CumulativeProbabilityLayer(
            num_features=self.args.feature_dim,
            max_followup=self.args.max_followup
        )

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, batch):
        """
        Forward pass of LMVNet.

        Args:
            batch: Dict with keys:
                - "current_image_cc", "previous_image_cc"
                - "current_image_mlo", "previous_image_mlo"
                - "time_gap"
        Returns:
            Dict with risk predictions:
                - "risk_multi", "risk_cc", "risk_mlo"
        """
        # Step 1: Longitudinal feature extraction
        longitudinal_output = self.longitudinal_feat_processor(
            batch["current_image_cc"],
            batch["previous_image_cc"],
            batch["current_image_mlo"],
            batch["previous_image_mlo"],
            batch["time_gap"]
        )

        f_cc = longitudinal_output['f_cc_long']  # [B, C, H, W]
        f_mlo = longitudinal_output['f_mlo_long']  # [B, C, H, W]

        # Step 2: Cross-attention blocks
        for attn_block in self.cross_attn_blocks:
            f_cc, f_mlo = attn_block(f_cc, f_mlo)

        # Step 3: Multi-view pooling and fusion
        pooled_cc = self.global_avg_pool(f_cc).flatten(1)  # [B, C]
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1)  # [B, C]

        # Concatenate views and reduce dimensionality
        multi_view_features = torch.cat([pooled_cc, pooled_mlo], dim=1)
        multi_view_features = self.view_fc(multi_view_features)

        # Step 4: Risk prediction
        risk_multi = self.cumulative_risk(multi_view_features)
        risk_cc = self.cumulative_risk(pooled_cc)
        risk_mlo = self.cumulative_risk(pooled_mlo)

        return {
            'risk_multi': risk_multi,
            'risk_cc': risk_cc,
            'risk_mlo': risk_mlo
        }

    # -------------------------
    # Risk head helpers
    # -------------------------
    def get_risk_heads(self, outputs, batch):
        """
        Returns a dictionary of risk heads for loss computation.

        Args:
            outputs: Dict of model outputs
            batch: Dict of batch inputs
        Returns:
            Dict mapping head name → (logits, target, mask)
        """
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "multi": (outputs["risk_multi"], target, mask),
            "cc": (outputs["risk_cc"], target, mask),
            "mlo": (outputs["risk_mlo"], target, mask),
        }

    def get_primary_risk_head(self, outputs):
        """
        Returns the primary risk prediction (sigmoid of multi-view risk).

        Args:
            outputs: Dict of model outputs
        Returns:
            Tensor of predicted risk probabilities
        """
        return torch.sigmoid(outputs["risk_multi"])