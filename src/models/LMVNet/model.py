import logging
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from .model_utils import CrossAttentionBlock, LongitudinalFeatureProcessor
from models.common_parts import CumulativeProbabilityLayer, BaseRiskModel


class LMVNet(BaseRiskModel):
    """
    Longitudinal Multi-View Network (LMVNet) for breast cancer risk prediction.

    Processes current and previous mammography images (CC and MLO views),
    aligns features longitudinally using registration-based deformation,
    applies cross-attention mechanisms for view fusion, and outputs
    cumulative risk predictions for both individual views and combined multi-view.

    Architecture:
        1. Longitudinal Feature Processing: Warps previous images to align with current
        2. Cross-Attention Blocks: Fuses CC and MLO view features
        3. Multi-view Pooling and Fusion: Combines CC and MLO representations
        4. Risk Prediction Heads: Outputs risk estimates for each view and combined

    Args:
        mammo_reg_net: Registration network for image alignment (e.g., MammoRegNet)
        args: Configuration arguments containing model hyperparameters
    """

    def __init__(
        self,
        mammo_reg_net,
        args ):
        """
        Initialize LMVNet components.

        Args:
            mammo_reg_net: Pretrained registration network for longitudinal alignment
            args: Configuration namespace with attributes:
                  - feature_dim: Feature dimension for attention blocks
                  - num_heads: Number of attention heads
                  - num_attn_blocks: Number of cross-attention blocks
                  - dropout, drop_path: Regularization parameters
                  - ffn_expansion_factor: MLP expansion factor in attention
                  - max_followup: Maximum follow-up years for risk prediction
        """
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

    def forward(self, batch):
        """
        Forward pass of LMVNet with longitudinal alignment and cross-attention fusion.

        Processes longitudinal mammography pairs (current + previous) with CC and MLO views,
        performs feature alignment, applies cross-attention for view fusion, and outputs
        risk predictions at view and multi-view levels.

        Args:
            batch: Dictionary containing batch tensors with keys:
                   - "current_image_cc": Current CC view [B, 1, H, W]
                   - "previous_image_cc": Previous CC view [B, 1, H, W]
                   - "current_image_mlo": Current MLO view [B, 1, H, W]
                   - "previous_image_mlo": Previous MLO view [B, 1, H, W]
                   - "time_gap": Time between exams [B, 1]
                   - "target": Risk labels [B, num_years]
                   - "y_mask": Mask for valid label positions [B, num_years]

        Returns:
            Dictionary with risk prediction tensors:
                - "risk_multi": Multi-view fused risk logits [B, num_years]
                - "risk_cc": CC-only risk logits [B, num_years]
                - "risk_mlo": MLO-only risk logits [B, num_years]
        """
        # Step 1: Longitudinal feature extraction and alignment
        # Warps previous images to align with current using registration network
        longitudinal_output = self.longitudinal_feat_processor(
            batch["current_image_cc"],
            batch["previous_image_cc"],
            batch["current_image_mlo"],
            batch["previous_image_mlo"],
            batch["time_gap"],
        )

        f_cc = longitudinal_output["f_cc_long"]  # [B, C, H, W]
        f_mlo = longitudinal_output["f_mlo_long"]  # [B, C, H, W]

        # Step 2: Cross-attention fusion of CC and MLO views
        # Iteratively fuse features across views
        for attn_block in self.cross_attn_blocks:
            f_cc, f_mlo = attn_block(f_cc, f_mlo)

        # Step 3: Multi-view pooling and dimensionality reduction
        # Aggregate spatial features via global average pooling
        pooled_cc = self.global_avg_pool(f_cc).flatten(1)  # [B, C]
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1)  # [B, C]

        # Concatenate and fuse CC + MLO features
        multi_view_features = torch.cat([pooled_cc, pooled_mlo], dim=1)
        multi_view_features = self.view_fc(multi_view_features)

        # Step 4: Risk prediction via cumulative probability layers
        risk_multi = self.cumulative_risk(multi_view_features)
        risk_cc = self.cumulative_risk(pooled_cc)
        risk_mlo = self.cumulative_risk(pooled_mlo)

        return {
            "risk_multi": risk_multi,
            "risk_cc": risk_cc,
            "risk_mlo": risk_mlo,
        }

    def get_risk_heads(
        self, outputs, batch):
        """
        Extract risk heads for multi-task loss computation.

        Prepares model outputs paired with targets and masks for each view,
        enabling simultaneous loss computation across multi-view predictions.

        Args:
            outputs: Model outputs containing risk predictions for each view
            batch: Batch inputs containing targets and masks

        Returns:
            Dictionary mapping risk head name to (logits, target, mask) tuples:
                - "multi": Multi-view fused risk
                - "cc": CC-view-only risk
                - "mlo": MLO-view-only risk
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
        Extract primary risk prediction head for evaluation metrics.

        Returns the multi-view fused risk prediction transformed via sigmoid
        to produce probabilities in [0, 1] range for evaluation against binary labels.

        Args:
            outputs: Model outputs containing all risk predictions

        Returns:
            Tensor of primary risk probabilities (multi-view) [B, num_years]
        """
        return torch.sigmoid(outputs["risk_multi"])