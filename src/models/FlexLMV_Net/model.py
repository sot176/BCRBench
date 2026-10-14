
from __future__ import annotations

import torch
import torch.nn as nn

from models.common_parts import (
    BaseRiskModel,
    CumulativeProbabilityLayer,
)
from config.config import cfg


# Helpers specific to this multiple-prior model.
from .model_utils import LongitudinalFeatureProcessorMultiplePriors, CrossAttentionBlock
from models.common_parts import CrossAttentionBlock

class FlexLMVNet(BaseRiskModel):
    """
    Longitudinal multi-view breast cancer risk prediction.

    Pipeline:
        1. Extract current and prior Mirai feature maps.
        2. Align prior features to the current examination.
        3. Aggregate prior and difference features temporally.
        4. Fuse CC and MLO using cross-attention.
        5. Predict cumulative risks for CC, MLO, and fused features.

    Required configuration:
        max_followup: Number of follow-up years.
        num_prior_img: Maximum number of prior examinations.
        feature_dim: Mirai feature-map channels (default 512).
        num_heads: Cross-attention heads.
        num_attn_blocks: Number of cross-attention blocks.
        dropout: Attention dropout.
        drop_path: Stochastic depth probability.
        ffn_expansion_factor: Attention FFN expansion factor.
        aggregation_mode: Temporal aggregation mode ("gated").
        aggregator_hidden_dim: Hidden dimension of temporal scoring.
        finetune: Whether to fine-tune the Mirai encoder.
        encoder_path: Path to the pretrained Mirai backbone.
    """

    def __init__(self, mammo_reg_net, args):
        super().__init__(args)

        feature_channels = getattr(args, "feature_dim", 512)
        self.num_prior_img = getattr(args, "num_prior_img", 2)

        # Current + aggregated prior + aggregated difference features.
        longitudinal_channels = feature_channels * 3

        self.longitudinal_feat_processor = (
            LongitudinalFeatureProcessorMultiplePriors(
                mammo_reg_net=mammo_reg_net,
                finetune=getattr(args, "finetune", False),
                aggregation_mode=getattr(
                    args, "aggregation_mode", "gated"
                ),
                num_prior_img=self.num_prior_img,
                aggregator_hidden_dim=getattr(
                    args, "aggregator_hidden_dim", 64
                ),
                encoder_path=getattr(
                    args,
                    "encoder_path",
                    cfg["paths"]["mirai_path"],
                ),
                feature_channels=feature_channels,
                dropout=getattr(args, "dropout", 0.4),
            )
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                in_channels=longitudinal_channels,
                reduced_channels=longitudinal_channels,
                heads=getattr(args, "num_heads", 4),
                dropout=getattr(args, "dropout", 0.4),
                drop_path=getattr(args, "drop_path", 0.2),
                ffn_expansion_factor=getattr(
                    args, "ffn_expansion_factor", 2
                ),
            )
            for _ in range(getattr(args, "num_attn_blocks", 1))
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.view_fc = nn.Linear(
            longitudinal_channels * 2,
            longitudinal_channels,
        )

        self.cumulative_risk = CumulativeProbabilityLayer(
            num_features=longitudinal_channels,
            max_followup=getattr(args, "max_followup", 5),
        )

    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                current_image_cc
                previous_images_cc
                current_image_mlo
                previous_images_mlo
                time_gap

        Returns:
            risk_multi, risk_cc, risk_mlo
        """
        longitudinal_output = self.longitudinal_feat_processor(
            batch["current_image_cc"],
            batch["previous_images_cc"],
            batch["current_image_mlo"],
            batch["previous_images_mlo"],
            batch["time_gap"],
        )

        f_cc = longitudinal_output["f_cc_long"]
        f_mlo = longitudinal_output["f_mlo_long"]

        for block in self.cross_attn_blocks:
            f_cc, f_mlo = block(f_cc, f_mlo)

        pooled_cc = self.global_avg_pool(f_cc).flatten(1)
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1)

        pooled_multi = self.view_fc(
            torch.cat([pooled_cc, pooled_mlo], dim=1)
        )

        return {
            "risk_multi": self.cumulative_risk(pooled_multi),
            "risk_cc": self.cumulative_risk(pooled_cc),
            "risk_mlo": self.cumulative_risk(pooled_mlo),
        }

    def get_risk_heads(self, outputs, batch):
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "multi": (outputs["risk_multi"], target, mask),
            "cc": (outputs["risk_cc"], target, mask),
            "mlo": (outputs["risk_mlo"], target, mask),
        }

    def get_primary_risk_head(self, outputs):
        return torch.sigmoid(outputs["risk_multi"])
