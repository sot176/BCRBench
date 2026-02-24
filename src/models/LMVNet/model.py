import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from config.config import cfg
from asymmetry_model import extract_mirai_backbone
from .model_utils import CrossAttentionBlock
from models.common_parts.model_utils  import ContinuousPosEncoding, SpatialTransformerBlock, CumulativeProbabilityLayer

class LongitudinalFeatureProcessor(nn.Module):
    """
    Implements Steps 1-4 of the longitudinal risk prediction pipeline.
    This module extracts, aligns, subtracts, and concatenates features from
    current and prior mammogram views.
    """

    def __init__(self, mammo_reg_net: nn.Module,  finetune_all: bool = False):
        super().__init__()
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )
        self.mammo_reg_net = mammo_reg_net

        # The block responsible for applying the deformation field to feature maps
        self.feat_transformer = SpatialTransformerBlock(mode='bilinear')

        self.positional_encoding = ContinuousPosEncoding(dim=512)

        # It's good practice to freeze models that are not being trained
        self.encoder.requires_grad_(False)
        self.mammo_reg_net.requires_grad_(False)
        self.encoder.eval()
        self.mammo_reg_net.eval()

        # Unfreeze if finetuning is enabled
        if finetune_all:
            print("Finetuning all layers of the encoder")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.encoder.train()

        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _process_view(self, img_cur: torch.Tensor, img_pri: torch.Tensor, time_gap):
        """
        Helper function to run the feature processing pipeline on a single view (CC or MLO).

        Args:
            img_cur (Tensor): Current image (B, 1, H, W).
            img_pri (Tensor): Prior image (B, 1, H, W).

        Returns:
            f_long (Tensor): The concatenated longitudinal feature tensor.
        """
        # Ensure images have 3 channels for pre-trained encoders

        img_cur_3c = img_cur.repeat(1, 3, 1, 1)
        img_pri_3c = img_pri.repeat(1, 3, 1, 1)

        # --- Step 1: Feature extraction ---
        f_cur = self.encoder(img_cur_3c)
        f_pri = self.encoder(img_pri_3c)

        # --- Step 2: Temporal Feature Alignment ---
        # Get deformation field from the registration network using original images
        registration_outputs = self.mammo_reg_net(img_cur, img_pri)  # MammoRegNet may take B,1,H,W
        deformation_field = registration_outputs[1]
        # Downsample deformation field to match the feature map's resolution
        deformation_field_downsampled = F.interpolate(
            deformation_field.detach(),  # Detach to prevent gradients from flowing into RegNet
            size=(f_cur.shape[2], f_cur.shape[3]),
            mode='bilinear',
            align_corners=True
        )

        # Rescale the displacement values in the deformation field
        scaling_factor_y = f_cur.shape[2] / img_cur.shape[2]
        scaling_factor_x = f_cur.shape[3] / img_cur.shape[3]

        deformation_field_downsampled[:, 0, :, :] *= scaling_factor_x  # x-displacements
        deformation_field_downsampled[:, 1, :, :] *= scaling_factor_y  # y-displacements

        # Apply the alignment to the prior feature map
        f_pri_aligned = self.feat_transformer(f_pri, deformation_field_downsampled)

        # --- Step 3: Temporal Subtraction ---
        f_diff = torch.abs(f_cur - f_pri_aligned)
        B, C, H, W = f_diff.shape
        # Apply positional encoding to the difference map
        flattened_feats = f_diff.flatten(start_dim=2).permute(2, 0, 1)  # [N, B, C]
        fdif_with_time = self.positional_encoding(flattened_feats, time_gap)
        f_diff = fdif_with_time.permute(1, 2, 0).view(B, C, H, W)

        # --- Step 4: Concatenation ---
        f_long = torch.cat([f_cur, f_pri, f_diff], dim=1)  # [B, 1536, H, W]
        return f_long

    def forward(self, img_cur_cc, img_pri_cc, img_cur_mlo, img_pri_mlo, time_gap):
        """
        Main forward pass to process both CC and MLO views.
        """

        f_cc_long = self._process_view(img_cur_cc, img_pri_cc, time_gap)
        f_mlo_long = self._process_view(img_cur_mlo, img_pri_mlo, time_gap)

        return {
            'f_cc_long': f_cc_long,
            'f_mlo_long': f_mlo_long
        }


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