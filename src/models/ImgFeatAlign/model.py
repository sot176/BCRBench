import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from config.config import cfg
from models.common_parts import extract_mirai_backbone
from .model_utils import  RiskModelWithAttention
from models.common_parts import SpatialTransformerBlock


class ImgFeatAlign(nn.Module):
    """
    Combines downsampled deformation field applied to feature maps for risk prediction.
    """
    def __init__(self, mammo_reg_net: nn.Module, finetune_all: bool = False):
        super().__init__()
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )
        if finetune_all:
            print("Finetuning all layers of the encoder")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.encoder.train()

        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.risk_prediction_model = RiskModelWithAttention()
        self.feat_transformer = SpatialTransformerBlock(mode='bilinear')

        self.mammo_reg_net = mammo_reg_net
        self.mammo_reg_net.requires_grad_(False)
        self.mammo_reg_net.eval()

    def forward(self, batch):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            warped_pri_img (Tensor): Warped prior image after registration (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)
            deformation_field (Tensor): Deformation field used for warping (B, 2, H, W)

        Returns:
            dict: Risk prediction and related intermediate features
        """
        img_cur = batch["current_image"]
        img_pri = batch["previous_image"]
        time_gap = batch["time_gap"]

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

        return {
            'risk_prediction': self.risk_prediction_model(f_cur, f_pri, f_pri_aligned, f_diff, time_gap),
            'deformation_field': deformation_field,
            'aligned_prior_feature': f_pri_aligned,
            'prior_feature_before_alignment': f_pri,
            'current_feature': f_cur,
            'diff_feature': f_diff,
        }

    def get_risk_heads(self, outputs, batch):
        risk_pred = outputs["risk_prediction"]

        return {
            "fused": (
                risk_pred["pred_fused"],
                batch["target"],
                batch["y_mask"],
            ),
            "cur": (
                risk_pred["pred_cur"],
                batch["target"],
                batch["y_mask"],
            ),
            "pri": (
                risk_pred["pred_pri"],
                batch["target_prior"],
                batch["y_mask_prior"],
            ),
        }

    def get_primary_risk_head(self, outputs):
        return outputs["risk_prediction"]["pred_fused"]
    