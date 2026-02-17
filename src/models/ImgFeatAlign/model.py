import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from config.config import cfg
from mirai_localized_dif_head import extract_mirai_backbone
from model_utils import SpatialTransformerBlock, RiskModelWithAttention


class ImgFeatAlign(nn.Module):
    """
    Combines downsampled deformation field applied to feature maps for risk prediction.
    """
    def __init__(self):
        super().__init__()
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )
        self.encoder.requires_grad = False
        self.risk_prediction_model = RiskModelWithAttention()
        self.feat_transformer = SpatialTransformerBlock(mode='bilinear')

    def forward(self, img_cur, img_pri, warped_pri_img, deformation_field, time_gap):
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
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        # Resize deformation field to match feature map resolution
        deformation_field_downsampled = F.interpolate(
            deformation_field.detach().cpu(),
            size=(fcur.shape[2], fcur.shape[3]),
            mode='bilinear',
            align_corners=True
        ).to(fpri.device)

        scaling_factor_y = fcur.shape[2] / img_cur.shape[2]
        scaling_factor_x = fcur.shape[3] / img_cur.shape[3]

        deformation_field_downsampled[:, 0, :, :] *= scaling_factor_x  # x-direction
        deformation_field_downsampled[:, 1, :, :] *= scaling_factor_y  # y-direction

        fpri_aligned = self.feat_transformer(fpri, deformation_field_downsampled)
        fdiff = torch.abs(fcur - fpri_aligned)

        return {
            'risk_prediction': self.risk_prediction_model(fcur, fpri, fpri_aligned, fdiff, time_gap),
            'deformation_field': deformation_field,
            'aligned_prior_feature': fpri_aligned,
            'prior_feature_before_alignment': fpri,
            'current_feature': fcur,
            'diff_feature': fdiff,
        }
