from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.common_parts import extract_mirai_backbone, ContinuousPosEncoding, SpatialTransformerBlock


class LongitudinalFeatureProcessor(nn.Module):
    """
    Processes current and prior mammogram views (CC and MLO) with longitudinal alignment.

    Pipeline:
        1. Extracts spatial features via pretrained encoder for each view
        2. Estimates deformation field using registration network
        3. Aligns prior features to current via learned spatial transformation
        4. Computes temporal difference features with positional encoding
        5. Concatenates current, prior, and difference features for multi-modal input

    Args:
        mammo_reg_net: Registration network for deformation field estimation
        args: Configuration with attributes:
              - pos_encoding_dim: Dimension for positional encoding of time gaps
              - finetune_all: Whether to fine-tune encoder (default: frozen)
    """

    def __init__(self, mammo_reg_net, args):
        """
        Initialize longitudinal feature processor with encoder and registration components.

        Args:
            mammo_reg_net: Pretrained registration network (frozen during initialization)
            args: Configuration namespace with pos_encoding_dim and finetune_all attributes
        """
        super().__init__()
        self.encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])
        self.mammo_reg_net = mammo_reg_net.eval()  # frozen by default
        self.feat_transformer = SpatialTransformerBlock(mode="bilinear")
        self.positional_encoding = ContinuousPosEncoding(dim=args.pos_encoding_dim)

        # Freeze encoder by default
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if args.finetune_all:
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder.train()

    @staticmethod
    def _to_3ch(x):
        """
        Convert grayscale images to 3-channel format for encoder compatibility.

        Args:
            x: Grayscale tensor [B, 1, H, W]

        Returns:
            3-channel expanded tensor [B, 3, H, W] (channels repeated)
        """
        return x.expand(-1, 3, -1, -1)

    def _process_view(
        self, img_cur, img_pri, time_gap):
        """
        Process single mammography view with longitudinal alignment and temporal encoding.

        Extracts features from current and prior images, registers prior to current,
        computes temporal differences with time-aware positional encoding, and concatenates.

        Args:
            img_cur: Current mammogram [B, 1, H, W]
            img_pri: Prior mammogram [B, 1, H, W]
            time_gap: Time between exams [B, 1] (in years or years normalized)

        Returns:
            Longitudinal features concatenating current, prior, and difference [B, 3*C, H, W]
        """
        # Convert to 3-channel
        f_cur = self.encoder(self._to_3ch(img_cur))
        f_pri = self.encoder(self._to_3ch(img_pri))

        # Step 1: Register prior to current - obtain deformation field
        registration_outputs = self.mammo_reg_net(img_cur, img_pri)
        deformation_field = registration_outputs[1]
        deformation_field = self._resize_flow(deformation_field, f_cur.shape, img_cur.shape)
        f_pri_aligned = self.feat_transformer(f_pri, deformation_field)

        # Step 2: Compute temporal difference with time-aware positional encoding
        f_diff = torch.abs(f_cur - f_pri_aligned)
        B, C, H, W = f_diff.shape
        f_diff_flat = f_diff.flatten(2).permute(2, 0, 1)  # [N, B, C]
        f_diff_encoded = self.positional_encoding(f_diff_flat, time_gap)
        f_diff = f_diff_encoded.permute(1, 2, 0).view(B, C, H, W)

        # Step 3: Concatenate current, prior aligned, and temporal difference features
        f_long = torch.cat([f_cur, f_pri, f_diff], dim=1)  # [B, 3*C, H, W]
        return f_long

    @staticmethod
    def _resize_flow(flow, target_shape, src_shape):
        """
        Resize and rescale deformation field to match feature map resolution.

        Interpolates flow field from image resolution to feature resolution and
        rescales flow values proportionally to the resolution change.

        Args:
            flow: Deformation field from registration network [B, 2, Hi, Wi]
            target_shape: Target feature shape [B, C, Hf, Wf]
            src_shape: Source image shape [B, 1, Hi, Wi]

        Returns:
            Rescaled deformation field [B, 2, Hf, Wf]
        """
        B, C, Hf, Wf = target_shape
        _, _, Hi, Wi = src_shape

        flow_resized = F.interpolate(
            flow.detach(), size=(Hf, Wf), mode="bilinear", align_corners=True
        )
        flow_resized[:, 0] *= Wf / Wi
        flow_resized[:, 1] *= Hf / Hi
        return flow_resized

    def forward(self,img_cur_cc,img_pri_cc,img_cur_mlo,img_pri_mlo,time_gap):
        """
        Process both CC and MLO views with longitudinal alignment.

        Args:
            img_cur_cc: Current CC view [B, 1, H, W]
            img_pri_cc: Prior CC view [B, 1, H, W]
            img_cur_mlo: Current MLO view [B, 1, H, W]
            img_pri_mlo: Prior MLO view [B, 1, H, W]
            time_gap: Time between exams [B, 1]

        Returns:
            Dictionary with processed longitudinal features:
                - "f_cc_long": CC view features [B, 3*C, H, W]
                - "f_mlo_long": MLO view features [B, 3*C, H, W]
        """
        f_cc_long = self._process_view(img_cur_cc, img_pri_cc, time_gap)
        f_mlo_long = self._process_view(img_cur_mlo, img_pri_mlo, time_gap)
        return {"f_cc_long": f_cc_long, "f_mlo_long": f_mlo_long}

