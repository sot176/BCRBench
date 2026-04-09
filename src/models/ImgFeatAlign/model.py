import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from models.common_parts import extract_mirai_backbone, SpatialTransformerBlock
from .model_utils import RiskModelWithAttention


class ImgFeatAlign(nn.Module):
    """
    Feature-level alignment using deformation fields for longitudinal risk prediction.
    """

    def __init__(self, mammo_reg_net: nn.Module, args):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])

        self._set_encoder_trainable(args.finetune_all)

        # -------------------------
        # Modules
        # -------------------------
        self.risk_model = RiskModelWithAttention(args)
        self.spatial_transform = SpatialTransformerBlock(mode="bilinear")

        # Registration network (frozen)
        self.mammo_reg_net = mammo_reg_net.eval()
        self.mammo_reg_net.requires_grad_(False)

    # -------------------------
    # Helpers
    # -------------------------

    def _set_encoder_trainable(self, finetune: bool):
        """Enable/disable encoder training."""
        for p in self.encoder.parameters():
            p.requires_grad = finetune

        self.encoder.train(mode=finetune)

        if finetune:
            print("Finetuning encoder")
        else:
            print("Freezing encoder")

    @staticmethod
    def _to_3ch(x):
        """Convert (B,1,H,W) → (B,3,H,W)"""
        return x.expand(-1, 3, -1, -1)

    @staticmethod
    def _resize_flow(flow, target_shape, src_shape):
        """
        Resize and rescale deformation field to match feature resolution.
        """
        B, _, Hf, Wf = target_shape
        _, _, Hi, Wi = src_shape

        flow = F.interpolate(
            flow,
            size=(Hf, Wf),
            mode="bilinear",
            align_corners=True,
        )

        scale_x = Wf / Wi
        scale_y = Hf / Hi

        flow = flow.clone()  # avoid in-place ops on shared tensors
        flow[:, 0] *= scale_x
        flow[:, 1] *= scale_y

        return flow

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - current_image: (B,1,H,W)
                - previous_image: (B,1,H,W)
                - time_gap: (B,)
        """

        img_cur = batch["current_image"]
        img_pri = batch["previous_image"]
        time_gap = batch["time_gap"]

        # -------------------------
        # 1. Feature extraction
        # -------------------------
        f_cur = self.encoder(self._to_3ch(img_cur))
        f_pri = self.encoder(self._to_3ch(img_pri))

        # -------------------------
        # 2. Registration
        # -------------------------
        registration_outputs = self.mammo_reg_net(img_cur, img_pri)  # MammoRegNet may take B,1,H,W
        flow = registration_outputs[1]

        flow = flow.detach()
        flow_resized = self._resize_flow(
            flow,
            target_shape=f_cur.shape,
            src_shape=img_cur.shape,
        )

        # -------------------------
        # 3. Feature alignment
        # -------------------------
        f_pri_aligned = self.spatial_transform(f_pri, flow_resized)

        # -------------------------
        # 4. Temporal difference
        # -------------------------
        f_diff = torch.abs(f_cur - f_pri_aligned)

        # -------------------------
        # 5. Risk prediction
        # -------------------------
        risk_outputs = self.risk_model(
            f_cur,
            f_pri,
            f_pri_aligned,
            f_diff,
            time_gap,
        )

        return {
            "risk_prediction": risk_outputs,
            "deformation_field": flow,
            "aligned_prior_feature": f_pri_aligned,
            "prior_feature": f_pri,
            "current_feature": f_cur,
            "diff_feature": f_diff,
        }

    # -------------------------
    # Heads
    # -------------------------

    def get_risk_heads(self, outputs, batch):
        risk = outputs["risk_prediction"]

        return {
            "fused": (risk["pred_fused"], batch["target"], batch["y_mask"]),
            "cur": (risk["pred_cur"], batch["target"], batch["y_mask"]),
            "pri": (
                risk["pred_pri"],
                batch["target_prior"],
                batch["y_mask_prior"],
            ),
        }

    def get_primary_risk_head(self, outputs):
        logits = outputs["risk_prediction"]["pred_fused"]
        return torch.sigmoid(logits)