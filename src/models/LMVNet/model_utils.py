import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config.config import cfg
from models.common_parts import extract_mirai_backbone, ContinuousPosEncoding, SpatialTransformerBlock


class LongitudinalFeatureProcessor(nn.Module):
    """
    Processes current and prior mammogram views (CC and MLO):
    - Extracts features via pretrained encoder
    - Aligns prior features to current via deformation field
    - Computes temporal differences
    - Concatenates current, prior, and difference features
    """
    def __init__(self, mammo_reg_net: nn.Module, finetune_all: bool = False):
        super().__init__()
        self.encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])
        self.mammo_reg_net = mammo_reg_net.eval()  # frozen by default
        self.feat_transformer = SpatialTransformerBlock(mode='bilinear')
        self.positional_encoding = ContinuousPosEncoding(dim=512)

        # Freeze encoder by default
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if finetune_all:
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder.train()

    @staticmethod
    def _to_3ch(x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale (B,1,H,W) → 3-channel (B,3,H,W)."""
        return x.expand(-1, 3, -1, -1)

    def _process_view(self, img_cur: torch.Tensor, img_pri: torch.Tensor, time_gap: torch.Tensor) -> torch.Tensor:
        """Process one view (CC or MLO) and return longitudinal features."""
        # Convert to 3-channel
        f_cur = self.encoder(self._to_3ch(img_cur))
        f_pri = self.encoder(self._to_3ch(img_pri))

        # --- Alignment ---
        registration_outputs = self.mammo_reg_net(img_cur, img_pri)   
        deformation_field = registration_outputs[1]
        deformation_field = self._resize_flow(deformation_field, f_cur.shape, img_cur.shape)
        f_pri_aligned = self.feat_transformer(f_pri, deformation_field)

        # --- Temporal difference ---
        f_diff = torch.abs(f_cur - f_pri_aligned)
        B, C, H, W = f_diff.shape
        f_diff_flat = f_diff.flatten(2).permute(2, 0, 1)  # [N, B, C]
        f_diff_encoded = self.positional_encoding(f_diff_flat, time_gap)
        f_diff = f_diff_encoded.permute(1, 2, 0).view(B, C, H, W)

        # --- Concatenate features ---
        f_long = torch.cat([f_cur, f_pri, f_diff], dim=1)  # [B, 3*C, H, W]
        return f_long

    @staticmethod
    def _resize_flow(flow: torch.Tensor, target_shape, src_shape) -> torch.Tensor:
        """Resizes and rescales deformation field to match feature map resolution."""
        B, C, Hf, Wf = target_shape
        _, _, Hi, Wi = src_shape

        flow_resized = F.interpolate(flow.detach(), size=(Hf, Wf), mode='bilinear', align_corners=True)
        flow_resized[:, 0] *= Wf / Wi
        flow_resized[:, 1] *= Hf / Hi
        return flow_resized

    def forward(self, img_cur_cc, img_pri_cc, img_cur_mlo, img_pri_mlo, time_gap) -> Dict[str, torch.Tensor]:
        f_cc_long = self._process_view(img_cur_cc, img_pri_cc, time_gap)
        f_mlo_long = self._process_view(img_cur_mlo, img_pri_mlo, time_gap)
        return {"f_cc_long": f_cc_long, "f_mlo_long": f_mlo_long}


# -------------------------
# DropPath
# -------------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape = (batch, 1, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------------
# Feedforward Network
# -------------------------
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Cross-Attention Block
# -------------------------
class CrossAttentionBlock(nn.Module):
    """Cross-attention between CC and MLO views with residual FFN and drop path."""
    def __init__(self, in_channels, reduced_channels, heads=4,
                 dropout=0.1, drop_path=0.1, ffn_expansion_factor=4):
        super().__init__()
        self.dim = reduced_channels
        self.num_heads = heads

        # MultiheadAttention
        self.mha_self = nn.MultiheadAttention(embed_dim=self.dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.mha_cross = nn.MultiheadAttention(embed_dim=self.dim, num_heads=heads, dropout=dropout, batch_first=True)

        # LayerNorm & FFN
        self.norm1_cc = nn.LayerNorm(self.dim)
        self.norm1_mlo = nn.LayerNorm(self.dim)
        self.norm2_cc = nn.LayerNorm(self.dim)
        self.norm2_mlo = nn.LayerNorm(self.dim)
        hidden_dim = int(self.dim * ffn_expansion_factor)
        self.ffn_cc = FFN(self.dim, hidden_dim, dropout)
        self.ffn_mlo = FFN(self.dim, hidden_dim, dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, f_cc: torch.Tensor, f_mlo: torch.Tensor):
        B, C, H, W = f_cc.shape
        N = H * W

        # Flatten spatial dims
        x_cc = f_cc.flatten(2).transpose(1, 2)
        x_mlo = f_mlo.flatten(2).transpose(1, 2)
        skip_cc, skip_mlo = x_cc, x_mlo

        # --- Attention ---
        def attend(x, y):
            self_attn, _ = self.mha_self(x, x, x)
            cross_attn, _ = self.mha_cross(x, y, y)
            out = self.drop_path(self.proj_drop(self_attn + cross_attn))
            return out

        x_cc_post = self.norm1_cc(skip_cc + attend(x_cc, x_mlo))
        x_mlo_post = self.norm1_mlo(skip_mlo + attend(x_mlo, x_cc))

        # --- FFN ---
        x_cc_post = self.norm2_cc(x_cc_post + self.drop_path(self.ffn_cc(x_cc_post)))
        x_mlo_post = self.norm2_mlo(x_mlo_post + self.drop_path(self.ffn_mlo(x_mlo_post)))

        # Reshape back to [B, C, H, W]
        f_cc_out = x_cc_post.transpose(1, 2).view(B, C, H, W)
        f_mlo_out = x_mlo_post.transpose(1, 2).view(B, C, H, W)
        return f_cc_out, f_mlo_out