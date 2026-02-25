import torch
import torch.nn as nn
import sys

from config.config import cfg
from asymmetry_model import extract_mirai_backbone
from models.common_parts  import ContinuousPosEncoding, SpatialTransformerBlock


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


class FFN(nn.Module):  # Defined outside for clarity, or can be an inner class
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Modern choice, or nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels, heads=4,
                 dropout=0.1, drop_path=0.1, ffn_expansion_factor=4):
        super().__init__()
        self.dim = reduced_channels
        self.num_heads = heads

        # MultiheadAttention for self and cross attention
        self.mha_self = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)
        self.mha_cross = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)

        # Dropout & DropPath
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # LayerNorms
        self.norm1_cc = nn.LayerNorm(self.dim)
        self.norm1_mlo = nn.LayerNorm(self.dim)
        self.norm2_cc = nn.LayerNorm(self.dim)
        self.norm2_mlo = nn.LayerNorm(self.dim)

        # FFN
        hidden_dim = int(self.dim * ffn_expansion_factor)
        self.ffn_cc = FFN(self.dim, hidden_dim, dropout)
        self.ffn_mlo = FFN(self.dim, hidden_dim, dropout)

    def forward(self, f_cc, f_mlo):
        B, C, H, W = f_cc.shape
        N = H * W

        # Flatten spatial dimensions: [B, C, H, W] -> [B, N, C]
        x_cc = f_cc.flatten(2).transpose(1, 2)
        x_mlo = f_mlo.flatten(2).transpose(1, 2)
        skip_cc, skip_mlo = x_cc, x_mlo

        # --- CC view attention ---
        out_cc_self, _ = self.mha_self(x_cc, x_cc, x_cc)
        out_cc_cross, _ = self.mha_cross(x_cc, x_mlo, x_mlo)
        out_cc = out_cc_self + out_cc_cross
        out_cc = self.drop_path(self.proj_drop(out_cc))
        x_cc_post_attn = self.norm1_cc(skip_cc + out_cc)

        # --- MLO view attention ---
        out_mlo_self, _ = self.mha_self(x_mlo, x_mlo, x_mlo)
        out_mlo_cross, _ = self.mha_cross(x_mlo, x_cc, x_cc)
        out_mlo = out_mlo_self + out_mlo_cross
        out_mlo = self.drop_path(self.proj_drop(out_mlo))
        x_mlo_post_attn = self.norm1_mlo(skip_mlo + out_mlo)

        # --- FFN ---
        ffn_cc_out = self.ffn_cc(x_cc_post_attn)
        ffn_mlo_out = self.ffn_mlo(x_mlo_post_attn)

        out_cc = self.norm2_cc(x_cc_post_attn + self.drop_path(ffn_cc_out))
        out_mlo = self.norm2_mlo(x_mlo_post_attn + self.drop_path(ffn_mlo_out))

        # Reshape back to [B, C, H, W]
        out_cc = out_cc.transpose(1, 2).view(B, C, H, W)
        out_mlo = out_mlo.transpose(1, 2).view(B, C, H, W)

        return out_cc, out_mlo


