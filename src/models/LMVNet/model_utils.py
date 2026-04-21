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


class DropPath(nn.Module):
    """
    Stochastic Depth regularization per sample in residual paths.

    During training, randomly drops entire feature maps with probability `drop_prob`,
    scaling remaining features to maintain expected value. Disabled during evaluation.

    Reference: https://arxiv.org/abs/2002.05990 (Stochastic Depth)

    Args:
        drop_prob: Probability of dropping a sample [0.0, 1.0]
    """

    def __init__(self, drop_prob= 0.0):
        """
        Initialize DropPath module.

        Args:
            drop_prob: Dropout probability for stochastic depth
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Apply stochastic depth to input tensor.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor with stochastic depth applied (training), or unchanged tensor (eval)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape = (batch, 1, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FFN(nn.Module):
    """
    Position-wise Feed Forward Network (FFN) in Transformer architectures.

    Two-layer MLP with GELU activation and residual connections:
        dim → hidden_dim → dim

    Args:
        dim: Input and output feature dimension
        hidden_dim: Hidden layer dimension (typically 2-4x input)
        dropout: Dropout probability
    """

    def __init__(self, dim, hidden_dim, dropout = 0.0):
        """
        Initialize feed-forward network.

        Args:
            dim: Input/output feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor [*, dim]

        Returns:
            Output tensor [*, dim]
        """
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention fusion block for CC and MLO mammography views.

    Performs bidirectional multi-head self-attention and cross-attention between views,
    followed by position-wise feed-forward networks with residual connections and
    stochastic depth regularization.

    Architecture:
        For each view:
            1. Self-attention (view attends to itself)
            2. Cross-attention (view attends to other view)
            3. LayerNorm with residual connection
            4. FFN with residual connection and DropPath

    Args:
        in_channels: Input channel dimension (unused, for compatibility)
        reduced_channels: Feature dimension for attention operations
        heads: Number of attention heads
        dropout: Dropout probability
        drop_path: Stochastic depth probability
        ffn_expansion_factor: Hidden dimension expansion factor in FFN
    """

    def __init__(
        self,
        in_channels,
        reduced_channels,
        heads = 4,
        dropout = 0.3,
        drop_path = 0.2,
        ffn_expansion_factor = 4,
    ):
        """
        Initialize cross-attention block.

        Args:
            in_channels: Input channel dimension
            reduced_channels: Attention feature dimension
            heads: Number of attention heads
            dropout: Dropout probability
            drop_path: Stochastic depth probability
            ffn_expansion_factor: FFN hidden dimension multiplier
        """
        super().__init__()
        self.dim = reduced_channels
        self.num_heads = heads

        # Multi-head attention layers
        self.mha_self = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.mha_cross = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for each view and stage
        self.norm1_cc = nn.LayerNorm(self.dim)
        self.norm1_mlo = nn.LayerNorm(self.dim)
        self.norm2_cc = nn.LayerNorm(self.dim)
        self.norm2_mlo = nn.LayerNorm(self.dim)

        # Feed-forward networks per view
        hidden_dim = int(self.dim * ffn_expansion_factor)
        self.ffn_cc = FFN(self.dim, hidden_dim, dropout)
        self.ffn_mlo = FFN(self.dim, hidden_dim, dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0 else nn.Identity()
        )

    def forward(self, f_cc, f_mlo):
        """
        Forward pass with bidirectional cross-attention between views.

        Applies self-attention and cross-attention to each view, followed by
        layer normalization, feed-forward networks, and stochastic depth.

        Args:
            f_cc: CC view features [B, C, H, W]
            f_mlo: MLO view features [B, C, H, W]

        Returns:
            Tuple of fused view features:
                - f_cc_out: Updated CC features [B, C, H, W]
                - f_mlo_out: Updated MLO features [B, C, H, W]
        """
        B, C, H, W = f_cc.shape
        N = H * W

        # Step 1: Flatten spatial dimensions to sequence format [B, N, C]
        x_cc = f_cc.flatten(2).transpose(1, 2)
        x_mlo = f_mlo.flatten(2).transpose(1, 2)
        skip_cc, skip_mlo = x_cc, x_mlo

        # Step 2: Self and cross-attention with residual connections
        def attend(x, y):
            """
            Apply self-attention and cross-attention with dropout.

            Args:
                x: Query tensor [B, N, C]
                y: Key/Value tensor from other view [B, N, C]

            Returns:
                Attention output with dropout [B, N, C]
            """
            self_attn, _ = self.mha_self(x, x, x)
            cross_attn, _ = self.mha_cross(x, y, y)
            out = self.drop_path(self.proj_drop(self_attn + cross_attn))
            return out

        x_cc_post = self.norm1_cc(skip_cc + attend(x_cc, x_mlo))
        x_mlo_post = self.norm1_mlo(skip_mlo + attend(x_mlo, x_cc))

        # Step 3: Feed-forward networks with residual and DropPath
        x_cc_post = self.norm2_cc(x_cc_post + self.drop_path(self.ffn_cc(x_cc_post)))
        x_mlo_post = self.norm2_mlo(x_mlo_post + self.drop_path(self.ffn_mlo(x_mlo_post)))

        # Step 4: Reshape back to spatial format [B, C, H, W]
        f_cc_out = x_cc_post.transpose(1, 2).view(B, C, H, W)
        f_mlo_out = x_mlo_post.transpose(1, 2).view(B, C, H, W)
        return f_cc_out, f_mlo_out