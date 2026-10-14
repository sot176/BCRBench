import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# Spatial Transformer
# -------------------------

class SpatialTransformerBlock(nn.Module):
    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def _create_base_grid(self, B, H, W, device):
        """Create normalized base grid."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    def forward(self, x, flow):
        """
        Args:
            x:     [B, C, H, W]
            flow:  [B, 2, H, W] (dx, dy in pixel space)

        Returns:
            Warped tensor: [B, C, H, W]
        """
        B, _, H, W = x.shape
        device = x.device

        base_grid = self._create_base_grid(B, H, W, device)

        # Normalize flow to [-1, 1]
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0] = flow[:, 0] * 2 / (W - 1)  # dx
        flow_norm[:, 1] = flow[:, 1] * 2 / (H - 1)  # dy

        flow_norm = flow_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]

        sampling_grid = base_grid + flow_norm

        return F.grid_sample(
            x,
            sampling_grid,
            mode=self.mode,
            align_corners=True,
        )


# -------------------------
# Continuous Positional Encoding
# -------------------------

class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, max_time=5.0, num_steps=240):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.max_time = max_time
        self.num_steps = num_steps

        position = torch.linspace(0, max_time, steps=num_steps).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, times):
        """
        Args:
            x:     (N, B, C)
            times: (B,)
        """
        # Normalize to index space
        t = torch.clamp(times, 0, self.max_time)
        t = t * (self.num_steps - 1) / self.max_time

        t_floor = torch.floor(t).long()
        t_ceil = torch.ceil(t).long().clamp(max=self.num_steps - 1)

        alpha = (t - t_floor).unsqueeze(1)

        pe_floor = self.pe[t_floor]
        pe_ceil = self.pe[t_ceil]

        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil

        return self.dropout(x + pe_interp.unsqueeze(0))


# -------------------------
# Cumulative Probability Layer
# -------------------------

class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        super().__init__()

        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)

        # Register mask as buffer (not parameter!)
        mask = torch.tril(torch.ones(max_followup, max_followup))
        self.register_buffer("triangular_mask", mask.T)

    def hazards(self, x):
        return self.relu(self.hazard_fc(x))

    def forward(self, x):
        """
        Returns:
            cumulative logits: [B, T]
        """
        hazards = self.hazards(x)  # [B, T]
        B, T = hazards.shape

        expanded = hazards.unsqueeze(-1).expand(B, T, T)
        masked = expanded * self.triangular_mask

        cum_logits = masked.sum(dim=1) + self.base_hazard_fc(x)

        return cum_logits



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
