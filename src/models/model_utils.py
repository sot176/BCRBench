import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialTransformerBlock(nn.Module):
    def __init__(self, mode="bilinear"):
        """
        Applies a deformation field to an input tensor using grid sampling.

        Args:
            mode (str): Interpolation mode to use ('bilinear' or 'nearest')
        """
        super().__init__()
        self.mode = mode

    def forward(self, f_pri, deformation_field):
        """
        Args:
            f_pri (Tensor): Prior feature map of shape [B, C, H, W]
            deformation_field (Tensor): Flow field of shape [B, 2, H, W] (dx, dy)

        Returns:
            Tensor: Warped feature map of shape [B, C, H, W]
        """
        B, _, H, W = f_pri.shape

        # Generate identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=f_pri.device),
            torch.arange(W, device=f_pri.device),
            indexing="ij"
        )
        grid = torch.stack((grid_y, grid_x), dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)           # [B, 2, H, W]

        # Add deformation
        new_grid = grid + deformation_field.to(f_pri.device)  # [B, 2, H, W]

        # Normalize to [-1, 1]
        new_grid[:, 0, :, :] = 2.0 * (new_grid[:, 0, :, :] / (H - 1) - 0.5)
        new_grid[:, 1, :, :] = 2.0 * (new_grid[:, 1, :, :] / (W - 1) - 0.5)

        # Reshape to [B, H, W, 2] and flip last dim to match grid_sample format
        new_grid = new_grid.permute(0, 2, 3, 1)[..., [1, 0]]  # [B, H, W, 2]

        # Apply spatial transformation
        f_pri_aligned = F.grid_sample(
            f_pri, new_grid, mode=self.mode, align_corners=True
        )

        return f_pri_aligned



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



class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=5, num_steps=240):
        """
        Continuous sinusoidal positional encoding with linear interpolation over time.

        Args:
            dim (int): Dimension of the encoding.
            drop (float): Dropout rate.
            maxtime (float): Maximum time value for normalization.
            num_steps (int): Number of discrete time steps for encoding table.
        """
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.maxtime = maxtime
        self.num_steps = num_steps

        # Precompute sinusoidal encodings
        position = torch.linspace(0, maxtime, steps=num_steps).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        """
        Args:
            xs (Tensor): Input tensor of shape (N, B, C).
            times (Tensor): Time values of shape (B,).

        Returns:
            Tensor: Time-encoded input of shape (N, B, C).
        """
        times = torch.clamp(times, 0, self.maxtime) * (self.num_steps - 1) / self.maxtime
        t_floor = torch.floor(times).long()
        t_ceil = torch.ceil(times).long()
        alpha = (times - t_floor).unsqueeze(1)  # (B, 1)

        # Linear interpolation
        pe_floor = self.pe[t_floor]  # (B, C)
        pe_ceil = self.pe[t_ceil]    # (B, C)
        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil  # (B, C)

        return self.dropout(xs + pe_interp.unsqueeze(0))  # (N, B, C)


class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(CumulativeProbabilityLayer, self).__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)  # could also be (num_features → max_followup)
        self.relu = nn.ReLU(inplace=True)

        # proper lower-triangular mask
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        """
        Returns:
            hazards: [B, T] probabilities per year
        """
        hazards = self.hazards(x)  # [B, T] logits
        B, T = hazards.size()

        expanded = hazards.unsqueeze(-1).expand(B, T, T)  # [B, T, T]
        masked = expanded *self.upper_triagular_mask  # [B, T, T]

        cum_logits = torch.sum(masked, dim=1) + self.base_hazard_fc(x)  # [B, T]
        return cum_logits

