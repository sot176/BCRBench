import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        """
        Predict cumulative cancer probabilities via time-dependent hazard estimation.

        Args:
            num_features (int): Feature size from the model.
            max_followup (int): Number of follow-up years (prediction steps).
        """
        super().__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)

        # Lower-triangular mask (T x T)
        mask = torch.tril(torch.ones(max_followup, max_followup)).T
        self.register_buffer("upper_triagular_mask", mask)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features of shape (B, C)

        Returns:
            Tensor: Cumulative probability over time (B, T)
        """
        B = x.size(0)
        raw_hazards = self.relu(self.hazard_fc(x))  # (B, T)
        base_hazard = self.base_hazard_fc(x)        # (B, 1)

        expanded = raw_hazards.unsqueeze(-1).expand(B, -1, raw_hazards.size(1))  # (B, T, T)
        masked = expanded * self.upper_triagular_mask  # (B, T, T)

        cum_probs = masked.sum(dim=1) + base_hazard  # (B, T)
        return cum_probs


class TemporalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Multi-head self-attention with residual connection and feedforward
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        return self.norm(x + self.ff(x))


class RiskModelWithAttention(nn.Module):
    def __init__(self, num_years=5, time_encoding_dim=512):
        super().__init__()
        self.positional_encoding = ContinuousPosEncoding(dim=time_encoding_dim)
        self.attention_layer = TemporalAttentionLayer(dim=512, num_heads=8)
        self.feature_projection = nn.Linear(512, 512)

        # Prediction heads
        self.cumulative_prob_layer_fused = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_cur = CumulativeProbabilityLayer(512, num_years)
        self.cumulative_prob_layer_pri = CumulativeProbabilityLayer(512, num_years)

    def forward(self, f_cur, f_pri, f_pri_aligned, f_dif, time_gap):
        """
        Args:
            f_cur: Current image features (B, C, H, W)
            f_pri: Prior image features (B, C, H, W)
            f_pri_aligned: Spatially aligned prior features (B, C, H, W)
            f_dif: Difference map (B, C, H, W)
            time_gap: Time gap (tensor) (B, 1)
        """
        B, C, H, W = f_dif.shape

        # Time-aware encoding of difference map
        #flattened_feats = f_dif.flatten(start_dim=2).permute(2, 0, 1)  # [N, B, C]
        #fdif_with_time = self.positional_encoding(flattened_feats, time_gap)
        #fdif_with_time = fdif_with_time.permute(1, 2, 0).view(B, C, H, W)

        # Global average pooling
        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, (1, 1)).view(B, C)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, (1, 1)).view(B, C)
        f_pri_aligned_pooled = F.adaptive_avg_pool2d(f_pri_aligned, (1, 1)).view(B, C)
        #fdif_pooled = F.adaptive_avg_pool2d(fdif_with_time, (1, 1)).view(B, C)

        # Temporal attention
        stacked = torch.stack([f_pri_aligned_pooled, f_cur_pooled], dim=1)  # [B, 3, C]

        #stacked = torch.stack([f_pri_aligned_pooled, fdif_pooled, f_cur_pooled], dim=1)  # [B, 3, C]
        attended = self.attention_layer(stacked.permute(1, 0, 2))  # [3, B, C]
        attended = attended.permute(1, 0, 2).mean(dim=1)  # [B, C]
        fused_feat = self.feature_projection(attended)

        return {
            "pred_fused": self.cumulative_prob_layer_fused(fused_feat),
            "pred_cur": self.cumulative_prob_layer_cur(f_cur_pooled),
            "pred_pri": self.cumulative_prob_layer_pri(f_pri_pooled),
        }


#  Spatial Transformer for applying deformation fields
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
