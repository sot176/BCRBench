import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_parts import ContinuousPosEncoding, CumulativeProbabilityLayer


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
        flattened_feats = f_dif.flatten(start_dim=2).permute(2, 0, 1)  # [N, B, C]
        fdif_with_time = self.positional_encoding(flattened_feats, time_gap)
        fdif_with_time = fdif_with_time.permute(1, 2, 0).view(B, C, H, W)

        # Global average pooling
        f_cur_pooled = F.adaptive_avg_pool2d(f_cur, (1, 1)).view(B, C)
        f_pri_pooled = F.adaptive_avg_pool2d(f_pri, (1, 1)).view(B, C)
        f_pri_aligned_pooled = F.adaptive_avg_pool2d(f_pri_aligned, (1, 1)).view(B, C)
        fdif_pooled = F.adaptive_avg_pool2d(fdif_with_time, (1, 1)).view(B, C)

        # Temporal attention
        stacked = torch.stack([f_pri_aligned_pooled, fdif_pooled, f_cur_pooled], dim=1)  # [B, 3, C]
        attended = self.attention_layer(stacked.permute(1, 0, 2))  # [3, B, C]
        attended = attended.permute(1, 0, 2).mean(dim=1)  # [B, C]
        fused_feat = self.feature_projection(attended)

        return {
            "pred_fused": self.cumulative_prob_layer_fused(fused_feat),
            "pred_cur": self.cumulative_prob_layer_cur(f_cur_pooled),
            "pred_pri": self.cumulative_prob_layer_pri(f_pri_pooled),
        }
