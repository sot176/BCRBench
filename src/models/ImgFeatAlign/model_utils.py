import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_parts import ContinuousPosEncoding, CumulativeProbabilityLayer


# -------------------------
# Temporal Attention Layer
# -------------------------

class TemporalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # we use (S, B, C)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """
        Args:
            x: (S, B, C)
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


# -------------------------
# Risk Model
# -------------------------

class RiskModelWithAttention(nn.Module):
    def __init__(self, feature_dim=512, num_years=5, num_heads=8):
        super().__init__()

        self.feature_dim = feature_dim

        self.positional_encoding = ContinuousPosEncoding(dim=feature_dim)
        self.attention_layer = TemporalAttentionLayer(dim=feature_dim, num_heads=num_heads)

        self.feature_projection = nn.Linear(feature_dim, feature_dim)

        # Prediction heads
        self.head_fused = CumulativeProbabilityLayer(feature_dim, num_years)
        self.head_cur = CumulativeProbabilityLayer(feature_dim, num_years)
        self.head_pri = CumulativeProbabilityLayer(feature_dim, num_years)

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def global_pool(x):
        """Global average pooling → (B, C)"""
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, f_cur, f_pri, f_pri_aligned, f_dif, time_gap):
        """
        Args:
            f_cur:         (B, C, H, W)
            f_pri:         (B, C, H, W)
            f_pri_aligned: (B, C, H, W)
            f_dif:         (B, C, H, W)
            time_gap:      (B,) or (B, 1)
        """
        B, C, H, W = f_dif.shape

        # -------------------------
        # Time-aware encoding
        # -------------------------
        fdif_flat = f_dif.flatten(2).transpose(1, 2)   # (B, N, C)
        fdif_flat = fdif_flat.transpose(0, 1)          # (N, B, C)

        fdif_time = self.positional_encoding(fdif_flat, time_gap)

        fdif_time = fdif_time.transpose(0, 1).transpose(1, 2)  # (B, C, N)
        fdif_time = fdif_time.view(B, C, H, W)

        # -------------------------
        # Pooling
        # -------------------------
        f_cur_pooled = self.global_pool(f_cur)
        f_pri_pooled = self.global_pool(f_pri)
        f_pri_aligned_pooled = self.global_pool(f_pri_aligned)
        fdif_pooled = self.global_pool(fdif_time)

        # -------------------------
        # Temporal attention fusion
        # -------------------------
        sequence = torch.stack(
            [f_pri_aligned_pooled, fdif_pooled, f_cur_pooled],
            dim=0,  # (S=3, B, C)
        )

        attended = self.attention_layer(sequence)  # (3, B, C)
        fused = attended.mean(dim=0)               # (B, C)

        fused = self.feature_projection(fused)

        # -------------------------
        # Outputs
        # -------------------------
        return {
            "pred_fused": self.head_fused(fused),
            "pred_cur": self.head_cur(f_cur_pooled),
            "pred_pri": self.head_pri(f_pri_pooled),
        }