import torch
import torch.nn as nn


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


