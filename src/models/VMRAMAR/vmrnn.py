import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

############################################################
# Conv Downsample / Upsample
############################################################
class ConvDownsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"Expected L={H*W}, got L={L}"
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        B, C, H_out, W_out = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H_out * W_out, C)
        return x, H_out, W_out


class ConvUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"Expected L={H*W}, got L={L}"
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)
        x = self.conv(x)                                # (B, out_ch, H*2, W*2)
        B, C, H_out, W_out = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H_out * W_out, C)
        return x, H_out, W_out


############################################################
# VSB Block
############################################################
class VSB(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hx=None):
        shortcut = x
        x = self.norm(x)
        if hx is not None:
            hx = self.norm(hx)
            x = self.linear(torch.cat([x, hx], dim=-1))
        x, _ = self.self_attention(x, x, x)
        x = self.drop_path(x)
        return shortcut + x


############################################################
# VMRNN Cell
############################################################
class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, depth, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        # Remove input_resolution — cells don't need it, shapes come at runtime
        self.layers = nn.ModuleList([
            VSB(hidden_dim, drop_path, norm_layer)
            for _ in range(depth)
        ])

    def forward(self, xt, states):
        if states is None:
            hx = torch.zeros_like(xt)
            cx = torch.zeros_like(xt)
        else:
            hx, cx = states
        x = xt
        for layer in self.layers:
            x = layer(x, hx)
        gate = torch.sigmoid(x)
        cell = torch.tanh(x)
        Ct = gate * (cx + cell)
        Ht = gate * torch.tanh(Ct)
        return Ht, (Ht, Ct)


############################################################
# Downsample Encoder
############################################################
class DownSample(nn.Module):
    def __init__(self, embed_dim, depths):
        super().__init__()
        self.cells = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        dim = embed_dim
        for depth in depths:
            self.cells.append(VMRNNCell(dim, depth))
            self.downsamplers.append(ConvDownsample(dim, dim * 2))
            dim *= 2
        self.out_dim = dim  # dim at the bottom of the encoder

    def forward(self, x, H, W, states):
        if states is None:
            states = [None] * len(self.cells)
        new_states = []
        skips = []                          # save pre-downsample features for skip connections
        for i, (cell, down) in enumerate(zip(self.cells, self.downsamplers)):
            x, state = cell(x, states[i])
            new_states.append(state)
            skips.append((x, H, W))        # (B, L, C) before spatial reduction
            x, H, W = down(x, H, W)
        return new_states, skips, x, H, W


############################################################
# Upsample Decoder
############################################################
class UpSample(nn.Module):
    def __init__(self, embed_dim, depths):
        """
        embed_dim: base encoder dim.
        depths:    list of depths, same length as DownSample depths.
        
        Decoder starts from embed_dim * 2^len(depths) (bottom of encoder)
        and halves dim each step, mirroring the encoder.
        """
        super().__init__()
        self.upsamplers = nn.ModuleList()
        self.cells = nn.ModuleList()
        self.skip_projs = nn.ModuleList()   # project concatenated skip+x back to dim

        n = len(depths)
        dim = embed_dim * (2 ** n)          # start at bottom dim

        for depth in depths:
            out_dim = dim // 2
            self.upsamplers.append(ConvUpsample(dim, out_dim))
            # After skip concat, channel count doubles → project back to out_dim
            self.skip_projs.append(nn.Linear(out_dim * 2, out_dim))
            self.cells.append(VMRNNCell(out_dim, depth))
            dim = out_dim

    def forward(self, x, H, W, skips, states):
        if states is None:
            states = [None] * len(self.cells)
        new_states = []
        for i, (up, proj, cell) in enumerate(
            zip(self.upsamplers, self.skip_projs, self.cells)
        ):
            x, H, W = up(x, H, W)                           # upsample → out_dim
            skip_x, _, _ = skips[-(i + 1)]                  # matching encoder level
            x = proj(torch.cat([x, skip_x], dim=-1))        # fuse skip connection
            x, state = cell(x, states[i])
            new_states.append(state)
        return new_states, x, H, W


############################################################
# Full VMRNN
############################################################
class VMRNN(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        depths_down=(2, 2, 6),
        depths_up=(2, 2, 2),
        feature_resolution=(64, 52),
    ):
        super().__init__()
        assert len(depths_down) == len(depths_up), \
            "depths_down and depths_up must have the same length"
        self.feature_resolution = feature_resolution
        self.down = DownSample(embed_dim, depths_down)
        self.up = UpSample(embed_dim, depths_up)

    def forward(self, x, states_down=None, states_up=None):
        """
        Args:
            x:           (B, L, C)  where L = H*W
            states_down: list of (Ht, Ct) per encoder level, or None
            states_up:   list of (Ht, Ct) per decoder level, or None
        Returns:
            out:         (B, L, C)  — same spatial size as input
            states_down: updated encoder states
            states_up:   updated decoder states
        """
        H, W = self.feature_resolution
        states_down, skips, x, H_bot, W_bot = self.down(x, H, W, states_down)
        states_up, x, H_out, W_out = self.up(x, H_bot, W_bot, skips, states_up)
        return x, states_down, states_up