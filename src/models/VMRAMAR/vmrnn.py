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
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        return x, H, W

class ConvUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        if H * W != L:
            # fallback: compute W from L and H
            H, W = L // C, C
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        return x

############################################################
# VSB Block
############################################################
class VSB(nn.Module):
    def __init__(self, hidden_dim, input_resolution, drop_path=0., norm_layer=nn.LayerNorm, self_attention=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.norm = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hx=None):
        shortcut = x
        x = self.norm(x)
        if hx is not None:
            hx = self.norm(hx)
            x = torch.cat([x, hx], dim=-1)
            x = self.linear(x)
        x, _ = self.self_attention(x, x, x)
        x = self.drop_path(x)
        return shortcut + x

############################################################
# VMRNN Cell
############################################################
class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, self_attention=None):
        super().__init__()
        self.layers = nn.ModuleList([
            VSB(hidden_dim, input_resolution, drop_path, norm_layer, self_attention)
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
    def __init__(self, embed_dim, depths, feature_resolution, self_attention):
        super().__init__()
        H, W = feature_resolution
        self.H, self.W = H, W
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        dim = embed_dim

        for depth in depths:
            res = (H, W)
            self.layers.append(VMRNNCell(dim, res, depth, self_attention=self_attention))
            self.downsample.append(ConvDownsample(dim, dim * 2))
            dim *= 2
            H //= 2
            W //= 2

    def forward(self, x, states):
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        H, W = self.H, self.W
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
            x, H, W = self.downsample[i](x, H, W)
        return new_states, x

############################################################
# Upsample Decoder
############################################################
class UpSample(nn.Module):
    def __init__(self, embed_dim, depths, feature_resolution, self_attention):
        super().__init__()
        H, W = feature_resolution
        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        dim = embed_dim * (2 ** len(depths))

        for depth in depths:
            res = (H, W)
            self.layers.append(VMRNNCell(dim, res, depth, self_attention=self_attention))
            self.upsample.append(ConvUpsample(dim, dim // 2))
            dim //= 2
            H *= 2
            W *= 2

    def forward(self, x, states):
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
            x = self.upsample[i](x)
        return new_states, x

############################################################
# Full VMRNN
############################################################
class VMRNN(nn.Module):
    def __init__(self, embed_dim=256, depths_down=[2,2,6], depths_up=[2,2,2], feature_resolution=(64,52), self_attention=None):
        super().__init__()
        self.down = DownSample(embed_dim, depths_down, feature_resolution, self_attention)
        self.up = UpSample(embed_dim, depths_up, feature_resolution, self_attention)

    def forward(self, x, states_down=None, states_up=None):
        states_down, x = self.down(x, states_down)
        states_up, x = self.up(x, states_up)
        return x, states_down, states_up