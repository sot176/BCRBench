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
        x = x.permute(0, 2, 3, 1).view(B, H_out*W_out, C)
        return x, H_out, W_out

class ConvUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, H=None, W=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.H = H
        self.W = W

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape

        # Use provided H/W, else fallback to stored resolution
        H = H or self.H
        W = W or self.W
        if H is None or W is None:
            raise ValueError(f"Cannot infer H/W for ConvUpsample: L={L}, C={C}")

        # reshape to image
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.conv(x)                             # (B, out_ch, H*2, W*2)
        B, C, H_out, W_out = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H_out*W_out, C)  # back to (B, L, C)
        return x, H_out, W_out

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
        self.input_Resolution = input_resolution

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
    def __init__(self, embed_dim, depths, feature_resolution):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        H, W = feature_resolution
        dim = embed_dim

        for i, depth in enumerate(depths):
            res = (H // (2**i), W // (2**i))
            self.layers.append(VMRNNCell(dim, res, depth))
            self.downsample.append(ConvDownsample(dim, dim*2))
            dim *= 2

    def forward(self, x, states):
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        H, W = self.layers[0].input_resolution
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
            x, H, W = self.downsample[i](x, H, W)
        return new_states, x, H, W


############################################################
# Upsample Decoder
############################################################
class UpSample(nn.Module):
    def __init__(self, embed_dim, depths, feature_resolution):
        super().__init__()
        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        H, W = feature_resolution
        dim = embed_dim * (2**len(depths))

        for i, depth in enumerate(depths):
            res = (H // (2**(len(depths)-i)), W // (2**(len(depths)-i)))
            self.layers.append(VMRNNCell(dim, res, depth))
            self.upsample.append(ConvUpsample(dim, dim//2))
            dim //= 2

    def forward(self, x, states, H, W):
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
            x, H, W = self.upsample[i](x, H, W)
        return new_states, x, H, W

############################################################
# Full VMRNN
############################################################
class VMRNN(nn.Module):
    def __init__(self, embed_dim=256, depths_down=[2,2,6], depths_up=[2,2,2], feature_resolution=(64,52)):
        super().__init__()
        self.down = DownSample(embed_dim, depths_down, feature_resolution)
        self.up = UpSample(embed_dim, depths_up, feature_resolution)

    def forward(self, x, states_down=None, states_up=None):
        states_down, x, H, W = self.down(x, states_down)
        states_up, x, H, W = self.up(x, states_up, H, W)
        return x, states_down, states_up