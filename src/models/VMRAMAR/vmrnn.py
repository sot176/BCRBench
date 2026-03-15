import torch
import torch.nn as nn
from einops import rearrange
from timm.models.swin_transformer import PatchMerging
from timm.models.layers import DropPath
import torch.nn.functional as F

############################################################
# Patch Merging with automatic padding
############################################################
class PatchMergingWrapper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.merge = PatchMerging(dim=dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Pad H/W if odd
        pad_H = H % 2
        pad_W = W % 2
        if pad_H or pad_W:
            x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H))
            H += pad_H
            W += pad_W

        # Merge patches
        x = self.merge(x)
        H, W = H // 2, W // 2
        x = x.view(B, H * W, -1)
        return x, H, W

############################################################
# Patch Expanding
############################################################
class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, C * 2)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=(C*2)//4)
        x = x.view(B, -1, (C*2)//4)
        x = self.norm(x)
        return x

############################################################
# VSB Block (Vision State Space Block)
############################################################
class VSB(nn.Module):
    def __init__(self, hidden_dim, input_resolution, drop_path=0., norm_layer=nn.LayerNorm, self_attention=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.norm = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm(x)
        if hx is not None:
            hx = self.norm(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)

        x, _ = self.self_attention(x, x, x)
        x = self.drop_path(x)
        x = shortcut + x
        return x

############################################################
# VMRNN Cell
############################################################
class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, self_attention=None):
        super().__init__()
        self.layers = nn.ModuleList([
            VSB(hidden_dim, input_resolution, drop_path, norm_layer, self_attention=self_attention)
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

        for i, depth in enumerate(depths):
            res = (H // (2 ** i), W // (2 ** i))
            self.layers.append(VMRNNCell(dim, res, depth, self_attention=self_attention))
            self.downsample.append(PatchMergingWrapper(dim))
            dim *= 2

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

        for i, depth in enumerate(depths):
            res = (H // (2 ** (len(depths) - i)), W // (2 ** (len(depths) - i)))
            self.layers.append(VMRNNCell(dim, res, depth, self_attention=self_attention))
            self.upsample.append(PatchExpanding(res, dim))
            dim //= 2

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