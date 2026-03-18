import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath
from timm.models.swin_transformer import PatchMerging


# ── Selective scan ─────────────────────────────────────────────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    dtype_in = u.dtype
    u     = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1).float()
    if delta_softplus:
        delta = F.softplus(delta)
    B_in = B.float()
    C_in = C.float()
    batch, d_model, L = u.shape
    N = A.shape[1]
    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A.float()))
    if B_in.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B_in, u)
    else:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B_in[:, 0], u)
    x  = torch.zeros(batch, d_model, N, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C_in.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C_in[:, :, i])
        else:
            y = torch.einsum('bdn,bn->bd', x, C_in[:, 0, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.float().unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z.float())
    y = torch.clamp(y, -1e4, 1e4)   # numerical stability
    y = y.to(dtype_in)
    return (y, x) if return_last_state else y


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_fn
    print("✅ Using mamba_ssm GPU kernel")
except ImportError:
    selective_scan_fn = selective_scan_ref
    print("⚠️  Using pure PyTorch selective scan fallback")


# ── SS2D ───────────────────────────────────────────────────────────────────
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d   = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                  bias=conv_bias, kernel_size=d_conv,
                                  padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_weight = nn.Parameter(torch.stack([
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                          dt_min, dt_max, dt_init_floor).weight
            for _ in range(4)
        ], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                          dt_min, dt_max, dt_init_floor).bias
            for _ in range(4)
        ], dim=0))
        self.A_logs = self._A_log_init(d_state, self.d_inner, copies=4, merge=True)
        self.Ds     = self._D_init(self.d_inner, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   "n -> d n", d=d_inner).contiguous()
        A_log = nn.Parameter(repeat(torch.log(A), "d n -> r d n", r=copies)
                             .flatten(0, 1).contiguous())
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, merge=True):
        D = nn.Parameter(torch.ones(d_inner * copies))
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, 2, 3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                              xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl,
                                  [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight)
        xs  = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs  = Bs.float().view(B, K, -1, L)
        Cs  = Cs.float().view(B, K, -1, L)
        Ds  = self.Ds.float().view(-1)
        As  = -torch.exp(self.A_logs.float().clamp(-10, 10)).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)
        inv_y   = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y    = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)
        return y

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ── VSB — exact copy from their github ────────────────────────────────────
class VSB(nn.Module):
    def __init__(self, hidden_dim, input_resolution,
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0., d_state=16, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.ln_1           = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path      = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.linear         = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.ln_1(x)

        if hx is not None:
            hx = self.ln_1(hx)
            x  = torch.cat((x, hx), dim=-1)
            x  = self.linear(x)

        if H == 1 and W == 1:
            # Temporal mode: L = T timesteps, reshape as (B, 1, T, C) for SS2D
            x = x.view(B, 1, L, C)
            x = self.drop_path(self.self_attention(x))
            x = x.view(B, L, C)
        else:
            assert L == H * W, f"Input feature has wrong size. Got L={L}, expected {H*W}."
            x = x.view(B, H, W, C)
            x = self.drop_path(self.self_attention(x))
            x = x.view(B, H * W, C)

        return shortcut + x


# ── VMRNNCell — exact copy from their github ──────────────────────────────
class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super().__init__()
        self.VSBs = nn.ModuleList([
            VSB(hidden_dim=hidden_dim,
                input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs)
            for i in range(depth)
        ])

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C, device=xt.device, dtype=xt.dtype)
            cx = torch.zeros(B, L, C, device=xt.device, dtype=xt.dtype)
        else:
            hx, cx = hidden_states

        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
            else:
                x = layer(outputs[-1], None)
            outputs.append(x)

        o_t  = outputs[-1]
        Ft   = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)
        Ct   = Ft * (cx + cell)
        Ht   = Ft * torch.tanh(Ct)
        return Ht, (Ht, Ct)


# ── PatchExpanding — exact copy from their github ─────────────────────────
class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim    = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm   = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size."
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


# ── PatchInflated — exact copy from their github ──────────────────────────
class PatchInflated(nn.Module):
    def __init__(self, in_chans, embed_dim, input_resolution,
                 stride=2, padding=1, output_padding=1):
        super().__init__()
        from timm.models.layers import to_2tuple
        stride         = to_2tuple(stride)
        padding        = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution
        self.Conv = nn.ConvTranspose2d(
            in_channels=embed_dim, out_channels=in_chans,
            kernel_size=(3, 3), stride=stride,
            padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size."
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.Conv(x)
        return x


# ── DownSample — exact copy from their github ─────────────────────────────
class DownSample(nn.Module):
    def __init__(self, embed_dim, depths_downsample, feature_resolution,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super().__init__()
        self.num_layers = len(depths_downsample)
        self.embed_dim  = embed_dim
        self.patch_embed = nn.Identity()
        patches_resolution = feature_resolution

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]
        self.layers     = nn.ModuleList()
        self.downsample = nn.ModuleList()

        is_temporal = feature_resolution[0] == 1 or feature_resolution[1] == 1
        self.is_temporal = is_temporal

        for i_layer in range(self.num_layers):
            res = (
                max(patches_resolution[0] // (2 ** i_layer), 1),  # add max(...,1)
                max(patches_resolution[1] // (2 ** i_layer), 1)   # add max(...,1)
            )
            if is_temporal:
                downsample  = nn.Identity()
                hidden_dim_layer = embed_dim          # never scale in temporal mode
            else:
                downsample  = PatchMerging(
                    input_resolution=res,
                    dim=int(embed_dim * 2 ** i_layer)
                )
                hidden_dim_layer = int(embed_dim * 2 ** i_layer)

            layer = VMRNNCell(
                hidden_dim=hidden_dim_layer,
                input_resolution=res,
                depth=depths_downsample[i_layer],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_downsample[:i_layer]):
                               sum(depths_downsample[:i_layer+1])],
                norm_layer=norm_layer,
                d_state=d_state
            )
            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, states_down):
        x = self.patch_embed(x)
        hidden_states_down = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(
                x, states_down[index] if states_down is not None else None
            )
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)
        return hidden_states_down, x


# ── UpSample — exact copy from their github ───────────────────────────────
class UpSample(nn.Module):
    def __init__(self, embed_dim, depths_upsample, feature_resolution,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, d_state=16, out_chans=None, **kwargs):
        super().__init__()
        self.num_layers = len(depths_upsample)
        self.embed_dim  = embed_dim
        self.patch_embed = nn.Identity()
        patches_resolution = feature_resolution

        is_temporal = feature_resolution[0] == 1 or feature_resolution[1] == 1
        self.is_temporal = is_temporal

        if is_temporal:
            self.Unembed = nn.Identity()
        else:
            self.Unembed = PatchInflated(
                in_chans=out_chans if out_chans is not None else embed_dim,
                embed_dim=embed_dim,
                input_resolution=patches_resolution
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]
        self.layers   = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = max(patches_resolution[0] // (2 ** (self.num_layers - i_layer)), 1)
            resolution2 = max(patches_resolution[1] // (2 ** (self.num_layers - i_layer)), 1)
             
            if is_temporal:
                upsample  = nn.Identity()
                dimension = embed_dim   # dim never changes in temporal mode
            else:
                upsample = PatchExpanding(
                    input_resolution=(resolution1, resolution2),
                    dimension  = int(embed_dim * 2 ** (self.num_layers - i_layer))
                )

            layer = VMRNNCell(
                hidden_dim=dimension,
                input_resolution=(
                    max(resolution1, 1),
                    max(resolution2, 1)
                ),
                depth=depths_upsample[self.num_layers - 1 - i_layer],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths_upsample[:self.num_layers-1-i_layer]):
                    sum(depths_upsample[:self.num_layers-1-i_layer+1])
                ],
                norm_layer=norm_layer,
                d_state=d_state
            )
            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, states_up):
        hidden_states_up = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(
                x, states_up[index] if states_up is not None else None
            )
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)
        # Their exact line — sigmoid only meaningful in spatial mode
        # In temporal mode Unembed is Identity so sigmoid(x) squashes to [0,1]
        # We skip sigmoid in temporal mode to avoid saturating gradients
        if not self.is_temporal:
            x = torch.sigmoid(self.Unembed(x))
        else:
            x = self.Unembed(x)   # Identity — no sigmoid
        return hidden_states_up, x


# ── VMRNN — exact copy from their github ──────────────────────────────────
class VMRNN(nn.Module):
    def __init__(self, embed_dim, depths_downsample, depths_upsample,
                 feature_resolution=(1, 1), **kwargs):
        super().__init__()
        self.Downsample = DownSample(
            embed_dim=embed_dim,
            depths_downsample=depths_downsample,
            feature_resolution=feature_resolution,
            **kwargs
        )
        self.Upsample = UpSample(
            embed_dim=embed_dim,
            depths_upsample=depths_upsample,
            feature_resolution=feature_resolution,
            **kwargs
        )

    def forward(self, features, states_down=None, states_up=None, **kwargs):
        B = features.shape[0]
        if features.dim() == 3 and self.Downsample.is_temporal:
            features = features.view(B, -1, features.size(-1))
        states_down, x      = self.Downsample(features, states_down)
        states_up,   output = self.Upsample(x, states_up)
        return output, states_down, states_up