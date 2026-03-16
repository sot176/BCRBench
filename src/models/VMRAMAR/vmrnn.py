import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath

# ── Proper PyTorch selective scan fallback ─────────────────────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """
    Pure PyTorch S6 selective scan. Matches mamba_ssm signature exactly.
    u, delta: (B, D, L)
    A:        (D, N)
    B, C:     (B, N, L)
    D:        (D,)
    z:        (B, D, L)
    """
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

    # Discretize: deltaA (B,D,L,N), deltaB_u (B,D,L,N)
    deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A.float()))
    if B_in.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B_in, u)
    else:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B_in[:, 0], u)

    # Sequential recurrence
    x  = torch.zeros(batch, d_model, N, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C_in.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C_in[:, :, i])
        else:
            y = torch.einsum('bdn,bn->bd', x, C_in[:, 0, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)   # (B, D, L)

    if D is not None:
        y = y + u * D.float().unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z.float())

    y = y.to(dtype_in)
    return (y, x) if return_last_state else y


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_fn
    MAMBA_AVAILABLE = True
    print("✅ Using mamba_ssm GPU kernel")
except ImportError:
    selective_scan_fn = selective_scan_ref
    MAMBA_AVAILABLE = False
    print("⚠️  mamba_ssm not found — using pure PyTorch selective scan fallback")


# ── SS2D — exact copy from their vmamba.py, uses selective_scan_fn above ──
import math
from einops import repeat

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(self.expand * self.d_model)
        self.dt_rank  = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj  = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d   = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                  bias=conv_bias, kernel_size=d_conv,
                                  padding=(d_conv - 1) // 2)
        self.act      = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False).weight
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

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds     = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
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
        A_log = nn.Parameter(torch.log(A))
        if copies > 1:
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
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B,4,D,L)

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
        As  = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
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


# ── VSB — their implementation, using SS2D ────────────────────────────────
class VSB(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop=0., d_state=16):
        super().__init__()
        self.ln_1           = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop, d_state=d_state)
        self.drop_path      = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.linear         = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, hx=None):
        """
        x:  (B, H, W, C)   — spatial format as SS2D expects
        hx: (B, H, W, C)   — previous hidden state, same shape
        """
        shortcut = x
        x = self.ln_1(x)
        if hx is not None:
            hx = self.ln_1(hx)
            x  = self.linear(torch.cat([x, hx], dim=-1))
        x = self.drop_path(self.self_attention(x))
        return shortcut + x


# ── VMRNNCell — their implementation ─────────────────────────────────────
class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop_path=0., attn_drop=0., d_state=16):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.input_resolution = input_resolution   # (H, W)
        self.vsbs = nn.ModuleList([
            VSB(hidden_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attn_drop=attn_drop, d_state=d_state)
            for i in range(depth)
        ])

    def forward(self, xt, hidden_states):
        """
        xt:            (B, L, C)  where L = H*W
        hidden_states: None or (Ht, Ct) each (B, L, C)
        Returns:
            Ht: (B, L, C)
            (Ht, Ct)
        """
        H, W = self.input_resolution
        B, L, C = xt.shape

        if hidden_states is None:
            hx = torch.zeros_like(xt)
            cx = torch.zeros_like(xt)
        else:
            hx, cx = hidden_states

        # Reshape to spatial for SS2D
        x  = xt.view(B, H, W, C)
        hx_s = hx.view(B, H, W, C)

        # First VSB fuses x with previous hidden state
        x = self.vsbs[0](x, hx_s)
        # Subsequent VSBs refine without hidden state
        for vsb in self.vsbs[1:]:
            x = vsb(x, None)

        # Back to sequence
        x = x.view(B, L, C)

        # LSTM-style update
        Ft = torch.sigmoid(x)
        Ct = Ft * (cx + torch.tanh(x))
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)


# ── DownSample / UpSample — their structure, spatial PatchMerging/Expanding ──
class PatchMerging(nn.Module):
    """Halves spatial resolution, doubles channels."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm      = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        # Pad if odd
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            _, H, W, _ = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x  = self.norm(x)
        x  = self.reduction(x)                       # (B, H/2, W/2, 2C)
        return x


class PatchExpanding(nn.Module):
    """Doubles spatial resolution, halves channels."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = norm_layer(dim // 2)

    def forward(self, x):
        from einops import rearrange
        B, H, W, C = x.shape
        x = self.expand(x)                           # (B, H, W, 2C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=2, p2=2, c=C // 2)          # (B, 2H, 2W, C/2)
        x = self.norm(x)
        return x


class DownSample(nn.Module):
    def __init__(self, embed_dim, depths, feature_resolution,
                 drop_path_rate=0.1, attn_drop=0., d_state=16):
        super().__init__()
        H, W   = feature_resolution
        dpr    = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers     = nn.ModuleList()
        self.downsample = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            res = (H // (2 ** i), W // (2 ** i))
            self.layers.append(VMRNNCell(
                dim, res, depth,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                attn_drop=attn_drop, d_state=d_state
            ))
            if i < len(depths) - 1:
                self.downsample.append(PatchMerging(dim))
                dim *= 2                          
            else:
                self.downsample.append(nn.Identity())
                                                  

        self.out_dim = dim                         

    def forward(self, x, states):
        """x: (B, H, W, C)"""
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        skips      = []
        for layer, down, state in zip(self.layers, self.downsample, states):
            B, H, W, C = x.shape
            x_seq = x.view(B, H * W, C)
            x_seq, new_state = layer(x_seq, state)
            new_states.append(new_state)
            x = x_seq.view(B, H, W, C)
            skips.append(x)
            x = down(x)
        return new_states, skips, x


class UpSample(nn.Module):
    def __init__(self, embed_dim, depths, feature_resolution,
                 drop_path_rate=0.1, attn_drop=0., d_state=16):
        super().__init__()
        H, W   = feature_resolution
        n      = len(depths)
        dpr    = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers   = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()

        # Start from bottom dim — matches DownSample's out_dim
        # DownSample doubles dim at each level except last, so:
        dim = embed_dim * (2 ** (n - 1))

        for i, depth in enumerate(depths):
            level = n - 1 - i
            res   = (H // (2 ** level), W // (2 ** level))
            self.layers.append(VMRNNCell(
                dim, res, depth,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                attn_drop=attn_drop, d_state=d_state
            ))
            self.upsample.append(
                PatchExpanding(dim) if i < n - 1 else nn.Identity()
            )
            # After skip cat, fuse channels back to dim
            self.skip_fuse.append(nn.Linear(dim * 2, dim))
            out_dim = dim // 2 if i < n - 1 else dim
            dim = out_dim

    def forward(self, x, skips, states):
        """x: (B, H, W, C)"""
        if states is None:
            states = [None] * len(self.layers)
        new_states = []
        for i, (layer, up, fuse, state) in enumerate(
            zip(self.layers, self.upsample, self.skip_fuse, states)
        ):
            # Fuse with skip from encoder
            skip = skips[-(i + 1)]
            if x.shape[1:3] != skip.shape[1:3]:
                x = x[:, :skip.shape[1], :skip.shape[2], :]
            x = fuse(torch.cat([x, skip], dim=-1))

            B, H, W, C = x.shape
            x_seq = x.view(B, H * W, C)
            x_seq, new_state = layer(x_seq, state)
            new_states.append(new_state)
            x = x_seq.view(B, H, W, C)
            x = up(x)
        return new_states, x


# ── Full VMRNN ─────────────────────────────────────────────────────────────
class VMRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                 spatial_h=16, spatial_w=16,
                 depths_down=(2, 2), depths_up=(2, 2),
                 drop_path_rate=0.1, attn_drop=0., d_state=16):
        super().__init__()
        assert len(depths_down) == len(depths_up)
        feature_resolution = (spatial_h, spatial_w)

        # Project input embedding to spatial hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.down = DownSample(hidden_dim, depths_down, feature_resolution,
                               drop_path_rate, attn_drop, d_state)
        self.up   = UpSample(hidden_dim, depths_up, feature_resolution,
                             drop_path_rate, attn_drop, d_state)

        # Project back to input_dim for the rest of the model
        self.out_norm = nn.LayerNorm(hidden_dim)          # final decoder output = hidden_dim
        self.out_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, input_dim),             # 64 → 512
        )

    def forward(self, Tt, states_down=None, states_up=None):
        B = Tt.shape[0]
        H, W = self.down.layers[0].input_resolution

        x = self.input_proj(Tt)                        # (B, hidden_dim)
        x = x.unsqueeze(1).unsqueeze(1)
        x = x.expand(B, H, W, -1).contiguous()        # (B, H, W, hidden_dim)

        states_down, skips, x = self.down(x, states_down)
        states_up,   x        = self.up(x, skips, states_up)

        x = self.out_norm(x)                           # (B, H, W, hidden_dim) — correct shape now
        x = x.permute(0, 3, 1, 2)                     # (B, hidden_dim, H, W)
        out = self.out_proj(x)                         # (B, input_dim)

        return out, states_down, states_up