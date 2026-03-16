import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################
# Linear Projection layer — fuses Tt and Ht-1, reshapes to spatial
############################################################
class LinearProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, spatial_h, spatial_w):
        super().__init__()
        self.spatial_h  = spatial_h
        self.spatial_w  = spatial_w
        self.hidden_dim = hidden_dim
        hidden_flat_dim = hidden_dim * spatial_h * spatial_w   # full flattened size

        # input_dim: dim of Tt  (e.g. 512)
        # hidden_flat_dim: dim of Ht_prev flattened (e.g. 512*8*8 = 32768)
        self.proj = nn.Linear(
            input_dim + hidden_flat_dim,               # was: input_dim + hidden_dim ← bug
            hidden_flat_dim
        )

    def forward(self, Tt, Ht_prev):
        """
        Tt:     (B, input_dim)
        Ht_prev:(B, hidden_dim * H * W)  — already flat
        """
        x  = torch.cat([Tt, Ht_prev], dim=-1)         # (B, input_dim + hidden_flat_dim)
        x  = self.proj(x)                              # (B, hidden_flat_dim)
        B  = x.shape[0]
        return x.view(B, self.hidden_dim, self.spatial_h, self.spatial_w)

############################################################
# VSS Block — approximates S6 directional scanning
# Full Mamba requires custom CUDA kernels; this uses depthwise
# conv + channel mixing as a tractable approximation
############################################################
class VSSBlock(nn.Module):
    """
    Approximation of the VSS Block from the paper:
      Primary stream:   DWConv3x3 → SiLU → directional mixing
      Secondary stream: SiLU(A)
      Combine: A3 + B1, then sigmoid gate
    """
    def __init__(self, dim):
        super().__init__()
        # Primary stream
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

        # Directional mixing — approximate 4-direction S6 with 4 depthwise convs
        # with different asymmetric kernels to capture directional context
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),  # horizontal
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),  # vertical
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),  # horizontal flip
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),  # vertical flip
        ])
        self.merge_norm = nn.LayerNorm(dim)
        self.merge_proj = nn.Linear(dim * 4, dim)

    def forward(self, A):
        """
        A: (B, C, H, W)
        Returns: Ft (B, C, H, W) — sigmoid gate, and Y (B, C, H, W)
        """
        # Primary stream: DWConv → SiLU → 4-direction scan → merge → LayerNorm
        A1 = F.silu(self.dw_conv(A))                                 # (B, C, H, W)

        # Directional scans (approximate S6 with asymmetric depthwise convs)
        dir_feats = []
        for i, conv in enumerate(self.dir_convs):
            feat = A1
            # Flip for reverse directions (dirs 2 and 3)
            if i == 2:
                feat = torch.flip(feat, dims=[-1])
            elif i == 3:
                feat = torch.flip(feat, dims=[-2])
            feat = conv(feat)
            if i == 2:
                feat = torch.flip(feat, dims=[-1])
            elif i == 3:
                feat = torch.flip(feat, dims=[-2])
            dir_feats.append(feat)

        # Merge 4 directions → LayerNorm
        merged = torch.cat(dir_feats, dim=1)                         # (B, 4C, H, W)
        B, _, H, W = merged.shape
        merged = merged.permute(0, 2, 3, 1)                          # (B, H, W, 4C)
        A3 = self.merge_norm(self.merge_proj(merged))                # (B, H, W, C)
        A3 = A3.permute(0, 3, 1, 2)                                  # (B, C, H, W)

        # Secondary stream: SiLU(A)
        B1 = F.silu(A)                                               # (B, C, H, W)

        # Combine — paper eq (4): Y = A3 + B1
        Y = A3 + B1                                                  # (B, C, H, W)

        # Gate — paper eq: Ft = sigmoid(Y)
        Ft = torch.sigmoid(Y)

        return Ft, Y


############################################################
# VMRNN Cell — follows paper equations exactly
############################################################
class VMRNNCell(nn.Module):
    """
    Per paper:
        Xt  = LP(Tt, Ht-1)           — linear projection + reshape
        Ft  = VSS(Xt)                — gating signal
        Ct  = Ft ⊙ (tanh(Y) + Ct-1) — cell update
        Ht  = Ft ⊙ tanh(Ct)         — hidden update
    """
    def __init__(self, input_dim, hidden_dim, spatial_h, spatial_w):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_h  = spatial_h
        self.spatial_w  = spatial_w
        self.spatial_size = spatial_h * spatial_w

        self.lp  = LinearProjection(input_dim, hidden_dim, spatial_h, spatial_w)
        self.vss = VSSBlock(hidden_dim)

    def forward(self, Tt, state):
        """
        Tt:    (B, input_dim)  — fused embedding for timestep t
        state: None or (Ht, Ct) where each is (B, hidden_dim*H*W)
        Returns:
            Ht_flat: (B, hidden_dim*H*W)
            (Ht_flat, Ct_flat)
        """
        B = Tt.shape[0]
        hidden_size = self.hidden_dim * self.spatial_size

        if state is None:
            Ht_prev = torch.zeros(B, hidden_size, device=Tt.device, dtype=Tt.dtype)
            Ct_prev = torch.zeros(B, hidden_size, device=Tt.device, dtype=Tt.dtype)
        else:
            Ht_prev, Ct_prev = state

        Ct_prev_spatial = Ct_prev.view(B, self.hidden_dim, self.spatial_h, self.spatial_w)

        # LP: fuse Tt and Ht-1, reshape to spatial
        Xt = self.lp(Tt, Ht_prev)                                    # (B, C, H, W)

        # VSS: get gate Ft and pre-gate Y
        Ft, Y = self.vss(Xt)                                         # both (B, C, H, W)

        # Cell and hidden state updates — paper equations
        Ct = Ft * (torch.tanh(Y) + Ct_prev_spatial)                  # (B, C, H, W)
        Ht = Ft * torch.tanh(Ct)                                     # (B, C, H, W)

        Ht_flat = Ht.contiguous().view(B, -1)
        Ct_flat = Ct.contiguous().view(B, -1)                                     # (B, C*H*W)

        return Ht_flat, (Ht_flat, Ct_flat)


############################################################
# Conv Downsample / Upsample (unchanged)
############################################################
class ConvDownsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, H, W):
        B, C, H_in, W_in = x.shape
        x = self.conv(x)
        return x, x.shape[2], x.shape[3]


class ConvUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, H, W):
        x = self.conv(x)
        return x, x.shape[2], x.shape[3]


############################################################
# Full VMRNN — downsample + upsample with skip connections
# Operates on spatial tensors (B, C, H, W) throughout
############################################################
class VMRNN(nn.Module):
    """
    At each timestep t receives Tt: (B, input_dim).
    Internally maintains spatial hidden/cell states at each scale.
    Returns output embedding: (B, output_dim).
    """
    def __init__(
        self,
        input_dim,                      # dim of Tt coming in (embed_dim after aggregator)
        hidden_dim=256,                 # spatial hidden state channels
        spatial_h=4,                    # spatial H for hidden state grid
        spatial_w=4,                    # spatial W for hidden state grid
        depths_down=(2, 2, 6),
        depths_up=(2, 2, 2),
    ):
        super().__init__()
        assert len(depths_down) == len(depths_up)
        self.spatial_h  = spatial_h
        self.spatial_w  = spatial_w
        self.hidden_dim = hidden_dim
        n_levels = len(depths_down)

        # ── Encoder cells + downsamplers ──────────────────────────────
        self.down_cells   = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        dim = hidden_dim
        # First cell takes input_dim as Tt, rest take previous hidden dim
        cell_input_dims = []
        self.scale_shapes = []  # (H, W) at each level
        H, W = spatial_h, spatial_w
        for i, depth in enumerate(depths_down):
            in_d = input_dim if i == 0 else hidden_dim * (2 ** (i-1)) * (H * W)
            cell_input_dims.append(in_d)
            self.down_cells.append(VMRNNCell(input_dim, dim, H, W))
            self.scale_shapes.append((H, W))
            self.downsamplers.append(ConvDownsample(dim, dim * 2))
            H, W = H // 2, W // 2
            dim *= 2

        # ── Decoder cells + upsamplers + skip projections ─────────────
        self.up_cells     = nn.ModuleList()
        self.upsamplers   = nn.ModuleList()
        self.skip_projs   = nn.ModuleList()
        for i, depth in enumerate(depths_up):
            out_dim = dim // 2
            self.upsamplers.append(ConvUpsample(dim, out_dim))
            self.skip_projs.append(nn.Conv2d(out_dim * 2, out_dim, kernel_size=1))
            self.up_cells.append(VMRNNCell(input_dim, out_dim,
                                           self.scale_shapes[-(i+1)][0],
                                           self.scale_shapes[-(i+1)][1]))
            dim = out_dim

        # ── Output projection: spatial hidden → 1D embedding ──────────
        self.out_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, input_dim),    # ← projects back to embed_dim (e.g. 512)
        )

    @staticmethod
    def _match_spatial(x, skip):
        """Crop x to match skip's spatial dims if ConvTranspose2d overshoots."""
        if x.shape[2:] != skip.shape[2:]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        return x

    def forward(self, Tt, states_down=None, states_up=None):
        """
        Tt:          (B, input_dim)
        states_down: list of (Ht, Ct) per encoder level, or None
        states_up:   list of (Ht, Ct) per decoder level, or None
        Returns:
            out:         (B, hidden_dim) — pooled output embedding
            states_down: updated
            states_up:   updated
        """
        n = len(self.down_cells)
        if states_down is None:
            states_down = [None] * n
        if states_up is None:
            states_up = [None] * n

        new_states_down = []
        new_states_up   = []
        skips = []

        # ── Encoder ───────────────────────────────────────────────────
        x = None
        H, W = self.spatial_h, self.spatial_w
        for i, (cell, down) in enumerate(zip(self.down_cells, self.downsamplers)):
            Ht_flat, state = cell(Tt, states_down[i])
            new_states_down.append(state)
            # Reshape hidden state to spatial for conv operations
            x_spatial = Ht_flat.view(
                Ht_flat.shape[0], self.down_cells[i].hidden_dim,
                self.scale_shapes[i][0], self.scale_shapes[i][1]
            )
            skips.append(x_spatial)
            x_spatial, H, W = down(x_spatial, H, W)
            x = x_spatial

        # ── Decoder ───────────────────────────────────────────────────
        for i, (cell, up, proj) in enumerate(
            zip(self.up_cells, self.upsamplers, self.skip_projs)
        ):
            x, H, W = up(x, H, W)
            skip = skips[-(i + 1)]
            x = self._match_spatial(x, skip)
            x = proj(torch.cat([x, skip], dim=1))        # skip fusion

            # Run up cell using Tt as input
            B = x.shape[0]
            x_flat = x.view(B, -1)
            # Create a compatible Ht from x for the cell
            Ht_flat, state = cell(Tt, states_up[i])
            new_states_up.append(state)
            # Add cell output to decoder features
            x = x + Ht_flat.view_as(x)

        # ── Output pool ───────────────────────────────────────────────
        out = self.out_proj(x)                           # (B, hidden_dim)

        return out, new_states_down, new_states_up