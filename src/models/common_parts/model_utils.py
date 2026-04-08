import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# Spatial Transformer
# -------------------------

class SpatialTransformerBlock(nn.Module):
    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def _create_base_grid(self, B, H, W, device):
        """Create normalized base grid."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    def forward(self, x, flow):
        """
        Args:
            x:     [B, C, H, W]
            flow:  [B, 2, H, W] (dx, dy in pixel space)

        Returns:
            Warped tensor: [B, C, H, W]
        """
        B, _, H, W = x.shape
        device = x.device

        base_grid = self._create_base_grid(B, H, W, device)

        # Normalize flow to [-1, 1]
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0] = flow[:, 0] * 2 / (W - 1)  # dx
        flow_norm[:, 1] = flow[:, 1] * 2 / (H - 1)  # dy

        flow_norm = flow_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]

        sampling_grid = base_grid + flow_norm

        return F.grid_sample(
            x,
            sampling_grid,
            mode=self.mode,
            align_corners=True,
        )


# -------------------------
# Continuous Positional Encoding
# -------------------------

class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, max_time=5.0, num_steps=240):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.max_time = max_time
        self.num_steps = num_steps

        position = torch.linspace(0, max_time, steps=num_steps).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, times):
        """
        Args:
            x:     (N, B, C)
            times: (B,)
        """
        # Normalize to index space
        t = torch.clamp(times, 0, self.max_time)
        t = t * (self.num_steps - 1) / self.max_time

        t_floor = torch.floor(t).long()
        t_ceil = torch.ceil(t).long().clamp(max=self.num_steps - 1)

        alpha = (t - t_floor).unsqueeze(1)

        pe_floor = self.pe[t_floor]
        pe_ceil = self.pe[t_ceil]

        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil

        return self.dropout(x + pe_interp.unsqueeze(0))


# -------------------------
# Cumulative Probability Layer
# -------------------------

class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        super().__init__()

        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)

        # Register mask as buffer (not parameter!)
        mask = torch.tril(torch.ones(max_followup, max_followup))
        self.register_buffer("triangular_mask", mask.T)

    def hazards(self, x):
        return self.relu(self.hazard_fc(x))

    def forward(self, x):
        """
        Returns:
            cumulative logits: [B, T]
        """
        hazards = self.hazards(x)  # [B, T]
        B, T = hazards.shape

        expanded = hazards.unsqueeze(-1).expand(B, T, T)
        masked = expanded * self.triangular_mask

        cum_logits = masked.sum(dim=1) + self.base_hazard_fc(x)

        return cum_logits