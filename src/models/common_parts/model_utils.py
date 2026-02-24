import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerBlock(nn.Module):
    def __init__(self, mode="bilinear"):
        """
        Applies a deformation field to an input tensor using grid sampling.

        Args:
            mode (str): Interpolation mode to use ('bilinear' or 'nearest')
        """
        super().__init__()
        self.mode = mode

    def forward(self, f_pri, deformation_field):
        """
        Args:
            f_pri (Tensor): Prior feature map of shape [B, C, H, W]
            deformation_field (Tensor): Flow field of shape [B, 2, H, W] (dx, dy)

        Returns:
            Tensor: Warped feature map of shape [B, C, H, W]
        """
        B, _, H, W = f_pri.shape

        # Generate identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=f_pri.device),
            torch.arange(W, device=f_pri.device),
            indexing="ij"
        )
        grid = torch.stack((grid_y, grid_x), dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)           # [B, 2, H, W]

        # Add deformation
        new_grid = grid + deformation_field.to(f_pri.device)  # [B, 2, H, W]

        # Normalize to [-1, 1]
        new_grid[:, 0, :, :] = 2.0 * (new_grid[:, 0, :, :] / (H - 1) - 0.5)
        new_grid[:, 1, :, :] = 2.0 * (new_grid[:, 1, :, :] / (W - 1) - 0.5)

        # Reshape to [B, H, W, 2] and flip last dim to match grid_sample format
        new_grid = new_grid.permute(0, 2, 3, 1)[..., [1, 0]]  # [B, H, W, 2]

        # Apply spatial transformation
        f_pri_aligned = F.grid_sample(
            f_pri, new_grid, mode=self.mode, align_corners=True
        )

        return f_pri_aligned


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=5, num_steps=240):
        """
        Continuous sinusoidal positional encoding with linear interpolation over time.

        Args:
            dim (int): Dimension of the encoding.
            drop (float): Dropout rate.
            maxtime (float): Maximum time value for normalization.
            num_steps (int): Number of discrete time steps for encoding table.
        """
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.maxtime = maxtime
        self.num_steps = num_steps

        # Precompute sinusoidal encodings
        position = torch.linspace(0, maxtime, steps=num_steps).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(num_steps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        """
        Args:
            xs (Tensor): Input tensor of shape (N, B, C).
            times (Tensor): Time values of shape (B,).

        Returns:
            Tensor: Time-encoded input of shape (N, B, C).
        """
        times = torch.clamp(times, 0, self.maxtime) * (self.num_steps - 1) / self.maxtime
        t_floor = torch.floor(times).long()
        t_ceil = torch.ceil(times).long()
        alpha = (times - t_floor).unsqueeze(1)  # (B, 1)

        # Linear interpolation
        pe_floor = self.pe[t_floor]  # (B, C)
        pe_ceil = self.pe[t_ceil]  # (B, C)
        pe_interp = (1 - alpha) * pe_floor + alpha * pe_ceil  # (B, C)

        return self.dropout(xs + pe_interp.unsqueeze(0))  # (N, B, C)


class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(CumulativeProbabilityLayer, self).__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)  # could also be (num_features → max_followup)
        self.relu = nn.ReLU(inplace=True)

        # proper lower-triangular mask
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        """
        Returns:
            hazards: [B, T] probabilities per year
        """
        hazards = self.hazards(x)  # [B, T] logits
        B, T = hazards.size()

        expanded = hazards.unsqueeze(-1).expand(B, T, T)  # [B, T, T]
        masked = expanded * self.upper_triagular_mask  # [B, T, T]

        cum_logits = torch.sum(masked, dim=1) + self.base_hazard_fc(x)  # [B, T]
        return cum_logits



