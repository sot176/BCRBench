
import torch
import torch.nn as nn
import torch.nn.functional as F


def hybrid_asymmetry(left, right, latent_h=5, latent_w=5, flexible=False, topk=None, bias_params=None):
    if bias_params is None:
        dif = torch.abs(left - right).contiguous()
    else:
        dif = torch.abs(left - right + bias_params).contiguous()

    kernel_h = max(dif.shape[-2] // latent_h, 1)
    kernel_w = max(dif.shape[-1] // latent_w, 1)

    if flexible:
        dif = F.max_pool2d(dif, (kernel_h, kernel_w), stride=(1, 1))
    else:
        dif = F.max_pool2d(dif, (kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    
    dif = dif.mean(dim=-3)  # (B, H_out, W_out)
    # Ensure dif is 3D
    if dif.dim() == 2:  # shape (B, N)
        dif = dif.unsqueeze(-1)  # (B, N, 1)
    elif dif.dim() == 1:  # shape (B,)
        dif = dif.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

    # Ensure there’s always at least one H_out, W_out
    B, H_out, W_out = dif.shape

    if topk is None:
        # Softmax over spatial map
        weights = F.softmax(dif.view(B, -1), dim=-1).view(B, H_out, W_out)

        # Coordinate grids
        x_coords = torch.arange(H_out, device=dif.device).float()
        y_coords = torch.arange(W_out, device=dif.device).float()

        # Expected coordinates (soft-argmax)
        x_mean = (weights.sum(dim=2) * x_coords).sum(dim=1)  # (B,)
        y_mean = (weights.sum(dim=1) * y_coords).sum(dim=1)  # (B,)

        # Soft asymmetry score
        max_asym = (dif * weights).sum(dim=(1, 2))  # (B,)

        return max_asym, {
            "x_argmin": x_mean,   # now soft coordinates
            "y_argmin": y_mean,
            "heatmap": dif        # keep gradients!
        }
    else:
        topk_vals, _ = torch.topk(dif.view(B, -1), topk, dim=-1)
        return topk_vals, {"x_argmin": torch.zeros(B, dtype=torch.long), "y_argmin": torch.zeros(B, dtype=torch.long), "heatmap": dif.detach()}


class SpatialAsymmetryDetector(nn.Module):
    """
    SAD: processes left/right feature maps per timestep using hybrid_asymmetry.

    Input:  left, right each (B, T, C, H, W)
    Output: dict with
        asymmetry_values: (B, T)
        asymmetry_coords: (B, T, 2)
        heatmap:          (B, T, latent_h, latent_w)
    """

    def __init__(self, args):
        super().__init__()
        self.feature_dim = 512
        self.latent_h    = getattr(args, "latent_h", 5)
        self.latent_w    = getattr(args, "latent_w", 5)

        self.use_bias = getattr(args, "use_sad_bias", False)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.feature_dim, 1, 1))

        self.use_bn = getattr(args, "use_sad_bn", False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.feature_dim)

    def forward(self, left_features, right_features):
        """
        left_features:  (B, T, C, H, W)
        right_features: (B, T, C, H, W)
        """
        B, T = left_features.shape[:2]
        asym_values, asym_coords, asym_maps = [], [], []

        for t in range(T):
            lt = left_features[:, t].clone()
            rt = right_features[:, t].clone()

            if self.use_bn:
                lt = self.bn(lt)
                rt = self.bn(rt)

            max_asym, other = hybrid_asymmetry(
                lt.clone(), rt.clone(),
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                flexible=getattr(self, "flexible_asymmetry", False),
                bias_params=self.bias if self.use_bias else None,
            )

            # --- Ensure max_asym is (B,) ---
            if max_asym.dim() != 1:
                max_asym = max_asym.view(B)

            # --- SAFE stacking of coordinates ---
            x_arg = other["x_argmin"]
            y_arg = other["y_argmin"]

            # Ensure x_arg/y_arg are (B, 1)
            if x_arg.dim() == 0:
                x_arg = x_arg.unsqueeze(0)
            if x_arg.dim() == 1:
                x_arg = x_arg.unsqueeze(1)
            if y_arg.dim() == 0:
                y_arg = y_arg.unsqueeze(0)
            if y_arg.dim() == 1:
                y_arg = y_arg.unsqueeze(1)

            asym_coords.append(torch.cat([x_arg, y_arg], dim=1))

            # --- SAFE heatmap shape ---
            heatmap = other["heatmap"]
            if heatmap.dim() == 2:   # (H_out, W_out) → add batch dim
                heatmap = heatmap.unsqueeze(0)
            asym_maps.append(heatmap)

            asym_values.append(max_asym)

        return {
            "asymmetry_values": torch.stack(asym_values, dim=1),   # (B, T)
            "asymmetry_coords": torch.stack(asym_coords, dim=1),   # (B, T, 2)
            "heatmap":          torch.stack(asym_maps,   dim=1),   # (B, T, latent_h, latent_w)
        }