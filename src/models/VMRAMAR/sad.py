
import torch
import torch.nn as nn
import torch.nn.functional as F


def hybrid_asymmetry(
    left, right, latent_h=5, latent_w=5,
    flexible=False, topk=None, bias_params=None, **kwargs,
):
    """
    Compute max-pooled asymmetry map between left and right feature maps.

    Args:
        left, right: (B, C, H, W) feature maps
        latent_h/w:  output grid size (number of pooling windows)
    Returns:
        max_asym: (B,)         scalar asymmetry per sample
        dict with x_argmin (B,), y_argmin (B,), heatmap (B, latent_h, latent_w)
    """
    if bias_params is None:
        dif = torch.abs(left - right)
    else:
        dif = torch.abs(left - right + bias_params)

    kernel_h = max(dif.shape[-2] // latent_h, 1)
    kernel_w = max(dif.shape[-1] // latent_w, 1)
   
    if flexible:
        dif = F.max_pool2d(dif, (kernel_h, kernel_w), stride=(1, 1))
    else:
        dif = F.max_pool2d(dif, (kernel_h, kernel_w), stride=(kernel_h, kernel_w))

    dif = torch.norm(dif, dim=-3)   # (B, H_out, W_out)

    if topk is None:
        max_by_ftr, y_argmin = torch.max(dif, dim=-1)          # (B, H_out)
        max_asym,   x_argmin = torch.max(max_by_ftr, dim=-1)   # (B,)

        # y at the winning x position → consistent (B,) shape
        # Ensure y_argmin is at least 2D
        if y_argmin.dim() == 1 or y_argmin.shape[1] == 1:  # handle H_out=1 or B=1
            best_y = y_argmin.view(-1)
        else:
            best_y = y_argmin[torch.arange(y_argmin.shape[0], device=y_argmin.device), x_argmin]

        # ensure x_argmin is also 1D
        x_argmin = x_argmin.view(-1)
        best_y   = best_y.view(-1)
        
        return max_asym, {
            "y_argmin": best_y.detach(),
            "x_argmin": x_argmin.detach(),
            "heatmap":  dif.detach(),
        }
    else:
        topk_vals, _ = torch.topk(dif.view(dif.shape[0], -1), topk, dim=-1)
        return topk_vals, {"y_argmin": -1, "x_argmin": -1, "heatmap": dif.detach()}


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
            lt = self.bn(left_features[:, t])  if self.use_bn else left_features[:, t]
            rt = self.bn(right_features[:, t]) if self.use_bn else right_features[:, t]

            max_asym, other = hybrid_asymmetry(
                lt, rt,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                flexible=getattr(self, "flexible_asymmetry", False),
                bias_params=self.bias if self.use_bias else None,
            )
            asym_values.append(max_asym)

            # --- SAFE stacking of coordinates ---
            x_arg = other["x_argmin"]
            y_arg = other["y_argmin"]

            # Ensure shape (B, 1) for concatenation
            if x_arg.dim() == 1:
                x_arg = x_arg.unsqueeze(1)
            if y_arg.dim() == 1:
                y_arg = y_arg.unsqueeze(1)

            asym_coords.append(torch.cat([x_arg, y_arg], dim=1))
            asym_maps.append(other["heatmap"])

        return {
            "asymmetry_values": torch.stack(asym_values, dim=1),   # (B, T)
            "asymmetry_coords": torch.stack(asym_coords, dim=1),   # (B, T, 2)
            "heatmap":          torch.stack(asym_maps,   dim=1),   # (B, T, H, W)
        }