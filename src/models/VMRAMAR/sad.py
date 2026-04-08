import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


def hybrid_asymmetry(
    left: torch.Tensor,
    right: torch.Tensor,
    latent_h: int = 5,
    latent_w: int = 5,
    flexible: bool = False,
    topk: Optional[int] = None,
    bias_params: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute hybrid asymmetry between left and right feature maps.

    Args:
        left:  (B, C, H, W)
        right: (B, C, H, W)
        latent_h: target latent height
        latent_w: target latent width
        flexible: use stride=1 pooling instead of block pooling
        topk: if specified, return topk max values instead of softmax
        bias_params: optional learnable bias to add before abs

    Returns:
        max_asym: (B,)
        details: dict with keys
            x_argmin, y_argmin: soft coordinates (B,)
            heatmap: pooled difference map (B, H_out, W_out)
    """
    dif = torch.abs(left - right + bias_params) if bias_params is not None else torch.abs(left - right)
    dif = dif.contiguous()

    kernel_h = max(dif.shape[-2] // latent_h, 1)
    kernel_w = max(dif.shape[-1] // latent_w, 1)

    # Max pooling
    stride = (1, 1) if flexible else (kernel_h, kernel_w)
    dif = F.max_pool2d(dif, kernel_size=(kernel_h, kernel_w), stride=stride)

    # Reduce channel dimension
    dif = dif.mean(dim=1)  # (B, H_out, W_out)
    B, H_out, W_out = dif.shape

    if topk is None:
        # Softmax over spatial dimensions for differentiable argmax
        weights = F.softmax(dif.view(B, -1), dim=-1).view(B, H_out, W_out)

        x_coords = torch.arange(H_out, device=dif.device).float()
        y_coords = torch.arange(W_out, device=dif.device).float()

        x_mean = (weights.sum(dim=2) * x_coords).sum(dim=1)
        y_mean = (weights.sum(dim=1) * y_coords).sum(dim=1)
        max_asym = (dif * weights).sum(dim=(1, 2))
        return max_asym, {"x_argmin": x_mean, "y_argmin": y_mean, "heatmap": dif}
    else:
        topk_vals, _ = torch.topk(dif.view(B, -1), topk, dim=-1)
        return topk_vals, {"x_argmin": torch.zeros(B, device=dif.device), "y_argmin": torch.zeros(B, device=dif.device), "heatmap": dif.detach()}


class SpatialAsymmetryDetector(nn.Module):
    """
    Detect spatial asymmetry between left/right features over multiple timesteps.

    Input:
        left_features:  (B, T, C, H, W)
        right_features: (B, T, C, H, W)
    Output:
        dict with
            asymmetry_values: (B, T)
            asymmetry_coords: (B, T, 2)
            heatmap: (B, T, H_out, W_out)
    """

    def __init__(self, args):
        super().__init__()
        self.feature_dim = getattr(args, "feature_dim", 512)
        self.latent_h = getattr(args, "latent_h", 5)
        self.latent_w = getattr(args, "latent_w", 5)
        self.flexible = getattr(args, "flexible_asymmetry", False)

        self.use_bias = getattr(args, "use_sad_bias", False)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.feature_dim, 1, 1))

        self.use_bn = getattr(args, "use_sad_bn", False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.feature_dim)

    def forward(self, left_features: torch.Tensor, right_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = left_features.shape
        asym_values, asym_coords, asym_maps = [], [], []

        for t in range(T):
            lt = left_features[:, t]
            rt = right_features[:, t]

            if self.use_bn:
                lt = self.bn(lt)
                rt = self.bn(rt)

            max_asym, other = hybrid_asymmetry(
                lt, rt,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                flexible=self.flexible,
                bias_params=self.bias if self.use_bias else None,
            )

            # Safe stacking
            x_arg = other["x_argmin"].unsqueeze(-1) if other["x_argmin"].dim() == 1 else other["x_argmin"]
            y_arg = other["y_argmin"].unsqueeze(-1) if other["y_argmin"].dim() == 1 else other["y_argmin"]

            asym_coords.append(torch.cat([x_arg, y_arg], dim=-1))
            asym_maps.append(other["heatmap"])
            asym_values.append(max_asym)

        return {
            "asymmetry_values": torch.stack(asym_values, dim=1),  # (B, T)
            "asymmetry_coords": torch.stack(asym_coords, dim=1),  # (B, T, 2)
            "heatmap": torch.stack(asym_maps, dim=1),             # (B, T, H_out, W_out)
        }