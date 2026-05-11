import torch
import torch.nn as nn

from .asymmetry_metrics import hybrid_asymmetry


class SpatialAsymmetryDetector(nn.Module):
    """
    Spatial Asymmetry Detector (SAD) module that uses hybrid_asymmetry
    to detect asymmetries between left and right views.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_dim = int(getattr(args, "feature_dim", 512))
        self.latent_h = int(getattr(args, "latent_h", 64))
        self.latent_w = int(getattr(args, "latent_w", 52))

        self.use_bias = getattr(args, "use_sad_bias", True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.feature_dim, 1, 1))
        else:
            self.bias = None

        self.use_bn = getattr(args, "use_sad_bn", True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.feature_dim)

    def forward(self, left_features, right_features):
        """
        Args:
            left_features: Tensor of shape (B, T, C, H, W)
            right_features: Tensor of shape (B, T, C, H, W)
        """
        if left_features.dim() != 5 or right_features.dim() != 5:
            raise ValueError(
                "SpatialAsymmetryDetector expects left/right tensors with shape "
                f"(B, T, C, H, W), got {left_features.shape} and {right_features.shape}."
            )

        if left_features.shape != right_features.shape:
            raise ValueError(
                f"Left/right feature shapes must match, got "
                f"{left_features.shape} and {right_features.shape}."
            )

        batch_size, time_steps, channels = left_features.shape[:3]

        if channels != self.feature_dim:
            raise ValueError(
                f"SAD was initialized with feature_dim={self.feature_dim}, "
                f"but received feature maps with C={channels}."
            )

        asymmetry_values = []
        asymmetry_coords = []
        asymmetry_maps = []

        for t in range(time_steps):
            left_t = left_features[:, t]
            right_t = right_features[:, t]

            if self.use_bn:
                left_t = self.bn(left_t)
                right_t = self.bn(right_t)

            max_asym, other = hybrid_asymmetry(
                left_t,
                right_t,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                flexible=getattr(self.args, "flexible_asymmetry", False),
                bias_params=self.bias,
            )

            asymmetry_values.append(max_asym)

            if "x_argmin" in other and "y_argmin" in other:
                asymmetry_coords.append(
                    torch.stack([other["x_argmin"], other["y_argmin"]], dim=1)
                )

            if "heatmap" in other:
                asymmetry_maps.append(other["heatmap"])

        output = {
            "asymmetry_values": torch.stack(asymmetry_values, dim=1),
        }

        if len(asymmetry_coords) == time_steps:
            output["asymmetry_coords"] = torch.stack(asymmetry_coords, dim=1)

        if len(asymmetry_maps) == time_steps:
            output["heatmap"] = torch.stack(asymmetry_maps, dim=1)

        return output
