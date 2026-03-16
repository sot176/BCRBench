import torch
import torch.nn as nn
import torch.nn.functional as F
from .asymmetry_metrics import hybrid_asymmetry
from models.Mirai.onconet.models.factory import RegisterModel
import torch
import torch.nn as nn
import torch.nn.functional as F

@RegisterModel("sad")
class SpatialAsymmetryDetector(nn.Module):
    """
    Spatial Asymmetry Detector (SAD) module that uses the existing hybrid_asymmetry function
    to detect asymmetries between left and right views.
    """
    def __init__(self, args):
        super(SpatialAsymmetryDetector, self).__init__()
        self.args = args
        self.feature_dim = 512
        self.latent_h = getattr(args, "latent_h", 64)
        self.latent_w = getattr(args, "latent_w", 52)

        # Optional bias term
        self.use_bias = getattr(args, "use_sad_bias", True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, self.feature_dim, 1, 1))

        # Optional batch normalization
        self.use_bn = getattr(args, "use_sad_bn", True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.feature_dim)

    def forward(self, left_features, right_features):
        """
        Process left and right feature maps using hybrid_asymmetry function.
        
        Args:
            left_features: Tensor of shape (B, T, C, H, W)
            right_features: Tensor of shape (B, T, C, H, W)
        """
        batch_size, time_steps = left_features.shape[:2]
        
        asymmetry_values = []
        asymmetry_coords = []
        asymmetry_maps = []
        
        # Process each timestep
        for t in range(time_steps):
            # Apply batch norm if enabled
            if self.use_bn:
                left_t = self.bn(left_features[:, t])
                right_t = self.bn(right_features[:, t])
            else:
                left_t = left_features[:, t]
                right_t = right_features[:, t]
            
            # Use hybrid_asymmetry function directly
            max_asym, other = hybrid_asymmetry(
                left_t, 
                right_t,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                flexible=getattr(self.args, "flexible_asymmetry", False),
                bias_params=self.bias if self.use_bias else None
            )
            
            asymmetry_values.append(max_asym)
            asymmetry_coords.append(torch.stack([other['x_argmin'], other['y_argmin']], dim=1))
            asymmetry_maps.append(other['heatmap'])
        
        return {
            'asymmetry_values': torch.stack(asymmetry_values, dim=1),  # (B, T)
            'asymmetry_coords': torch.stack(asymmetry_coords, dim=1),  # (B, T, 2)
            'heatmap': torch.stack(asymmetry_maps, dim=1)  # (B, T, H, W)
        } 