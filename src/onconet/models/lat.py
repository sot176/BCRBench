import torch
import torch.nn as nn
import torch.nn.functional as F
from onconet.models.factory import RegisterModel

@RegisterModel("lat")
class LongitudinalAsymmetryTracker(nn.Module):
    """
    Longitudinal Asymmetry Tracker (LAT) module that tracks temporal evolution
    of asymmetric regions using existing alignment and normalization approaches.
    """
    def __init__(self, args):
        super(LongitudinalAsymmetryTracker, self).__init__()
        self.args = args
        self.feature_dim = 512
        
        # Use same latent dimensions as hybrid_asymmetry
        self.latent_h = getattr(args, "latent_h", 64)
        self.latent_w = getattr(args, "latent_w", 52)
        self.displacement_threshold = 0.4 * min(self.latent_h, self.latent_w)
        
        # Feature transformation with dropout
        self.feature_transform = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(p=getattr(args, "lat_dropout", 0.1)),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Use same normalization parameters as LocalizedDifModel
        self.initial_asym_mean = getattr(args, "initial_asym_mean", 2000)
        self.initial_asym_std = getattr(args, "initial_asym_std", 300)
        
        self.use_bn = getattr(args, "use_lat_bn", True)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.feature_dim)

    def compute_displacement(self, coords1, coords2, scale_factors):
        """
        Compute normalized Euclidean displacement between coordinates.
        Uses same scaling approach as embed_explore's alignment functions.
        """
        coords1_scaled = coords1.float() * scale_factors
        coords2_scaled = coords2.float() * scale_factors
        return torch.norm(coords1_scaled - coords2_scaled, dim=-1)

    def forward(self, asymmetry_features, asymmetry_coords, asymmetry_maps):
        """
        Track asymmetry evolution over time using persistence weights.
        
        Args:
            asymmetry_features: Tensor of shape (B, T, D) containing asymmetry features
            asymmetry_coords: Tensor of shape (B, T, 2) containing coordinates of max asymmetry
            asymmetry_maps: Tensor of shape (B, T, H, W) containing asymmetry heatmaps
        """
        batch_size, time_steps, feat_dim = asymmetry_features.shape
        
        # Calculate scale factors for coordinate comparison
        scale_h = asymmetry_maps.shape[-2] / self.latent_h
        scale_w = asymmetry_maps.shape[-1] / self.latent_w
        scale_factors = torch.tensor([scale_h, scale_w], device=asymmetry_features.device)
        
        # Initialize persistence weights
        persistence_weights = torch.ones(batch_size, time_steps, device=asymmetry_features.device)
        
        # Track displacement between consecutive timepoints
        for t in range(1, time_steps):
            displacement = self.compute_displacement(
                asymmetry_coords[:, t],
                asymmetry_coords[:, t-1],
                scale_factors
            )
            
            # Update weights based on persistence
            is_persistent = displacement < self.displacement_threshold
            persistence_weights[:, t] = torch.where(
                is_persistent,
                persistence_weights[:, t-1] + 1.0,
                torch.ones_like(persistence_weights[:, t])
            )
        
        # Transform and normalize features
        transformed_features = self.feature_transform(asymmetry_features)
        if self.use_bn:
            transformed_features = self.bn(transformed_features.transpose(1, 2)).transpose(1, 2)
        
        # Apply temporal weighting
        persistence_weights = F.softmax(persistence_weights, dim=1)
        weighted_features = transformed_features * persistence_weights.unsqueeze(-1)
        
        # Use same normalization as LocalizedDifModel
        fused_features = weighted_features.sum(dim=1)
        normalized_features = (fused_features - self.initial_asym_mean) / self.initial_asym_std
        
        return normalized_features 