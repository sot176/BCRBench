
import torch
import torch.nn as nn
import torch.nn.functional as F


class LongitudinalAsymmetryTracker(nn.Module):
    """
    Input:
        asymmetry_features: (B, T, 512)
        asymmetry_coords:   (B, T, 2)
        asymmetry_maps:     (B, T, H, W)
    Output: (B, 512)
    """

    def __init__(self, args):
        super().__init__()
        self.feature_dim = 512
        self.latent_h    = getattr(args, "latent_h", 5)
        self.latent_w    = getattr(args, "latent_w", 5)
        self.displacement_threshold = 0.4 * min(self.latent_h, self.latent_w)

        self.feature_transform = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(p=getattr(args, "lat_dropout", 0.1)),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.use_bn = getattr(args, "use_lat_bn", True)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.feature_dim)

        # Learned output normalisation instead of hardcoded mean/std
        self.out_norm = nn.LayerNorm(self.feature_dim)

    def compute_displacement(self, coords1, coords2, scale_factors):
        return torch.norm(
            coords1.float() * scale_factors - coords2.float() * scale_factors,
            dim=-1,
        )

    def forward(self, asymmetry_features, asymmetry_coords, asymmetry_maps):
        B, T, D = asymmetry_features.shape

        scale_h = asymmetry_maps.shape[-2] / self.latent_h
        scale_w = asymmetry_maps.shape[-1] / self.latent_w
        scale_factors = torch.tensor(
            [scale_h, scale_w], device=asymmetry_features.device
        )

        # Persistence weights — upweight regions that persist across visits
        persistence_weights = torch.ones(B, T, device=asymmetry_features.device)
        for t in range(1, T):
            disp = self.compute_displacement(
                asymmetry_coords[:, t], asymmetry_coords[:, t - 1], scale_factors
            )
            is_persistent = disp < self.displacement_threshold
            persistence_weights[:, t] = torch.where(
                is_persistent,
                persistence_weights[:, t - 1] + 1.0,
                torch.ones_like(persistence_weights[:, t]),
            )

        # Transform features
        transformed = self.feature_transform(asymmetry_features)
        if self.use_bn:
            transformed = self.bn(transformed.transpose(1, 2)).transpose(1, 2)

        # Weighted aggregation
        weights  = F.softmax(persistence_weights, dim=1).unsqueeze(-1)
        fused    = (transformed * weights).sum(dim=1)  # (B, 512)
        return self.out_norm(fused)