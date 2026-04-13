import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LongitudinalAsymmetryTracker(nn.Module):
    """
    Tracks longitudinal asymmetry features across timepoints and fuses them
    into a single representation.

    Inputs:
        asymmetry_features: (B, T, D)   - Feature vectors per timepoint
        asymmetry_coords:   (B, T, 2)   - Coordinates of asymmetry maxima
        asymmetry_maps:     (B, T, H, W) - Full asymmetry maps (optional, for debugging)
    
    Output:
        fused_features: (B, D) - Aggregated feature vector
    """

    def __init__(self, args):
        super().__init__()
        self.feature_dim = getattr(args, "latent_dim", 512)
        self.latent_h = getattr(args, "latent_h", 5)
        self.latent_w = getattr(args, "latent_w", 5)
        self.displacement_threshold = getattr(args, "displacement_threshold", 0.4) * min(self.latent_h, self.latent_w)
        self.max_persistence = getattr(args, "max_persistence", 5.0)
        self.lat_dropout = getattr(args, "lat_dropout", 0.1)
        self.temperature = getattr(args, "persistence_temperature", 2.0)
        self.use_bn = getattr(args, "use_lat_bn", False)

        # Feature transform MLP
        self.feature_transform = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.lat_dropout),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.feature_dim)

        # Learned output normalization
        self.out_norm = nn.LayerNorm(self.feature_dim)

    def compute_displacement(self, coords1, coords2):
        """
        Compute normalized L2 displacement between coordinate sets.
        """
        norm_factor = torch.tensor([self.latent_h, self.latent_w], device=coords1.device, dtype=coords1.dtype)
        coords1_norm = coords1 / norm_factor
        coords2_norm = coords2 / norm_factor
        return torch.norm(coords1_norm - coords2_norm, dim=-1)  # (B,)

    def forward(
        self,
        asymmetry_features,
        asymmetry_coords,
        asymmetry_maps = None
    ):
        """
        Forward pass to compute fused longitudinal asymmetry features.
        """
        B, T, D = asymmetry_features.shape
        device = asymmetry_features.device

        # ── Initialize persistence weights ──────────────────────
        persistence_weights = torch.ones(B, T, device=device)

        # Vectorized displacement comparison across timepoints
        if T > 1:
            for t in range(1, T):
                disp = self.compute_displacement(asymmetry_coords[:, t], asymmetry_coords[:, t - 1])
                is_persistent = disp < self.displacement_threshold
                persistence_weights[:, t] = torch.where(
                    is_persistent,
                    torch.clamp(persistence_weights[:, t - 1] + 1.0, max=self.max_persistence),
                    torch.ones_like(persistence_weights[:, t])
                )

        # ── Transform features ────────────────────────────────
        transformed = self.feature_transform(asymmetry_features)  # (B, T, D)

        # Optional batch normalization
        if self.use_bn:
            # BN expects (B*T, D)
            transformed = self.bn(transformed.view(B * T, D)).view(B, T, D)

        # LayerNorm per feature
        transformed = F.layer_norm(transformed, normalized_shape=(D,))

        # ── Weighted aggregation ──────────────────────────────
        weights = F.softmax(persistence_weights / self.temperature, dim=1).unsqueeze(-1)  # (B, T, 1)
        fused = (transformed * weights).sum(dim=1)  # (B, D)

        return self.out_norm(fused)