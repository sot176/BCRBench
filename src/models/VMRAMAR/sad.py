import torch
import torch.nn as nn
import torch.nn.functional as F

from .asymmetry_metrics import hybrid_asymmetry


class SpatialAsymmetryDetector(nn.Module):
    def __init__(
        self,
        latent_h: int = 5,
        latent_w: int = 5,
        flexible: bool = False,
        embedding_channel: int = 512,
        initial_asym_mean: float = 8_000_000.0,
        initial_asym_std: float = 1_520_381.0,
    ) -> None:
        super().__init__()
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.flexible = flexible

        self.cc_stretch_params = nn.Parameter(torch.ones(embedding_channel))
        self.mlo_stretch_params = nn.Parameter(torch.ones(embedding_channel))

        self.learned_asym_mean = nn.Parameter(
            torch.tensor(float(initial_asym_mean), dtype=torch.float32)
        )
        self.learned_asym_std = nn.Parameter(
            torch.tensor(float(initial_asym_std), dtype=torch.float32)
        )

        self.pair_definitions = (
            (0, 1, "CC"),
            (2, 3, "MLO"),
        )

    def _stretch(self, tensor: torch.Tensor, view_name: str) -> torch.Tensor:
        params = self.cc_stretch_params if view_name == "CC" else self.mlo_stretch_params
        return tensor * params.view(1, -1, 1, 1)

    def forward(
        self,
        feature_maps: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_maps: Tensor of shape (B, T, V, C, H, W)
            view_mask: Tensor of shape (B, T, V)

        Returns:
            exam_scores:     (B, T)
            dominant_coords: (B, T, 2)
            coord_valid:     (B, T)
        """
        if feature_maps.dim() != 6:
            raise ValueError(
                f"SAD expects feature_maps with shape (B, T, V, C, H, W), got {feature_maps.shape}."
            )

        if view_mask.dim() != 3:
            raise ValueError(
                f"SAD expects view_mask with shape (B, T, V), got {view_mask.shape}."
            )

        batch_size, time_steps, num_views, channels, _, _ = feature_maps.shape

        if num_views < 4:
            raise ValueError(f"SAD expects at least 4 views, got V={num_views}.")

        if view_mask.shape != (batch_size, time_steps, num_views):
            raise ValueError(
                f"view_mask shape must be {(batch_size, time_steps, num_views)}, got {view_mask.shape}."
            )

        raw_pair_scores = []
        normalized_pair_scores = []
        pair_coords = []
        pair_valid = []

        for left_index, right_index, view_name in self.pair_definitions:
            left = feature_maps[:, :, left_index]
            right = torch.flip(feature_maps[:, :, right_index], dims=[-1])
            valid = view_mask[:, :, left_index] & view_mask[:, :, right_index]

            flat_left = left.reshape(batch_size * time_steps, *left.shape[2:])
            flat_right = right.reshape(batch_size * time_steps, *right.shape[2:])
            flat_valid = valid.reshape(batch_size * time_steps)

            flat_raw_scores = flat_left.new_zeros(batch_size * time_steps)
            flat_normalized_scores = flat_left.new_zeros(batch_size * time_steps)
            flat_coords = flat_left.new_zeros(batch_size * time_steps, 2)

            if flat_valid.any():
                valid_left = self._stretch(flat_left[flat_valid], view_name)
                valid_right = self._stretch(flat_right[flat_valid], view_name)

                valid_scores, details = hybrid_asymmetry(
                    valid_left,
                    valid_right,
                    latent_h=self.latent_h,
                    latent_w=self.latent_w,
                    flexible=self.flexible,
                )

                normalized = (
                    valid_scores - self.learned_asym_mean
                ) / self.learned_asym_std.abs().clamp(min=1e-6)

                flat_raw_scores[flat_valid] = valid_scores.to(flat_raw_scores.dtype)
                flat_normalized_scores[flat_valid] = torch.sigmoid(normalized).to(
                    flat_normalized_scores.dtype
                )

                flat_coords[flat_valid, 0] = details["y_argmax"].to(flat_coords.dtype)
                flat_coords[flat_valid, 1] = details["x_argmax"].to(flat_coords.dtype)

            raw_pair_scores.append(flat_raw_scores.reshape(batch_size, time_steps))
            normalized_pair_scores.append(
                flat_normalized_scores.reshape(batch_size, time_steps)
            )
            pair_coords.append(flat_coords.reshape(batch_size, time_steps, 2))
            pair_valid.append(valid)

        raw_scores = torch.stack(raw_pair_scores, dim=-1)
        normalized_scores = torch.stack(normalized_pair_scores, dim=-1)
        coords = torch.stack(pair_coords, dim=-2)
        valid = torch.stack(pair_valid, dim=-1)

        valid_float = valid.float()
        exam_scores = (normalized_scores * valid_float).sum(dim=-1) / valid_float.sum(
            dim=-1
        ).clamp(min=1.0)

        dominant_pair = raw_scores.masked_fill(~valid, float("-inf")).argmax(dim=-1)

        dominant_coords = coords.gather(
            dim=2,
            index=dominant_pair.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, time_steps, 1, 2),
        ).squeeze(2)

        coord_valid = valid.any(dim=-1)
        dominant_coords = dominant_coords * coord_valid.unsqueeze(-1).float()

        return exam_scores, dominant_coords, coord_valid
