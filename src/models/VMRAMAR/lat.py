import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LongitudinalAsymmetryTracker(nn.Module):
    def __init__(self, threshold_ratio: float = 0.4, persistent_weight: float = 1.0) -> None:
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.persistent_weight = persistent_weight

    def forward(
        self,
        scores: torch.Tensor,
        coords: torch.Tensor,
        coord_valid: torch.Tensor,
        exam_mask: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        valid = exam_mask & coord_valid
        if not valid.any():
            return scores.new_zeros(scores.size(0))

        threshold = self.threshold_ratio * float(window_size)
        weights = valid.float()
        for step in range(1, scores.size(1)):
            persistent = valid[:, step] & valid[:, step - 1]
            displacement = torch.norm(coords[:, step] - coords[:, step - 1], dim=-1)
            persistent = persistent & (displacement <= threshold)
            persistent_weight = persistent.float() * self.persistent_weight
            weights[:, step] += persistent_weight
            weights[:, step - 1] += persistent_weight

        weighted_scores = scores * weights
        return weighted_scores.sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
