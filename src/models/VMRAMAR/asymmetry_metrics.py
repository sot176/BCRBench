import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

def hybrid_asymmetry(
    left: torch.Tensor,
    right: torch.Tensor,
    latent_h = 5,
    latent_w = 5,
    verbose = False,
    flexible = False,
    topk = None,
    bias_params = None,
    **kwargs
):
    """
    Compute a hybrid asymmetry metric between two feature maps.

    Args:
        left (Tensor): Left feature map, shape [B, C, H, W].
        right (Tensor): Right feature map, shape [B, C, H, W].
        latent_h (int): Number of latent blocks in height.
        latent_w (int): Number of latent blocks in width.
        verbose (bool): Print warnings if H or W not divisible by latent_h/latent_w.
        flexible (bool): Use stride=1 max pooling instead of non-overlapping.
        topk (int, optional): Return top-k asymmetry values instead of maximum.
        bias_params (Tensor, optional): Optional bias added before computing difference.
    
    Returns:
        Tuple containing:
            - asymmetry scores (Tensor)
            - metadata dictionary with indices and heatmap
    """
    B, C, H, W = left.shape
    if not isinstance(latent_h, int):
        latent_h = getattr(latent_h, "latent_h", latent_h)
    if not isinstance(latent_w, int):
        latent_w = getattr(latent_w, "latent_w", latent_w)

    latent_h = int(latent_h)
    latent_w = int(latent_w)
    diff = torch.abs(left - right if bias_params is None else left - right + bias_params)
    kernel_h = max(diff.shape[-2] // latent_h, 1)
    kernel_w = max(diff.shape[-1] // latent_w, 1)
    stride = (1, 1) if flexible else (kernel_h, kernel_w)
    diff = F.max_pool2d(diff, (kernel_h, kernel_w), stride=stride)
    diff = torch.norm(diff, dim=-3)

    # -------------------------
    # Select top asymmetry or top-k
    # -------------------------
    if topk is None:
        max_by_row, x_indices = torch.max(diff, dim=-1)
        max_scores, y_indices = torch.max(max_by_row, dim=-1)
        x_indices = x_indices.gather(1, y_indices.unsqueeze(-1)).squeeze(-1)
        return max_scores, {
            "y_argmax": y_indices.detach(),
            "x_argmax": x_indices.detach(),
            "heatmap": diff.detach(),
        }

    topk_scores, _ = torch.topk(diff.view(diff.shape[0], -1), topk, dim=-1)
    return topk_scores, {
        "y_argmax": torch.full((diff.shape[0],), -1, device=diff.device, dtype=torch.long),
        "x_argmax": torch.full((diff.shape[0],), -1, device=diff.device, dtype=torch.long),
        "heatmap": diff.detach(),
    }