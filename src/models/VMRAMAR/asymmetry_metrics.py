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

    # Warn if dimensions not divisible by latent block size
    if verbose:
        if H % latent_h != 0:
            print(f"WARNING: Height dimension {H} not divisible by latent_h={latent_h}")
        if W % latent_w != 0:
            print(f"WARNING: Width dimension {W} not divisible by latent_w={latent_w}")

    # Compute absolute difference with optional bias
    dif = torch.abs(left - right + bias_params) if bias_params is not None else torch.abs(left - right)

    # Pooling kernel size
    kernel_h, kernel_w = H // latent_h, W // latent_w
    stride_h, stride_w = (1, 1) if flexible else (kernel_h, kernel_w)

    # Max pooling over spatial blocks
    dif_pooled = F.max_pool2d(dif, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

    # Compute norm over channels
    dif_norm = torch.norm(dif_pooled, dim=1)  # shape: [B, latent_h, latent_w]

    # -------------------------
    # Select top asymmetry or top-k
    # -------------------------
    if topk is None:
        # Max across width
        max_by_h, y_idx = torch.max(dif_norm, dim=2)  # [B, latent_h]
        # Max across height
        max_asym, x_idx = torch.max(max_by_h, dim=1)  # [B]

        # Select winning y position corresponding to x
        best_y_idx = y_idx[torch.arange(B, device=y_idx.device), x_idx]

        metadata = {
            'y_argmin': best_y_idx.detach(),
            'x_argmin': x_idx.detach(),
            'heatmap': dif_norm.detach()
        }
        return max_asym, metadata

    else:
        # Flatten spatial dimensions and select top-k
        dif_flat = dif_norm.view(B, -1)
        topk_vals, topk_indices = torch.topk(dif_flat, topk, dim=1)
        metadata = {
            'y_argmin': -1,
            'x_argmin': -1,
            'heatmap': dif_norm.detach()
        }
        return topk_vals, metadata