import torch
import torch.nn as nn

class ImageAggregator(nn.Module):
    """
    Fuses N spatial feature maps (one per mammogram view) via
    per-location attention across views.

    Input:  (B, N, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.fc   = nn.Linear(dim, dim)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape

        # Treat every spatial location as an independent batch element.
        # Reshape so attention runs over the N views at each (h,w) position.
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, N, C)
        x = x.view(B * H * W, N, C)                 # (B·H·W, N, C)

        residual = x
        x, _ = self.attn(x, x, x)                   # attend over N views
        x = self.norm(x + residual)
        x = self.act(self.fc(x))

        x = x.mean(dim=1)                            # (B·H·W, C)  — fuse views
        x = x.view(B, H, W, C)                       # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)
        return x