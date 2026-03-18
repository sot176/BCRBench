
import torch
import torch.nn as nn


class ImageAggregator(nn.Module):
    """
    Input:  (B, T, V, C)  — pooled embeddings, V views
    Output: (B, T, C)     — one fused embedding per visit
    """

    def __init__(self, dim: int, num_views: int = 4):
        super().__init__()
        # Separate FC per view (diagram: split → FC → attention → FC)
        self.view_fcs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_views)
        ])
        self.attn   = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.out_fc = nn.Linear(dim, dim)
        self.num_views = num_views

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, V, C)
        Returns: (B, T, C)
        """
        B, T, V, C = x.shape

        # Apply separate FC to each view
        view_feats = torch.stack([
            self.view_fcs[v](x[:, :, v, :]) for v in range(V)
        ], dim=2)                                       # (B, T, V, C)

        # Attention over views at each timestep
        BT = B * T
        view_feats = view_feats.view(BT, V, C)
        attn_out, _ = self.attn(view_feats, view_feats, view_feats)  # (BT, V, C)

        # Mean pool views → single embedding per visit
        fused = attn_out.mean(dim=1)                   # (BT, C)
        fused = self.out_fc(fused)                     # (BT, C)
        return fused.view(B, T, C)                     # (B, T, C)