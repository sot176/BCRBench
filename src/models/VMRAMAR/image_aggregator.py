import torch
import torch.nn as nn


class ImageAggregator(nn.Module):
    """
    Per the diagram: split CC/MLO → FC per view → cross-view attention → FC out.
    Input:  (B, T, V, C, H, W)
    Output: (B, T, C, H, W)  — one fused spatial map per visit
    """
    def __init__(self, dim):
        super().__init__()
        # Separate FC per view (diagram shows split → FC → FC)
        self.cc_fc  = nn.Linear(dim, dim)
        self.mlo_fc = nn.Linear(dim, dim)
        # Cross-view attention
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.out  = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, V, C, H, W)
        B, T, V, C, H, W = x.shape

        # Global pool spatial dims first — aggregation is over embeddings
        x_pooled = x.mean(dim=(-2, -1))           # (B, T, V, C)

        # Split CC (views 0,1) and MLO (views 2,3), average each pair
        cc  = x_pooled[:, :, :V//2].mean(dim=2)   # (B, T, C)
        mlo = x_pooled[:, :, V//2:].mean(dim=2)   # (B, T, C)

        # Per-view FC
        cc  = self.cc_fc(cc)                       # (B, T, C)
        mlo = self.mlo_fc(mlo)                     # (B, T, C)

        # Stack as sequence for cross-view attention
        views = torch.stack([cc, mlo], dim=2)      # (B, T, 2, C)
        BT = B * T
        views = views.view(BT, 2, C)
        views, _ = self.attn(views, views, views)  # (BT, 2, C)
        views = views.mean(dim=1).view(B, T, C)    # (B, T, C)
        out = self.out(views)                      # (B, T, C)

        # Broadcast back to spatial dims for VMRNN
        out = out.unsqueeze(-1).unsqueeze(-1).expand(B, T, C, H, W)
        return out                                 # (B, T, C, H, W)