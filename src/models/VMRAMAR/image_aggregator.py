import torch
import torch.nn as nn

class ImageAggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.view_fc = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, V, C, H, W)
        B, T, V, C, H, W = x.shape

        # Flatten views dimension for attention
        x = x.permute(0,1,4,5,2,3).contiguous()  # (B, T, H, W, V, C)
        x = x.view(B*T*H*W, V, C)               # batch= B*T*H*W, sequence=V, embed=C

        # Attention over views
        x = self.view_fc(x)
        x, _ = self.attn(x, x, x)
        x = self.out(x)

        # Fuse views
        x = x.mean(dim=1)  # (B*T*H*W, C)

        # Reshape back to (B, T, C, H, W)
        x = x.view(B, T, H, W, C).permute(0,1,4,2,3).contiguous()
        return x  # (B, T, C, H, W)