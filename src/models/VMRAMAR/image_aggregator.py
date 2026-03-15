import torch
import torch.nn as nn

class ImageAggregator(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.view_fc = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B,T,V,L,C)

        B,T,V,L,C = x.shape

        x = x.view(B*T*L, V, C)

        x = self.view_fc(x)

        x,_ = self.attn(x,x,x)

        x = self.out(x)

        x = x.mean(dim=1)   # fuse views

        x = x.view(B,T,L,C)

        return x