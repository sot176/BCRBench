import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict


# -------------------------
# Convolutional Blocks
# -------------------------
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Feedforward(nn.Module):
    """Two stacked ConvBlocks."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels)
        self.block2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


# -------------------------
# POE Latent Module
# -------------------------
class POELatent(nn.Module):
    """
    Probabilistic latent embedding with optional stochastic sampling.
    Adapted from: https://github.com/Li-Wanhua/POEs
    """
    def __init__(self, num_feat: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(num_feat, num_feat)
        self.log_var = nn.Linear(num_feat, num_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, max_t: int = 50, use_sto: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, num_feat]
            max_t: Number of stochastic samples
            use_sto: Whether to apply stochastic sampling
        Returns:
            drop_emb: Tensor after dropout and optional stochastic sampling
            emb: Deterministic embedding
            log_var: Log-variance
        """
        emb = self.embed(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5 * log_var)

        if use_sto:
            rep_emb = emb.unsqueeze(0).expand(max_t, *emb.shape)
            rep_std = std.unsqueeze(0).expand(max_t, *std.shape)
            noise = torch.randn_like(rep_emb)
            drop_emb = self.dropout(rep_emb + rep_std * noise)
        else:
            drop_emb = self.dropout(emb)

        return drop_emb, emb, log_var


# -------------------------
# Baseline Backbone
# -------------------------
class BaselineModel(nn.Module):
    """
    Generic CNN backbone extractor (ResNet, VGG, DenseNet, ConvNext, EfficientNet).
    Removes final pooling/classifier layers for feature extraction.
    """
    def __init__(self, arch: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        print(f"=> creating model '{arch}'")
        model = models.__dict__[arch](weights=models.get_model_weights(arch).DEFAULT if pretrained else None)

        # Determine number of output features
        if 'densenet' in arch:
            self.num_feat = model.classifier.in_features
        elif 'resnet' in arch:
            self.num_feat = model.fc.in_features
        elif 'vgg' in arch or 'convnext' in arch or 'efficientnet' in arch:
            self.num_feat = model.classifier[-1].in_features
        else:
            raise NotImplementedError(f"Unsupported architecture: {arch}")

        # Remove pooling and classifier layers
        modules = []
        for name, m in model.named_children():
            if isinstance(m, (nn.Linear, nn.AdaptiveAvgPool2d)):
                continue
            if name == 'classifier':
                continue
            modules.append(m)
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_num_feat(self) -> int:
        return self.num_feat


# -------------------------
# Simple Attention Pooling
# -------------------------
class SimpleAttentionPool(nn.Module):
    """
    Learn attention over spatial dimensions (e.g., slices or patches).
    Adapted from: https://github.com/reginabarzilaygroup/Sybil
    """
    def __init__(self, num_chan: int, num_dim: int):
        super().__init__()
        self.attention_fc = nn.Linear(num_chan, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(num_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            dict with keys:
                - 'attention_map': [B, 1, H, W]
                - 'hidden': [B, C]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, N]
        attn_scores = self.attention_fc(x_flat.transpose(1, 2))  # [B, N, 1]

        # Compute normalized attention map
        attn_map = self.norm(self.logsoftmax(attn_scores.transpose(1, 2)).view(B, 1, H, W))
        attn_weights = self.softmax(attn_scores.transpose(1, 2))  # [B, 1, N]

        # Weighted sum of features
        weighted_x = x_flat * attn_weights
        hidden = weighted_x.sum(dim=-1)  # [B, C]

        return {'attention_map': attn_map, 'hidden': hidden}