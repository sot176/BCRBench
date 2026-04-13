import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict
import numpy as np


def prob_to_score(prob, max_followup= 5):
    """Convert probability outputs to cumulative scores over follow-up periods."""
    score = np.zeros_like(prob)[:, :max_followup]
    for i in range(max_followup):
        score[:, i] = prob[:, :i+1].sum(axis=1)
    return score


# -------------------------
# Convolutional Blocks
# -------------------------
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Feedforward(nn.Module):
    """Two stacked ConvBlocks."""
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, num_feat = 2048, dropout = 0.1):
        super().__init__()
        self.embed = nn.Linear(num_feat, num_feat)
        self.log_var = nn.Linear(num_feat, num_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, max_t = 50, use_sto = True):
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
    def __init__(self, arch = 'resnet18', pretrained = True):
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

    def forward(self, x):
        return self.model(x)

    def get_num_feat(self):
        return self.num_feat


# -------------------------
# Simple Attention Pooling
# -------------------------
class SimpleAttentionPool(nn.Module):
    """
    Learn attention over spatial dimensions (e.g., slices or patches).
    Adapted from: https://github.com/reginabarzilaygroup/Sybil
    """
    def __init__(self, **kwargs):
        super(SimpleAttentionPool, self).__init__()
        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(kwargs['num_dim'])

    def forward(self, x):
        '''
        args:
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        '''
        output = {}
        B, C, W, H = x.shape
        # spatially_flat_size = (*x.size()[:2], -1)  # B, C, N

        spatially_flat_size = (B, C, -1)
        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1, 2))  # B, N, 1

        attention_map = self.norm(self.logsoftmax(attention_scores.transpose(1, 2)).view(B, -1)).view(B, 1, W, H)
        output['attention_map'] = attention_map
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # B, 1, N

        x = x * attention_scores  # B, C, N
        output['hidden'] = torch.sum(x, dim=-1)
        return output