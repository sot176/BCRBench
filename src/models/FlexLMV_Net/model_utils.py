
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg

from models.common_parts import extract_mirai_backbone, ContinuousPosEncoding, SpatialTransformerBlock


class SpatialAttentionModule(nn.Module):
    """Channel and spatial attention for longitudinal feature maps."""

    def __init__(self, channels: int):
        super().__init__()

        hidden_channels = max(channels // 8, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(
                channels, 1, kernel_size=3, padding=1, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        channel_weights = self.channel_attn(feat)
        refined = feat * channel_weights

        spatial_map = self.spatial_attn(refined)
        refined = refined * spatial_map

        return spatial_map, refined


class FeatureAwareTemporalAggregator(nn.Module):
    """
    Aggregate variable-length prior and difference feature sequences.

    Temporal importance is conditioned on current features, prior features,
    and the time gap. Spatial attention refines each prior feature map.
    """

    def __init__(
        self,
        channels: int,
        max_priors: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.channels = channels
        self.max_priors = max_priors

        self.score_net = nn.Sequential(
            nn.Linear(channels * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.out_dropout = nn.Dropout2d(dropout)
        self.spatial_attn = SpatialAttentionModule(channels)

        # Diagnostic information; populated during forward().
        self.spatial_attention_maps = None
        self.temporal_weights = None

    @staticmethod
    def _gap_tensor(gap, batch_size, device):
        if isinstance(gap, torch.Tensor):
            gap = gap.to(device=device, dtype=torch.float32).reshape(-1)

            if gap.numel() == 1:
                return gap.expand(batch_size)

            if gap.numel() != batch_size:
                raise ValueError(
                    f"Expected one gap or {batch_size} gaps, "
                    f"got {gap.numel()}."
                )

            return gap

        return torch.full(
            (batch_size,), float(gap),
            device=device, dtype=torch.float32
        )

    def _aggregate_one(self, cur_feat, feats, gaps):
        if not feats:
            return (
                torch.zeros_like(cur_feat),
                cur_feat.new_zeros(cur_feat.size(0), 0),
                None,
            )

        feats = feats[:self.max_priors]
        gaps = gaps[:len(feats)]

        batch_size = cur_feat.size(0)
        pooled_cur = F.adaptive_avg_pool2d(
            cur_feat, 1
        ).flatten(1)

        scores = []
        spatial_maps = []
        refined_feats = []

        for feat, gap in zip(feats, gaps):
            pooled_prior = F.adaptive_avg_pool2d(
                feat, 1
            ).flatten(1)

            gap_feat = self._gap_tensor(
                gap, batch_size, cur_feat.device
            ).unsqueeze(1)

            score_input = torch.cat(
                [pooled_cur, pooled_prior, gap_feat], dim=1
            )
            scores.append(self.score_net(score_input))

            spatial_map, refined_feat = self.spatial_attn(feat)
            spatial_maps.append(spatial_map)
            refined_feats.append(refined_feat)

        temporal_weights = torch.softmax(
            torch.cat(scores, dim=1), dim=1
        )

        # [B, N, C, H, W] and [B, N, 1, H, W]
        stacked_feats = torch.stack(refined_feats, dim=1)
        stacked_spatial = torch.stack(spatial_maps, dim=1)

        # Combine temporal importance with spatial attention.
        weights = (
            temporal_weights[:, :, None, None, None]
            * stacked_spatial
        ).squeeze(2)  # [B, N, H, W]

        weights = weights / (
            weights.sum(dim=1, keepdim=True) + 1e-8
        )

        fused = (
            weights.unsqueeze(2) * stacked_feats
        ).sum(dim=1)

        fused = self.out_dropout(fused)

        return fused, temporal_weights, stacked_spatial

    def forward(
        self,
        cur_feat,
        prior_feats,
        diff_feats,
        absolute_gaps,
        sequential_gaps=None,
    ):
        prior, prior_weights, prior_maps = self._aggregate_one(
            cur_feat, prior_feats, absolute_gaps
        )

        diff, diff_weights, diff_maps = self._aggregate_one(
            cur_feat, diff_feats, absolute_gaps
        )

        self.spatial_attention_maps = {
            "prior": prior_maps,
            "diff": diff_maps,
        }

        self.temporal_weights = {
            "prior": prior_weights,
            "diff": diff_weights,
        }

        return prior, diff, prior_weights, diff_weights


class LongitudinalFeatureProcessorMultiplePriors(nn.Module):
    """
    Extract current and prior Mirai feature maps, align priors to the
    current examination, encode longitudinal differences, and aggregate
    prior and difference representations separately.
    """

    def __init__(
        self,
        mammo_reg_net,
        finetune: bool = False,
        aggregation_mode: str = "gated",
        num_prior_img: int = 2,
        aggregator_hidden_dim: int = 64,
        encoder_path: str = cfg["paths"]["mirai_path"], 
        feature_channels: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        if num_prior_img < 1:
            raise ValueError("num_prior_img must be at least 1.")

        if aggregation_mode != "gated":
            raise ValueError(
                f"Unsupported aggregation_mode: {aggregation_mode}"
            )

        self.num_prior_img = num_prior_img
        self.aggregation_mode = aggregation_mode
        self.feature_channels = feature_channels

        self.encoder = extract_mirai_backbone(encoder_path)
        self.mammo_reg_net = mammo_reg_net
        self.feat_transformer = SpatialTransformerBlock(
            mode="bilinear"
        )

        self.positional_encoding = ContinuousPosEncoding(dim=feature_channels)

        # IMPORTANT: only enable encoder gradients when fine-tuning.
        self.encoder.requires_grad_(finetune)

        # Registration network remains frozen.
        self.mammo_reg_net.requires_grad_(False)
        self.mammo_reg_net.eval()

        self.aggregator = FeatureAwareTemporalAggregator(
            channels=feature_channels,
            max_priors=num_prior_img,
            hidden_dim=aggregator_hidden_dim,
            dropout=dropout,
        )

    def train(self, mode=True):
        """Keep the frozen registration network in evaluation mode."""
        super().train(mode)
        self.mammo_reg_net.eval()
        return self

    @staticmethod
    def _normalize_image(img, device=None):
        if img.ndim == 2:
            img = img[None, None]
        elif img.ndim == 3:
            img = img[None]

        if img.ndim != 4:
            raise ValueError(
                f"Expected image with 2-4 dimensions, got {img.shape}"
            )

        return img.to(device) if device is not None else img

    @staticmethod
    def _get_patient_priors(priors, b):
        if priors is None:
            return []

        if isinstance(priors, torch.Tensor):
            if priors.ndim == 3:
                return [priors[b]]
            if priors.ndim in (4, 5):
                return [priors[b, i] for i in range(priors.size(1))]

            raise ValueError(
                f"Unexpected prior tensor shape: {priors.shape}"
            )

        if b >= len(priors) or priors[b] is None:
            return []

        return list(priors[b])

    @staticmethod
    def _to_gap_tensor(gap, device):
        if isinstance(gap, torch.Tensor):
            return gap.to(device=device, dtype=torch.float32).reshape(())
        return torch.tensor(float(gap), device=device)

    def _get_patient_gaps(self, gaps, b, n_priors, device):
        if n_priors == 0:
            return []

        default_gap = lambda: torch.tensor(
            1.0, device=device, dtype=torch.float32
        )

        if gaps is None:
            return [default_gap() for _ in range(n_priors)]

        if isinstance(gaps, torch.Tensor):
            if gaps.ndim == 0:
                values = [gaps] * n_priors
            elif gaps.ndim == 1:
                # One value per patient, or one value per prior.
                if gaps.numel() == 1:
                    values = [gaps[0]] * n_priors
                elif gaps.numel() == n_priors:
                    values = list(gaps)
                else:
                    values = [gaps[b]] * n_priors
            elif gaps.ndim == 2:
                values = list(gaps[b, :n_priors])
            else:
                raise ValueError(
                    f"Unexpected time gap shape: {gaps.shape}"
                )

        else:
            if b >= len(gaps) or gaps[b] is None:
                return [default_gap() for _ in range(n_priors)]

            patient_gaps = gaps[b]

            if isinstance(patient_gaps, torch.Tensor):
                if patient_gaps.ndim == 0:
                    values = [patient_gaps] * n_priors
                else:
                    values = list(patient_gaps[:n_priors])
            else:
                values = list(patient_gaps[:n_priors])

        values = [
            self._to_gap_tensor(gap, device) for gap in values
        ]

        if not values:
            return [default_gap() for _ in range(n_priors)]

        # Keep the existing fallback for missing gap values.
        values.extend(
            [values[-1]] * (n_priors - len(values))
        )

        return values[:n_priors]

    @staticmethod
    def _sequential_gaps(absolute_gaps):
        """Convert cumulative time gaps to adjacent examination intervals."""
        sequential = []
        previous = None

        for gap in absolute_gaps:
            sequential.append(
                gap if previous is None else gap - previous
            )
            previous = gap

        return sequential

    def _encode_image(self, img):
        img = self._normalize_image(img)
        return self.encoder(img.expand(-1, 3, -1, -1))

    def _align_prior(self, img_cur, f_cur, img_prior):
        img_prior = self._normalize_image(
            img_prior, img_cur.device
        )
        f_prior = self._encode_image(img_prior)

        # Registration estimates the deformation in image coordinates.
        with torch.no_grad():
            deformation = self.mammo_reg_net(
                img_cur, img_prior
            )[1]

        deformation = F.interpolate(
            deformation,
            size=f_cur.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        deformation[:, 0] *= (
            f_cur.shape[-1] / img_cur.shape[-1]
        )
        deformation[:, 1] *= (
            f_cur.shape[-2] / img_cur.shape[-2]
        )

        return self.feat_transformer(f_prior, deformation)

    def _encode_difference(
        self, f_previous, f_current, sequential_gap
    ):
        diff = torch.abs(f_previous - f_current)

        batch_size, channels, height, width = diff.shape
        sequence = diff.flatten(2).permute(2, 0, 1)

        # Positional encoding operates over spatial positions here,
        # using the interval as its temporal input.
        gap = sequential_gap.reshape(1).to(
            device=diff.device, dtype=torch.float32
        )

        sequence = self.positional_encoding(sequence, gap)

        return sequence.permute(1, 2, 0).reshape(
            batch_size, channels, height, width
        )

    def _process_view(self, img_current, img_priors, time_gaps):
        batch_size = img_current.size(0)
        device = img_current.device

        f_current_all = self._encode_image(img_current)

        # Start with zeros so patients without usable priors are supported.
        f_prior_all = torch.zeros_like(f_current_all)
        f_diff_all = torch.zeros_like(f_current_all)

        for b in range(batch_size):
            img_current_b = img_current[b:b + 1]
            f_current_b = f_current_all[b:b + 1]

            priors = self._get_patient_priors(
                img_priors, b
            )[:self.num_prior_img]

            if not priors:
                continue

            gaps = self._get_patient_gaps(
                time_gaps, b, len(priors), device
            )
            sequential_gaps = self._sequential_gaps(gaps)

            aligned_priors = [
                self._align_prior(img_current_b, f_current_b, prior)
                for prior in priors
            ]

            # Preserve the current -> prior sequence used by the original
            # model. The order of priors must match the order of their gaps.
            sequence = [f_current_b, *aligned_priors]

            diff_features = [
                self._encode_difference(
                    sequence[i],
                    sequence[i + 1],
                    sequential_gaps[i],
                )
                for i in range(len(aligned_priors))
            ]

            prior, diff, _, _ = self.aggregator(
                f_current_b,
                aligned_priors,
                diff_features,
                gaps,
                sequential_gaps,
            )

            f_prior_all[b:b + 1] = prior
            f_diff_all[b:b + 1] = diff

        return torch.cat(
            [f_current_all, f_prior_all, f_diff_all],
            dim=1,
        )

    def forward(
        self,
        img_cur_cc,
        img_pris_cc,
        img_cur_mlo,
        img_pris_mlo,
        time_gap,
    ):
        return {
            "f_cc_long": self._process_view(
                img_cur_cc, img_pris_cc, time_gap
            ),
            "f_mlo_long": self._process_view(
                img_cur_mlo, img_pris_mlo, time_gap
            ),
        }


    