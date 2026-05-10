import torch
import torch.nn as nn
from .vmrnn import VMRNNEncoder
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.factory import get_model_by_name, load_model
from .model_utils import (
    FORMAL_VIEW_SEQUENCE,
    compute_asymmetry_feature,
    expand_risk_factors_per_img,
    register_onconet_alias,
    get_img_repr_dim,
    model_args,
    setup_feature_map_hook,
    zero_risk_factors_for_args,
)

register_onconet_alias(_onconet)

class VMRAMaR(BaseRiskModel):
    def __init__(self, args):
        super().__init__(args)

        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot,
                args,
                do_wrap_model=False,
            )
        else:
            self.image_encoder = get_model_by_name("custom_resnet", False, args)

        self.freeze_image_encoder = bool(getattr(args, "freeze_image_encoder", True))
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = get_img_repr_dim(self.image_encoder)

        self.use_asymmetry =(getattr(self.args, "use_asymmetry", False))
        print(f"[VMRAMaR] use_asymmetry = {self.use_asymmetry}")

        if self.use_asymmetry:
            setup_feature_map_hook(self, args)

            self.sad = SpatialAsymmetryDetector(
                latent_h=int(self.args.latent_h),
                latent_w=int(self.args.latent_w),
            )
            self.lat = LongitudinalAsymmetryTracker(
                threshold_ratio=float(getattr(self.args, "threshold_ratio", 0.4)),
                persistent_weight=float(getattr(self.args, "persistent_weight", 1.0)),
            )

        self.vmrnn_hidden_dim = int(getattr(self.args, "vmrnn_hidden_dim", 128))

        self.view_attention = nn.Sequential(
            nn.Linear(self.image_repr_dim, self.image_repr_dim),
            nn.Tanh(),
            nn.Linear(self.image_repr_dim, 1),
        )

        self.vmrnn = VMRNNEncoder(
            input_dim=self.image_repr_dim,
            hidden_dim=self.vmrnn_hidden_dim,
            spatial_resolution=(
                getattr(self.args, "vmrnn_spatial_resolution", None),
                (4, 4),
            ),
            downsample_depths=(
                getattr(self.args, "vmrnn_downsample_depths", None),
                (1, 2),
            ),
            upsample_depths=(
                getattr(self.args, "vmrnn_upsample_depths", None),
                (2, 1),
            ),
            dropout=float(getattr(self.args, "vmrnn_dropout", 0.1)),
            vss_backend=getattr(self.args, "vss_backend", "transformer"),
            released_weight_path=getattr(self.args, "vmrnn_released_weight_path", None),
        )

        final_dim = self.vmrnn_hidden_dim + (1 if self.use_asymmetry else 0)
        self.ahl = CumulativeProbabilityLayer(
            final_dim,
            max_followup=self.args.max_followup,
        )

    def forward(self, batch):
        images = batch["images"]        # (B, T, V, C, H, W)
        exam_mask = batch["exam_mask"]  # (B, T)
        view_mask = batch["view_mask"]  # (B, T, V)

        if self.freeze_image_encoder:
            self.image_encoder.eval()

        B, T, V, C, H, W = images.shape

        if V != len(FORMAL_VIEW_SEQUENCE):
            raise ValueError(
                f"Expected {len(FORMAL_VIEW_SEQUENCE)} formal views, got {V}."
            )

        flat_exam_mask = view_mask.reshape(B * T, V).bool()
        partial_exam_mask = flat_exam_mask.any(dim=1) & ~flat_exam_mask.all(dim=1)

        if partial_exam_mask.any():
            raise RuntimeError(
                "Formal Mirai fusion requires complete exams; partial exams are not supported."
            )

        total_views = B * T * V
        flat_images = images.contiguous().view(total_views, C, H, W)
        flat_view_mask = view_mask.reshape(total_views).bool()

        image_encoder_args = model_args(self.image_encoder)

        image_risk_factors = zero_risk_factors_for_args(
            image_encoder_args,
            B * T,
            flat_images.device,
            flat_images.dtype,
        )

        image_risk_factors_per_img = expand_risk_factors_per_img(
            image_risk_factors,
            V,
        )

        feat_maps = None

        if flat_view_mask.any():
            valid_images = flat_images[flat_view_mask]

            valid_risk_factors = None
            if image_risk_factors_per_img is not None:
                valid_risk_factors = [
                    rf[flat_view_mask] for rf in image_risk_factors_per_img
                ]

            if self.use_asymmetry:
                self._captured_feature_map = None

            with torch.set_grad_enabled(not self.freeze_image_encoder):
                _, valid_hidden, _ = self.image_encoder(
                    valid_images,
                    valid_risk_factors,
                    batch,
                )

            valid_feat_maps = None
            if self.use_asymmetry:
                valid_feat_maps = self._captured_feature_map
                if valid_feat_maps is None:
                    raise RuntimeError(
                        f"Feature hook did not capture layer {self.feature_map_layer_name!r}."
                    )

            hidden = valid_hidden.new_zeros((total_views, valid_hidden.size(-1)))
            hidden[flat_view_mask] = valid_hidden

            if valid_feat_maps is not None:
                feat_maps = valid_feat_maps.new_zeros(
                    (total_views, *valid_feat_maps.shape[1:])
                )
                feat_maps[flat_view_mask] = valid_feat_maps

        else:
            hidden = flat_images.new_zeros((total_views, self.image_repr_dim))

        hidden = hidden.view(B, T, V, -1)
        hidden = hidden[:, :, :, : self.image_repr_dim]

        view_mask_bool = view_mask.bool()
        attn_logits = self.view_attention(hidden).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~view_mask_bool, -1e9)

        attn_weights = torch.softmax(attn_logits, dim=2).unsqueeze(-1)
        fused_feats = (hidden * attn_weights).sum(dim=2)

        empty_exam = ~view_mask_bool.any(dim=2)
        fused_feats = fused_feats.masked_fill(empty_exam.unsqueeze(-1), 0.0)

        temporal_feature, temporal_sequence, reconstruction_sequence = self.vmrnn(
            fused_feats,
            exam_mask.bool(),
        )

        features = [temporal_feature]

        asymmetry_scores = None
        coords = None
        coord_valid = None
        r_aa = temporal_feature.new_zeros(B, 1)

        if self.use_asymmetry and feat_maps is not None:
            c_feat, hf, wf = feat_maps.shape[1:]
            feat_maps = feat_maps.view(B, T, V, c_feat, hf, wf)

            r_aa, asymmetry_scores, coords, coord_valid = compute_asymmetry_feature(
                self.sad,
                self.lat,
                feat_maps,
                view_mask,
                exam_mask,
            )

            if r_aa.dim() == 1:
                r_aa = r_aa.unsqueeze(-1)

            features.append(r_aa)

        combined = torch.cat(features, dim=-1)

        logits = self.ahl(combined)
        probs = torch.sigmoid(logits)

        return {
            "logit": logits,
            "probs": probs,
            "temporal_feature": temporal_feature,
            "temporal_sequence": temporal_sequence,
            "reconstruction_sequence": reconstruction_sequence,
            "exam_embeddings": fused_feats,
            "view_attention": attn_weights.squeeze(-1),
            "exam_asymmetry": asymmetry_scores,
            "coords": coords,
            "coord_valid": coord_valid,
            "r_aa": r_aa,
        }


    def get_risk_heads(self, outputs, batch):
        return {
            "logit_output": (
                outputs["logit"],
                batch["target"],
                batch["y_mask"],
            )
        }

    def get_primary_risk_head(self, outputs):
        return outputs["probs"]

