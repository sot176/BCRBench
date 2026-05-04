
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmrnn import VMRNNEncoder
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from models.common_parts import CumulativeProbabilityLayer
from models.Mirai import onconet as _onconet
from models.common_parts import BaseRiskModel
from models.Mirai.onconet.models.factory import get_model_by_name, load_model
from .model_utils import (
    FORMAL_VIEW_SEQUENCE,
    MAX_FOLLOWUP,
    compute_asymmetry_feature,
    expand_risk_factors_per_img,
    register_onconet_alias,
    get_img_repr_dim,
    make_transformer_batch,
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

        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot,
                args,
                do_wrap_model=False,
            )
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name("transformer", False, args)

        if hasattr(self.transformer, "args"):
            self.args.img_only_dim = getattr(
                self.transformer.args,
                "transformer_hidden_dim",
                getattr(self.transformer.args, "transformer_hidden_dim", self.image_repr_dim),
            )
        self.vmrnn_hidden_dim = 128

        self.temporal_projection = nn.Linear(
            self.args.transformer_hidden_dim,
            self.vmrnn_hidden_dim,
        )

        self.vmrnn = VMRNNEncoder(
            input_dim=self.vmrnn_hidden_dim,
            hidden_dim=self.vmrnn_hidden_dim,
            spatial_resolution=(4, 4),
            downsample_depths=(1, 2),
            upsample_depths=(2, 1),
            dropout=0.1,
            vss_backend="transformer",  # AMD-safe
        )

        # -------------------------
        # Asymmetry branch
        # -------------------------
        self.use_asymmetry = getattr(self.args, "use_asymmetry", True)
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

        # -------------------------
        # Risk head
        # -------------------------
        final_dim =  self.vmrnn_hidden_dim + (1 if self.use_asymmetry else 0)
        self.ahl = CumulativeProbabilityLayer(final_dim, max_followup=self.args.max_followup)
    
    
    def forward(self, batch):
        images = batch["images"]        # (B, T, V, C, H, W)
        exam_mask = batch["exam_mask"]  # (B, T)
        view_mask = batch["view_mask"]  # (B, T, V)

        if self.freeze_image_encoder:
            self.image_encoder.eval()

        B, T, V, C, H, W = images.shape

        if V != len(FORMAL_VIEW_SEQUENCE):
            raise ValueError(f"Expected {len(FORMAL_VIEW_SEQUENCE)} formal views, got {V}.")


        flat_exam_mask = view_mask.reshape(B * T, V).bool()
        flat_valid_exam_mask = exam_mask.reshape(B * T).bool()

        partial_exam_mask = flat_exam_mask.any(dim=1) & ~flat_exam_mask.all(dim=1)
        if partial_exam_mask.any():
            raise RuntimeError("Formal Mirai fusion requires complete exams; partial exams are not supported.")

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

            if valid_feat_maps is None:
                feat_maps = None
            else:
                feat_maps = valid_feat_maps.new_zeros(
                    (total_views, *valid_feat_maps.shape[1:])
                )
                feat_maps[flat_view_mask] = valid_feat_maps
        else:
            hidden = flat_images.new_zeros((total_views, self.image_repr_dim))
            feat_maps = None

        hidden = hidden.view(B * T, V, -1)
        hidden = hidden[:, :, :self.image_repr_dim]

        transformer_dim = int(
            getattr(
                self.transformer.args,
                "transformer_hidden_dim",
                getattr(self.transformer.args, "transfomer_hidden_dim", self.image_repr_dim),
            )
        )

        exam_embeddings = hidden.new_zeros(B * T, transformer_dim)
        complete_exam_mask = flat_valid_exam_mask & flat_exam_mask.all(dim=1)

        if complete_exam_mask.any():
            exam_tokens = hidden[complete_exam_mask]

            transformer_batch = make_transformer_batch(
                int(complete_exam_mask.sum().item()),
                hidden.device,
            )

            projected = self.transformer.projection_layer(exam_tokens)

            encoded = self.transformer.transformer(
                projected,
                transformer_batch["time_seq"],
                transformer_batch["view_seq"],
                transformer_batch["side_seq"],
            )

            transformer_args = model_args(self.transformer)
            transformer_risk_factors = zero_risk_factors_for_args(
                transformer_args,
                int(complete_exam_mask.sum().item()),
                encoded.device,
                encoded.dtype,
            )

            encoded_for_pool = encoded.transpose(1, 2).unsqueeze(-1)

            if transformer_risk_factors is None:
                _, pooled_hidden = self.transformer.aggregate_and_classify(encoded_for_pool)
            else:
                _, pooled_hidden = self.transformer.aggregate_and_classify(
                    encoded_for_pool,
                    transformer_risk_factors,
                )

            exam_embeddings[complete_exam_mask] = pooled_hidden

        exam_embeddings = exam_embeddings.view(B, T, -1)
        exam_embeddings = self.temporal_projection(exam_embeddings)

        history_embedding, states, reconstructions = self.vmrnn(exam_embeddings, exam_mask)


        features = [history_embedding]

        asymmetry_scores = None
        coords = None
        coord_valid = None
        r_aa = history_embedding.new_zeros(B, 1)

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

        if self.use_asymmetry:
            features.append(r_aa)

        for i, f in enumerate(features):
            if f.dim() != 2 or f.size(0) != B:
                raise ValueError(f"Feature {i} has invalid shape {f.shape}, expected (B, D)")

        combined = torch.cat(features, dim=-1)
        logits = self.ahl(combined)
        probs = torch.sigmoid(logits)

        return {
            "logit": logits,
            "probs": probs,
            "history_embedding": history_embedding,
            "exam_embeddings": exam_embeddings,
            "states": states,
            "vmrnn_reconstruction": reconstructions,
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

