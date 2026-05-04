from logging import root
from pyexpat import features
import sys
from unicodedata import name
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


sys.modules.setdefault("onconet", _onconet)
for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

MAX_FOLLOWUP = 5
FORMAL_VIEW_SEQUENCE = (
    ("RCC", 0, 0),
    ("RMLO", 1, 0),
    ("LCC", 0, 1),
    ("LMLO", 1, 1),
)


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

        self.image_repr_dim = self._get_img_repr_dim()

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
            self._setup_feature_map_hook(args)

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
        self.ahl = CumulativeProbabilityLayer(final_dim, max_followup=5)
    
    def _resolve_module(self, root: nn.Module, name: str) -> nn.Module:
        modules = dict(root.named_modules())
        if name not in modules:
            matches = [key for key in modules if name in key]
            raise ValueError(
                f"Feature map layer {name!r} not found. "
                f"Close matches: {matches[:20]}"
            )
        return modules[name]


    def _setup_feature_map_hook(self, args) -> None:
        self.feature_map_layer_name = getattr(args, "feature_map_layer", "layer4_1")
        self._captured_feature_map = None

        encoder_model = (
            self.image_encoder._model
            if hasattr(self.image_encoder, "_model")
            else self.image_encoder
        )

        self.feature_map_layer = self._resolve_module(
            encoder_model,
            self.feature_map_layer_name,
        )

        def capture_feature_map(module, inputs, output):
            self._captured_feature_map = output

        self.feature_map_hook = self.feature_map_layer.register_forward_hook(
            capture_feature_map
        )


    @staticmethod
    def _freeze_encoder(encoder):
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("[INFO] Image encoder frozen.")
    
    
    def _zero_risk_factors_for_args(
        self,
        args,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if not bool(getattr(args, "use_risk_factors", False)):
            return None

        key_to_dim = getattr(args, "risk_factor_key_to_num_class", None)
        risk_factor_keys = list(getattr(args, "risk_factor_keys", []) or [])

        if (not key_to_dim) and risk_factor_keys:
            from models.Mirai.onconet.utils.risk_factors import RiskFactorVectorizer
            RiskFactorVectorizer(args)
            key_to_dim = args.risk_factor_key_to_num_class

        if key_to_dim and risk_factor_keys:
            return [
                torch.zeros(batch_size, int(key_to_dim[key]), device=device, dtype=dtype)
                for key in risk_factor_keys
            ]

        rf_dim = int(getattr(args, "rf_dim", 0) or 0)
        if rf_dim > 0:
            return [torch.zeros(batch_size, rf_dim, device=device, dtype=dtype)]

        return None

    def _zero_risk_factors(self, batch_size, device, dtype):
        return self._zero_risk_factors_for_args(self.args, batch_size, device, dtype)

    def _get_img_repr_dim(self):
        if hasattr(self.image_encoder, "_model"):
            return self.image_encoder._model.args.img_only_dim
        return self.image_encoder.args.img_only_dim

    def _expand_risk_factors_per_img(self, risk_factors, num_imgs):
        if risk_factors is None:
            return None

        expanded = []
        for factor in risk_factors:
            factor = factor.unsqueeze(1).expand(-1, num_imgs, -1)
            factor = factor.contiguous().view(-1, factor.size(-1))
            expanded.append(factor)
        return expanded
    
    def _model_args(self, model):
        if hasattr(model, "_model"):
            return model._model.args
        return model.args
    
    def _transformer_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        view_seq = torch.tensor([view for _, view, _ in FORMAL_VIEW_SEQUENCE], device=device, dtype=torch.long)
        side_seq = torch.tensor([side for _, _, side in FORMAL_VIEW_SEQUENCE], device=device, dtype=torch.long)

        return {
            "time_seq": torch.zeros(batch_size, len(FORMAL_VIEW_SEQUENCE), device=device, dtype=torch.long),
            "view_seq": view_seq.unsqueeze(0).expand(batch_size, -1),
            "side_seq": side_seq.unsqueeze(0).expand(batch_size, -1),
        }

     
    def _compute_asymmetry_feature(self, feat_maps, view_mask, exam_mask):
        """
        returns:
            r_aa: (B, 1)
            asymmetry_scores, coords, coord_valid
        """
        asymmetry_scores, coords, coord_valid = self.sad(feat_maps, view_mask)

        window_size = max(
            int(self.sad.latent_h),
            int(self.sad.latent_w),
        )

        r_aa = self.lat(
            asymmetry_scores,
            coords,
            coord_valid,
            exam_mask,
            window_size=window_size,
        )

        if r_aa.dim() == 1:
            r_aa = r_aa.unsqueeze(-1)
        elif r_aa.dim() != 2:
            raise ValueError(f"Unexpected asymmetry feature shape: {r_aa.shape}")

        return r_aa, asymmetry_scores, coords, coord_valid
    
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

        image_encoder_args = self._model_args(self.image_encoder)

        image_risk_factors = self._zero_risk_factors_for_args(
            image_encoder_args,
            B * T,
            flat_images.device,
            flat_images.dtype,
        )
        image_risk_factors_per_img = self._expand_risk_factors_per_img(
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

            transformer_batch = self._transformer_batch(
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

            transformer_args = self._model_args(self.transformer)
            transformer_risk_factors = self._zero_risk_factors_for_args(
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

            r_aa, asymmetry_scores, coords, coord_valid = self._compute_asymmetry_feature(
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

