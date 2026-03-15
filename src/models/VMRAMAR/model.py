import torch
import torch.nn as nn
import sys

from models.common_parts  import  CumulativeProbabilityLayer
from models.common_parts import extract_mirai_backbone
from config.config import cfg
from .sad import SpatialAsymmetryDetector
from .lat import LongitudinalAsymmetryTracker
from .vmrnn import VMRNN
from .image_aggregator import ImageAggregator

class VMRAMaR(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        # --------------------------------------------------
        # Image encoder (Mirai backbone)
        # --------------------------------------------------
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.image_encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # --------------------------------------------------
        # Image Aggregator
        # --------------------------------------------------
        self.image_aggregator = ImageAggregator(args.embed_dim)

        # --------------------------------------------------
        # VMRNN
        # --------------------------------------------------
        self.vmrnn = VMRNN(
            embed_dim=args.embed_dim,
            depths_down=args.depths_downsample,
            depths_up=args.depths_upsample,
            feature_resolution=args.feature_resolution
        )

        # --------------------------------------------------
        # Asymmetry modules
        # --------------------------------------------------
        self.use_asymmetry = getattr(args, "use_asymmetry", True)
        if self.use_asymmetry:
            self.sad = SpatialAsymmetryDetector(args)
            self.lat = LongitudinalAsymmetryTracker(args)

        # --------------------------------------------------
        # Additive Hazard Layer
        # --------------------------------------------------
        input_dim = args.embed_dim

        if self.use_asymmetry:
            input_dim += args.asym_dim

        self.ahl =  CumulativeProbabilityLayer(input_dim, max_followup=5)

    def forward(self, data, risk_factors=None):

        x = data["images"] # (B,T,C, V,H,W)
        B, T, C, V, H, W = x.shape

        # --------------------------------------------------
        # Image encoder
        # --------------------------------------------------

        x = x.view(B * T * V, C, H, W)
        feats = self.image_encoder(x) # (B*T*V, C_feat, Hf, Wf)

        C_feat, Hf, Wf = feats.shape[1:]
        feats = feats.view(B, T, V, C_feat, Hf, Wf)

        # --------------------------------------------------
        # Convert to tokens
        # --------------------------------------------------

        L = Hf * Wf
        feats = feats.view(B, T, V, C_feat, L)
        feats = feats.permute(0, 1, 2, 4, 3).contiguous()  # (B,T,V,L,C)

        # --------------------------------------------------
        # Image Aggregator
        # --------------------------------------------------
        visit_embeddings = self.image_aggregator(feats)  # (B,T,L,C)
        # --------------------------------------------------
        # VMRNN temporal modeling
        # --------------------------------------------------

        states_down = None
        states_up = None
        outputs = []

        for t in range(T):
            xt = visit_embeddings[:, t]
            out, states_down, states_up = self.vmrnn(
                xt,
                states_down,
                states_up
            )
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)    # (B,T,L,C)

        # --------------------------------------------------
        # Temporal pooling
        # --------------------------------------------------
        temporal_feature = outputs.mean(dim=(1, 2))

        # --------------------------------------------------
        # Asymmetry features
        # --------------------------------------------------

        features = [temporal_feature]
        if self.use_asymmetry and V >= 4:
            left = feats[:, :, [0, 2]]
            right = feats[:, :, [1, 3]]
            asym = self.sad(left, right)
            asym_feature = self.lat(asym)
            features.append(asym_feature)
            print("Asym feat shape", asym_feature.shape)

        holistic_embedding = torch.cat(features, dim=1)

        # --------------------------------------------------
        # Risk prediction
        # --------------------------------------------------
        risk = self.ahl(holistic_embedding)

        return {"logit": risk}


    def get_risk_heads(self, outputs, batch):

        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "logit_output": (outputs["logit"], target, mask)
        }


    def get_primary_risk_head(self, outputs):
        return outputs["logit"]
