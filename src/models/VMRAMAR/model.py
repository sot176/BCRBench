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
        x = data["images"]  # (B, T, C, V, H, W)
        B, T, C, V, H, W = x.shape

        # --------------------------------------------------
        # Image encoder
        # --------------------------------------------------
        x = x.permute(0,1,3,2,4,5).contiguous()  # (B, T, V, C, H, W)
        x = x.view(B*T*V, C, H, W)
        feats = self.image_encoder(x)  # (B*T*V, C_feat, Hf, Wf)
        BTV, C_feat, Hf, Wf = feats.shape

        # --------------------------------------------------
        # Reshape back to (B, T, C, V, H, W) for ImageAggregator
        # --------------------------------------------------
        feats = feats.view(B, T, V, C_feat, Hf, Wf)

        # --------------------------------------------------
        # Image Aggregator: fuse views
        # --------------------------------------------------
        visit_embeddings = self.image_aggregator(feats)  # (B, T, C, H, W)

        # --------------------------------------------------
        # VMRNN temporal modeling
        # --------------------------------------------------
        states_down = None
        states_up = None
        outputs = []
        for t in range(T):
            xt = visit_embeddings[:, t]  # (B, C, H, W)
            B, C, Hc, Wc = xt.shape
            xt_flat = xt.view(B, Hc*Wc, C)  # (B, L, C)
            
            out, states_down, states_up = self.vmrnn(
                xt_flat, states_down, states_up
            )  # out: (B, L_out, C_out)
            
            # Infer output H/W
            L_out, C_out = out.shape[1], out.shape[2]
            H_out = W_out = int(L_out ** 0.5)
            out = out.view(B, H_out, W_out, C_out).permute(0, 3, 1, 2)  # (B, C_out, H_out, W_out)
            
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (B, T, C_out, H_out, W_out)


        # --------------------------------------------------
        # Temporal pooling over T and spatial dims
        # --------------------------------------------------
        temporal_feature = outputs.mean(dim=(1, 3, 4))  # (B, C)

        # --------------------------------------------------
        # Asymmetry features
        # --------------------------------------------------
        features = [temporal_feature]
        if self.use_asymmetry and V >= 4:
            left = feats[:, :, [0, 2]]  # (B, T, 2, C, H, W)
            right = feats[:, :, [1, 3]]  # (B, T, 2, C, H, W)
            asym = self.sad(left, right)
            asym_feature = self.lat(asym)
            features.append(asym_feature)

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
