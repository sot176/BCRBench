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

        if getattr(self.args, "freeze_image_encoder", False):
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
            input_dim=args.embed_dim,                          # 512
            hidden_dim=getattr(args, 'vmrnn_hidden_dim', 256),  # internal dim
            spatial_h=getattr(args, 'vmrnn_spatial_h', 16),
            spatial_w=getattr(args, 'vmrnn_spatial_w', 16),
            depths_down=args.depths_downsample,                # [2,2,6,2]
            depths_up=args.depths_upsample,                    # [2,2,6,2]
        )

        # --------------------------------------------------
        # Asymmetry modules
        # --------------------------------------------------
        self.use_asymmetry = getattr(args, "use_asymmetry", True)
        if self.use_asymmetry:
            self.sad = SpatialAsymmetryDetector(args)
            self.lat = LongitudinalAsymmetryTracker(args)
            latent_h = getattr(args, "latent_h", 64)
            latent_w = getattr(args, "latent_w", 52)
            self.asym_proj = nn.Linear(latent_h * latent_w, 512)
        # --------------------------------------------------
        # Additive Hazard Layer
        # --------------------------------------------------
        input_dim = args.embed_dim

        if self.use_asymmetry:
            input_dim += 512         # LAT always outputs 512 (feature_dim in lat.py)

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

        visit_embeddings_flat = visit_embeddings.mean(dim=(-2,-1))  # (B, T, C)

        # --------------------------------------------------
        # VMRNN temporal modeling
        # --------------------------------------------------
        states_down = None
        states_up = None
        outputs = []
        for t in range(T):
            Tt = visit_embeddings_flat[:, t]                 # (B, C)
            out, states_down, states_up = self.vmrnn(Tt, states_down, states_up)
            outputs.append(out)                              # (B, hidden_dim)

        outputs = torch.stack(outputs, dim=1)            # (B, T, C)

        # --------------------------------------------------
        # Temporal pooling over T and spatial dims
        # --------------------------------------------------
        temporal_feature = outputs.mean(dim=1)           # (B, C)

        # --------------------------------------------------
        # Asymmetry features
        # --------------------------------------------------
        features = [temporal_feature]
        if self.use_asymmetry and V >= 4:
            # feats: (B, T, V, C, H, W)
            # Views: 0=left CC, 1=right CC, 2=left MLO, 3=right MLO
            
            # Average CC+MLO views per side → (B, T, C, H, W)
            left  = feats[:, :, [0, 2]].mean(dim=2)   # (B, T, C, H, W)
            right = feats[:, :, [1, 3]].mean(dim=2)   # (B, T, C, H, W)

            asym = self.sad(left, right)                
            heatmaps = asym['heatmap']                           
            B_a, T_a, H_a, W_a = heatmaps.shape
            asym_features = heatmaps.view(B_a, T_a, H_a * W_a) # (B, T, H*W)
            asym_features = self.asym_proj(asym_features)       # (B, T, 512)

            asym_feature = self.lat(
                asym_features,                  # (B, T, 512)
                asym['asymmetry_coords'],       # (B, T, 2)
                asym['heatmap']                 # (B, T, H, W)
            )
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
