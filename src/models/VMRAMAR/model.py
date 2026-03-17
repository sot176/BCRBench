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
        self.image_aggregator = ImageAggregator(
            args.embed_dim,
            num_views=getattr(args, 'num_images', 4)
        )
        # --------------------------------------------------
        # VMRNN
        # --------------------------------------------------
        self.vmrnn = VMRNN(
            embed_dim=args.embed_dim,                    # their arg name
            depths_downsample=args.depths_downsample,    # their arg name  
            depths_upsample=args.depths_upsample,        # their arg name
            feature_resolution=(1, 1),                   # temporal mode — no spatial U-Net
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
            input_dim += 512         

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
        # ── Image aggregator: pool spatial first, then fuse views ─────────────
        # Pool spatial dims before aggregator — it operates on embeddings not maps
        feats_pooled = feats.mean(dim=(-2, -1))              # (B, T, V, C)
        visit_embeddings = self.image_aggregator(feats_pooled)  # (B, T, C)

        # ── VMRNN: recurrent loop over timesteps ──────────────────────────────
        # Per diagram: feed one T_t at a time, carry states across t
        states_down = None
        states_up   = None
        outputs     = []
        for t in range(T):
            Tt = visit_embeddings[:, t:t+1, :]              # (B, 1, C) — single timestep
            out, states_down, states_up = self.vmrnn(
                Tt, states_down=states_down, states_up=states_up
            )
            outputs.append(out)                              # (B, 1, C)

        outputs = torch.cat(outputs, dim=1)                  # (B, T, C)
        temporal_feature = outputs.mean(dim=1)               # (B, C)


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
            if heatmaps.dim() == 3:
                _, H_a, W_a = heatmaps.shape
                heatmaps = heatmaps.view(B, T, H_a, W_a)  # (B, T, 5, 5)
                     
            B_a, T_a, H_a, W_a = heatmaps.shape
            asym_features = heatmaps.view(B_a, T_a, H_a * W_a) # (B, T, H*W)

            coords = asym['asymmetry_coords']              # may be (B*T, 2)
            if coords.dim() == 2:
                coords = coords.view(B, T, 2)             # (B, T, 2)

            asym_feature = self.lat(
                asym_features,   # (B, T, 512)
                coords,          # (B, T, 2)
                heatmaps         # (B, T, 5, 5)
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
