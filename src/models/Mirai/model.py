import torch
import torch.nn as nn
from .onconet.models.factory import get_model_by_name, load_model, RegisterModel
from .onconet.models.hiddens_transfomer import AllImageTransformer
from models.common_parts import extract_mirai_backbone
from config.config import cfg


@RegisterModel("mirai_full")
class Mirai(nn.Module):

    def __init__(self, args):
        super(Mirai, self).__init__()
        self.args = args

        self.image_encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )

        if hasattr(args, "freeze_image_encoder") and args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot, AllImageTransformer, args, do_wrap_model=False
            )
        else:
            self.transformer = get_model_by_name('transformer', False, args)

    def forward(self, data, risk_factors=None, batch=None):
        x = data['images']
        batch = data

        B, C, N, H, W = x.size()

        # 1. Encode every view with the shared backbone
        x = x.transpose(1, 2).contiguous()   # (B, N, C, H, W)
        x = x.view(B * N, C, H, W)
        img_x = self.image_encoder(x)         # (B*N, precomputed_hidden_dim, h, w)

        # 2. Reshape into (B, N, precomputed_hidden_dim) — no spatial pooling here.
        #    AllImageTransformer.projection_layer expects flat per-image vectors,
        #    but the pool in aggregate_and_classify expects (B, D, N, 1).
        #    We match what the original does: pool spatial dims, keep N for the transformer.
        img_x = nn.functional.adaptive_avg_pool2d(img_x, 1)  # (B*N, D, 1, 1)
        img_x = img_x.flatten(1)                              # (B*N, D)
        img_x = img_x.view(B, N, -1)                         # (B, N, D)

        # 3. Transformer aggregates across views/timepoints and classifies
        #    Returns: logit, transformer_hidden (B, N, hidden_dim), activ_dict
        logit, transformer_hidden, activ_dict = self.transformer(
            img_x, risk_factors, batch
        )

        return {
            'logit': logit,
            'transformer_hidden': transformer_hidden,
            'activ_dict': activ_dict,
        }

    def get_risk_heads(self, outputs, batch):
        return {
            "logit_output": (outputs["logit"], batch["target"], batch["y_mask"])
        }

    def get_primary_risk_head(self, outputs):
        return outputs["logit"]