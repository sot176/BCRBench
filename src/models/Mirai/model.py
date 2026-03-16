import torch
import torch.nn as nn
from .onconet.models.factory import get_model_by_name, load_model, RegisterModel
from .onconet.models.hiddens_transfomer import AllImageTransformer
from models.common_parts import extract_mirai_backbone
from config.config import cfg
from models.common_parts  import  CumulativeProbabilityLayer


@RegisterModel("mirai_full")
class Mirai(nn.Module):

    def __init__(self, args):
        super(Mirai, self).__init__()
        self.args = args

        if args.img_encoder_snapshot is not None:
            self.image_encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(args, "freeze_image_encoder") and args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot, args, do_wrap_model=False
            )
            # Fix 1 — remove risk factor dependency from pool
            self.transformer.args.use_risk_factors = False
            self.transformer.pool = self.transformer.pool.internal_pool

            # Fix 2 — replace fc since original was 612 (512 + 100 risk factor dim)
            self.transformer.fc = nn.Linear(
                512,   
                self.transformer.fc.out_features    # keep same num classes
            )

            # Fix 3 — ensure survival layer exists with correct args
            self.transformer.args.survival_analysis_setup = args.survival_analysis_setup
            if args.survival_analysis_setup and not hasattr(self.transformer, 'prob_of_failure_layer'):
                self.transformer.prob_of_failure_layer = CumulativeProbabilityLayer(
                    512,
                    max_followup=args.max_followup
                )
            print("New fc shape:", self.transformer.fc.weight.shape)  # should be (2, 512) not (2, 612)

        else:
            self.transformer = get_model_by_name('transformer', False, args)


    def forward(self, data, batch=None):
        x = data["images"]
        B, C, N, H, W = x.size()
        batch=data

        # 1. Flatten views for the encoder
        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        # 2. Encode
        img_x = self.image_encoder(x)                                # (B*N, 512, h, w)

        # 3. Pool spatial dims and reshape
        img_x = nn.functional.adaptive_avg_pool2d(img_x, 1)         # (B*N, 512, 1, 1)
        img_x = img_x.flatten(1)                                     # (B*N, 512)
        img_x = img_x.view(B, N, -1)                                 # (B, N, 512)

        # 4. Transformer aggregates across views/timepoints
        logit, transformer_hidden, activ_dict = self.transformer(
            img_x, None, batch
        )

        return logit, transformer_hidden, activ_dict

    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs
        return {
            "logit_output": (logit, batch["target"], batch["y_mask"])
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return logit

