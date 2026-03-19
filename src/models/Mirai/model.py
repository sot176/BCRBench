import torch
import torch.nn as nn
from .onconet.models.factory import get_model_by_name, load_model, RegisterModel
from .onconet.models.hiddens_transfomer import AllImageTransformer
from models.common_parts import extract_mirai_backbone_full, extract_mirai_backbone
from config.config import cfg
from models.common_parts  import  CumulativeProbabilityLayer


@RegisterModel("mirai_full")
class Mirai(nn.Module):

    def __init__(self, args):
        super(Mirai, self).__init__()
        self.args = args

        self.image_encoder = extract_mirai_backbone(cfg["paths"]["mirai_path"])
       
        if hasattr(args, "freeze_image_encoder") and args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        self.transformer = AllImageTransformer(args)

        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, batch):
        x = batch["images"]
        B, C, N, H, W = x.size()

        # 1. Flatten views for the encoder
        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        # 2. Encode
        img_x = self.image_encoder(x)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]

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

