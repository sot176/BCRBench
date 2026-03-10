import torch.nn as nn
import sys
from .onconet.models.factory import get_model_by_name, load_model, RegisterModel
from .onconet.models.hiddens_transfomer import AllImageTransformer
from models.common_parts import extract_mirai_backbone
from config.config import cfg

@RegisterModel("mirai_full")
class Mirai(nn.Module):

    def __init__(self, args):
        super(Mirai, self).__init__()
        self.args = args
        sys.path.append(cfg["paths"]["asymMirai_master_onconet"])
        self.encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, AllImageTransformer, args, do_wrap_model=False)
        else:
            self.transformer = get_model_by_name('transformer', False, args)

    def forward(self, data, risk_factors=None, batch=None):
        x = data['images']
        batch=data
        B, C, N, H, W = x.size()
        x = x.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)

        return {'logit': logit, 'transformer_hidden': transformer_hidden, 'activ_dict': activ_dict}

    def get_risk_heads(self, outputs, batch):
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "logit_output": (outputs["logit"], target, mask) }
    
    def get_primary_risk_head(self, outputs):
        return outputs["logit"]

