import torch.nn as nn
from .model_utils import ResNet18Backbone, SimpleTransformer, load_model


class MiraiFull(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # Image encoder
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(
                args.img_encoder_snapshot,
                args,
                ResNet18Backbone
            )
        else:
            self.image_encoder = ResNet18Backbone(args)

        if getattr(args, "freeze_image_encoder", False):
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.image_repr_dim = 512

        # Transformer
         # --- Transformer ---
        if args.transformer_snapshot is not None:
            self.transformer = load_model(
                args.transformer_snapshot,
                args,
                SimpleTransformer
            )
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = SimpleTransformer(args)

    def forward(self, data, risk_factors=None, batch=None):
        x = data['images']
        B, C, N, H, W = x.shape

        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        _, img_hidden, _ = self.image_encoder(x)

        img_hidden = img_hidden.view(B, N, -1)

        logit, transformer_hidden, activ_dict = self.transformer(img_hidden)

        return {'logit': logit, 'transformer_hidden': transformer_hidden, 'activ_dict': activ_dict}

    
    def get_risk_heads(self, outputs, batch):
        target = batch["target"]
        mask = batch["y_mask"]

        return {
            "logit_output": (outputs["logit"], target, mask) }
    
    def get_primary_risk_head(self, outputs):
        return outputs["logit"]

