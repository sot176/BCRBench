import torch.nn as nn
from model_utils import ResNet18Backbone, SimpleTransformer, load_model


class MiraiFull(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # Image encoder
        if args.img_encoder_snapshot is not None:
            # Load pretrained snapshot if available
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = ResNet18Backbone(args)

        if getattr(args, "freeze_image_encoder", False):
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.image_repr_dim = self.image_encoder.hidden_dim

        # Transformer
         # --- Transformer ---
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = SimpleTransformer(args)

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.shape

        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        _, img_hidden, _ = self.image_encoder(x)

        img_hidden = img_hidden.view(B, N, -1)

        logit, transformer_hidden, activ_dict = self.transformer(img_hidden)

        return logit, transformer_hidden, activ_dict

