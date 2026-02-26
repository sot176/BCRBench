import torch.nn as nn
from model_utils import ResNet18Backbone, SimpleTransformer, load_model


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

        self.image_repr_dim = self.image_encoder.hidden_dim

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

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.shape

        x = x.transpose(1, 2).contiguous().view(B * N, C, H, W)

        _, img_hidden, _ = self.image_encoder(x)

        img_hidden = img_hidden.view(B, N, -1)

        logit, transformer_hidden, activ_dict = self.transformer(img_hidden)

        return logit, transformer_hidden, activ_dict

def main():
    import torch
    from types import SimpleNamespace

    # -------------------------------------------------
    # 1️⃣ Create minimal args
    # -------------------------------------------------
    args = SimpleNamespace(
        # model structure
        num_classes=5,
        dropout=0.2,
        hidden_dim=512,
        transfomer_hidden_dim=512,
        precomputed_hidden_dim=512,
        num_heads=8,
        num_layers=4,

        # image setup
        num_images=4,

        # optional heads
        survival_analysis_setup=False,
        predict_birads=False,
        use_region_annotation=False,
        pred_risk_factors=False,
        use_risk_factors=False,

        # snapshots
        img_encoder_snapshot="C:/UiT_PhD_datasets/mirai_models/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p",
        transformer_snapshot="C:/UiT_PhD_datasets/mirai_models/snapshots/mgh_mammo_MIRAI_Transformer_May20_2019.p",
    )

    # -------------------------------------------------
    # 2️⃣ Build Model
    # -------------------------------------------------
    model = MiraiFull(args)
    model.eval()

    print("\nModel successfully built!\n")
    print(model)

    # -------------------------------------------------
    # 3️⃣ Create Dummy Mammography Batch
    # -------------------------------------------------
    B = 2          # batch size
    C = 3          # channels
    N = 4          # number of views
    H = 224
    W = 224

    x = torch.randn(B, C, N, H, W)

  
    # -------------------------------------------------
    # 4️⃣ Forward Pass
    # -------------------------------------------------
    with torch.no_grad():
        logit, hidden, activ_dict = model(x, risk_factors=None, batch=None)

    # -------------------------------------------------
    # 5️⃣ Print Results
    # -------------------------------------------------
    print("\nForward pass successful!\n")
    print("Logit shape:", 
        logit["l"].shape if isinstance(logit, dict) else logit.shape)
    print("Hidden shape:", hidden.shape)
    print("Activ dict keys:", activ_dict.keys())


if __name__ == "__main__":
    main()
