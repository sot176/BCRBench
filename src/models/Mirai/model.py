

import torch
import torch.nn as nn
import sys

from models.common_parts import BaseRiskModel
from models.common_parts import extract_mirai_backbone
from config.config import cfg

# ----------------------------------------------------
# Register OncoNet aliases
# ----------------------------------------------------
from . import onconet as _onconet
sys.modules.setdefault("onconet", _onconet)

for _key in list(sys.modules.keys()):
    if _key.startswith("models.Mirai.onconet"):
        sys.modules.setdefault(
            _key.replace("models.Mirai.onconet", "onconet"),
            sys.modules[_key]
        )

from .onconet.models.factory import get_model_by_name, load_model


# ----------------------------------------------------
# Constants
# ----------------------------------------------------
MAX_FOLLOWUP = 5

FORMAL_VIEW_SEQUENCE = (
    ("LCC", 0, 1),
    ("RCC", 0, 0),
    ("LMLO", 1, 1),
    ("RMLO", 1, 0),
)


# ====================================================
# Mirai
# ====================================================
class Mirai(BaseRiskModel):

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.num_years = getattr(args, "num_years", MAX_FOLLOWUP)

        # ------------------------------------------------
        # Pretrained image encoder
        # ------------------------------------------------
        self.image_encoder = extract_mirai_backbone(
            cfg["paths"]["mirai_path"]
        )

        self.hidden_dim = args.transformer_hidden_dim

        # freeze encoder (official Mirai stage 2 behavior)
        self._freeze_encoder(self.image_encoder)

        # ------------------------------------------------
        # Transformer fusion module
        # ------------------------------------------------
        self.args.precomputed_hidden_dim = self.hidden_dim
        self.transformer = self._init_transformer(args)

        # ------------------------------------------------
        # Risk head
        # ------------------------------------------------
        self.risk_head = nn.Linear(
            self.hidden_dim,
            self.num_years
        )

    # =================================================
    # Helpers
    # =================================================
    @staticmethod
    def _freeze_encoder(module):
        for p in module.parameters():
            p.requires_grad = False
        module.eval()

    def _init_transformer(self, args):

        if getattr(args, "transformer_snapshot", None):
            return load_model(
                args.transformer_snapshot,
                args,
                do_wrap_model=False
            )

        return get_model_by_name(
            "transformer",
            False,
            args
        )

    def _make_transformer_batch(
        self,
        batch_size,
        device
    ):
        view_seq = torch.tensor(
            [x[1] for x in FORMAL_VIEW_SEQUENCE],
            device=device,
            dtype=torch.long
        )

        side_seq = torch.tensor(
            [x[2] for x in FORMAL_VIEW_SEQUENCE],
            device=device,
            dtype=torch.long
        )

        return {
            "time_seq": torch.zeros(
                batch_size,
                4,
                device=device,
                dtype=torch.long
            ),
            "view_seq": view_seq.unsqueeze(0).expand(batch_size, -1),
            "side_seq": side_seq.unsqueeze(0).expand(batch_size, -1),
        }

    # =================================================
    # Forward
    # =================================================
    def forward(self, batch):
        """
        images: (B,T,4,C,H,W)
        view_mask: (B,T,4)
        """

        images = batch["images"]
        view_mask = batch["view_mask"]

        B, T, V, C, H, W = images.shape

        assert V == 4, "Formal Mirai requires exactly 4 views"

        # ---------------------------------------------
        # flatten all images
        # ---------------------------------------------
        x = images.reshape(B * T * V, C, H, W)

        # ---------------------------------------------
        # encode all images independently
        # ---------------------------------------------
        img_x = self.image_encoder(x)

        # shape -> (B,T,4,D)
        img_x = img_x.reshape(B, T, V, -1)

        # ---------------------------------------------
        # process each exam independently
        # ---------------------------------------------
        exam_embeddings = []

        for t in range(T):

            tokens = img_x[:, t]        # (B,4,D)
            mask_t = view_mask[:, t]    # (B,4)

            # require complete exams
            valid_exam = mask_t.all(dim=1)

            emb = torch.zeros(
                B,
                self.hidden_dim,
                device=images.device
            )

            if valid_exam.any():

                valid_tokens = tokens[valid_exam]

                transformer_batch = self._make_transformer_batch(
                    valid_tokens.size(0),
                    images.device
                )

                # official transformer forward
                _, pooled_hidden, _ = self.transformer(
                    valid_tokens,
                    None,
                    transformer_batch
                )

                emb[valid_exam] = pooled_hidden

            exam_embeddings.append(emb)

        # (B,T,D)
        exam_embeddings = torch.stack(
            exam_embeddings,
            dim=1
        )

        # ---------------------------------------------
        # use latest available exam
        # ---------------------------------------------
        patient_embedding = exam_embeddings[:, -1]

        # ---------------------------------------------
        # multi-year risk logits
        # ---------------------------------------------
        logit = self.risk_head(patient_embedding)

        return logit, patient_embedding, {}

    # =================================================
    # Heads
    # =================================================
    def get_risk_heads(self, outputs, batch):
        logit, _, _ = outputs

        return {
            "logit_output": (
                logit,
                batch["target"],      # (B,5)
                batch["y_mask"]
            )
        }

    def get_primary_risk_head(self, outputs):
        logit, _, _ = outputs
        return torch.sigmoid(logit)