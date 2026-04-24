import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class VisitAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_visits = args.max_visits
        self.d_model = args.embed_dim

        self.register_buffer(
            "temporal_encoding",
            self.generate_positional_encodings(self.max_visits, self.d_model)
        )

        # projection (kept flexible in case dims differ)
        self.projection = nn.Linear(args.embed_dim, args.embed_dim)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=args.embed_dim,
            nhead=args.num_heads,
            dropout=args.dropout,
            batch_first=False  # important because you permute
        )

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, visit_embeddings, mask):
        # (B, T, D)
        B, T, D = visit_embeddings.shape

        visit_embeddings = self.projection(visit_embeddings)
        visit_embeddings = self.dropout(visit_embeddings)

        # match sequence length dynamically
        visit_embeddings = visit_embeddings + self.temporal_encoding[:T]

        # True = ignore
        transformer_mask = (mask == 0) if mask is not None else None

        visit_embeddings = visit_embeddings.permute(1, 0, 2)  # (T, B, D)

        out = self.transformer(
            visit_embeddings,
            src_key_padding_mask=transformer_mask
        )

        out = out.permute(1, 0, 2)  # (B, T, D)
        return out

    @staticmethod
    def generate_positional_encodings(length, d_model):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe