
import torch
from torch import nn

class TransformerVMRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()

        self.proj_in = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.history_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.reconstruction_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, sequence, exam_mask):
        """
        sequence: (B, T, D)
        exam_mask: (B, T)
        """

        x = self.proj_in(sequence)  # (B, T, H)

        # causal/valid masking
        key_padding_mask = ~exam_mask  # Transformer expects True = ignore

        x = self.transformer(
            x,
            src_key_padding_mask=key_padding_mask
        )

        states = self.history_projection(x)
        reconstructions = self.reconstruction_projection(x)

        # right-aligned last valid index (same as original repo)
        last_idx = exam_mask.size(1) - 1 - torch.argmax(
            exam_mask.flip(dims=[1]).int(), dim=1
        )

        batch_idx = torch.arange(x.size(0), device=x.device)
        history_embedding = states[batch_idx, last_idx]

        return history_embedding, states, reconstructions