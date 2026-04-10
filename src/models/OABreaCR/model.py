import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.common_parts import ContinuousPosEncoding, SpatialTransformerBlock
from .model_utils import SimpleAttentionPool, POELatent, Feedforward, BaselineModel, prob_to_score
from models.common_parts import BaseRiskModel


class OA_BreaCR(BaseRiskModel):
    """
    Longitudinal breast cancer risk prediction model with optional probabilistic latent embeddings.
    """
    def __init__(self, args):
        super().__init__(args)

        # -------------------------
        # Backbone Encoder
        # -------------------------
        self.model = BaselineModel(arch=self.args.arch)
        num_feat = self.model.get_num_feat()

        # -------------------------
        # Output heads
        # -------------------------
        self.final = nn.Linear(num_feat, self.args.num_output_neurons)
        self.final_single = nn.Linear(num_feat, self.args.num_output_neurons)
        self.difference_single = nn.Linear(num_feat, self.args.num_output_neurons)

        # -------------------------
        # Attention pooling
        # -------------------------
        self.pooling = SimpleAttentionPool(
            num_chan=num_feat,
            conv_pool_kernel_size=7,
            stride=1,
            num_dim=(self.args.img_size[0] // 32) * (self.args.img_size[1] // 32),
        )

        # -------------------------
        # Optional POE latent
        # -------------------------
        self.POE = getattr(self.args, "use_poe", True)
        if self.POE:
            self.POELatent = POELatent(num_feat=num_feat)
        self.max_t = getattr(self.args, "max_t", 50)
        self.use_sto = getattr(self.args, "use_sto", True)

        # -------------------------
        # Registration and alignment
        # -------------------------
        self.reg_transformer = SpatialTransformerBlock(mode='bilinear')
        self.flew = Feedforward(in_channels=2, out_channels=2)
        self.pos_encoding = ContinuousPosEncoding(dim=num_feat)

        # -------------------------
        # Feature fusion MLP
        # -------------------------
        self.mlp = nn.Sequential(
            nn.Linear(num_feat * 3, num_feat),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(num_feat, num_feat)
        )

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target_x = batch["current_image"]
        prior_x = batch["previous_image"]
        time_gap = batch["time_gap"]

        # -------------------------
        # Ensure 3 channels
        # -------------------------
        target_x = target_x.repeat(1, 3, 1, 1) if target_x.shape[1] == 1 else target_x
        prior_x = prior_x.repeat(1, 3, 1, 1) if prior_x.shape[1] == 1 else prior_x

        # -------------------------
        # Encode features
        # -------------------------
        feat_current = self.model(target_x)
        feat_prior = self.model(prior_x)

        # -------------------------
        # Attention pooling
        # -------------------------
        hidden_current = self.pooling(feat_current)
        hidden_prior = self.pooling(feat_prior)

        attention_current = hidden_current['attention_map']
        attention_prior = hidden_prior['attention_map']

        # -------------------------
        # Compute flow field and register
        # -------------------------
        flow_field = self.flew(torch.cat([attention_current, attention_prior], dim=1))
        moved_prior = self.reg_transformer(feat_prior, flow_field)
        diff_feat = torch.abs(feat_current - moved_prior)
        hidden_diff = self.pooling(diff_feat)

        # -------------------------
        # Individual predictions
        # -------------------------
        logit_current = self.final_single(hidden_current['hidden'])
        logit_prior = self.final_single(hidden_prior['hidden'])

        diff_hidden_encoded = self.pos_encoding(
            hidden_diff['hidden'].unsqueeze(0), time_gap
        ).squeeze(0)
        logit_diff = self.difference_single(diff_hidden_encoded)

        # -------------------------
        # Concatenate features and MLP
        # -------------------------
        fused_feat = torch.cat([
            hidden_current['hidden'],
            hidden_prior['hidden'],
            diff_hidden_encoded
        ], dim=1)
        fused_feat = self.mlp(fused_feat)

        # -------------------------
        # Optional POE stochastic embedding
        # -------------------------
        if self.POE:
            fused_feat, emb, log_var = self.POELatent(fused_feat, max_t=self.max_t, use_sto=self.use_sto)
        else:
            emb, log_var = None, None

        # -------------------------
        # Final prediction
        # -------------------------
        logit = self.final(fused_feat)

        # -------------------------
        # Registration loss
        # -------------------------
        reg_loss = self.compute_reg_loss(attention_current, self.reg_transformer(attention_prior, flow_field))

        return {
            'final': logit,
            'current': logit_current,
            'prior': logit_prior,
            'difference': logit_diff,
            'emb_final': emb,
            'log_var_final': log_var,
            'flow_field': flow_field,
            'loss': reg_loss,
        }

    # -------------------------
    # Loss helpers
    # -------------------------
    @staticmethod
    def compute_reg_loss(x: torch.Tensor, target_x_source: torch.Tensor) -> torch.Tensor:
        """MSE loss for registration alignment."""
        return torch.mean((x - target_x_source) ** 2) * 1e-2

    # -------------------------
    # Risk head
    # -------------------------
    
    def get_risk_heads(self, outputs, batch):
        return {
            "final": {
                "risk": outputs.get("final"),
                "risk_label": batch["years_to_cancer"],
                "years_lfu": batch["years_to_last_followup"],
                "emb": outputs.get("emb_final"),
                "log_var": outputs.get("log_var_final"),
                "weight": 1.0,
            },
            "current": {
                "risk": outputs.get("current"),
                "risk_label": batch["years_to_cancer"],
                "years_lfu": batch["years_to_last_followup"],
                "emb": outputs.get("emb_current"),
                "log_var": outputs.get("log_var_current"),
                "weight": 0.2,
            },
            "prior": {
                "risk": outputs.get("prior"),
                "risk_label": batch["years_to_cancer_prior"],
                "years_lfu": batch["years_to_last_followup_prior"],
                "emb": outputs.get("emb_prior"),
                "log_var": outputs.get("log_var_prior"),
                "weight": 0.2,
            },
            "difference": {
                "risk": outputs.get("difference"),
                "risk_label": batch["years_to_cancer"],
                "years_lfu": batch["years_to_last_followup"],
                "emb": outputs.get("emb_difference"),
                "log_var": outputs.get("log_var_difference"),
                "weight": 0.2,
            },
        }

    def get_primary_risk_head(self, outputs, max_followup = 5):
        """Return softmax-normalized cumulative risk score."""
        risk = outputs["final"]
        if risk.dim() == 3:  # stochastic dimension
            risk = risk.mean(dim=0)
        prob = F.softmax(risk, dim=-1)
        score = prob_to_score(prob.detach().cpu().numpy(), max_followup=max_followup)
        return torch.tensor(score, device=prob.device, dtype=torch.float)
    
   