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
        super().__init__()

        # -------------------------
        # Backbone Encoder
        # -------------------------
        self.model = BaselineModel(arch=args.arch)
        num_feat = self.model.get_num_feat()

        # -------------------------
        # Output heads
        # -------------------------
        self.final = nn.Linear(num_feat, args.num_output_neurons)
        self.final_single = nn.Linear(num_feat, args.num_output_neurons)
        self.difference_single = nn.Linear(num_feat, args.num_output_neurons)

        # -------------------------
        # Attention pooling
        # -------------------------
        self.pooling = SimpleAttentionPool(
            num_chan=num_feat,
            conv_pool_kernel_size=7,
            stride=1,
            num_dim=(args.img_size[0] // 32) * (args.img_size[1] // 32),
        )

        # -------------------------
        # Optional POE latent
        # -------------------------
        self.POE = getattr(args, "use_poe", True)
        if self.POE:
            self.POELatent = POELatent(num_feat=num_feat)
        self.max_t = getattr(args, "max_t", 50)
        self.use_sto = getattr(args, "use_sto", True)

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
    def get_risk_heads(self, outputs: Dict, batch: Dict) -> Dict:
        """
        Builds targets and masks for all heads.
        Used by both loss computation and evaluation.
        """
        y_true, y_mask = self.compute_risk_target_and_mask(
            outputs["final"],
            batch["years_to_cancer"],
            batch["years_to_last_followup"],
        )
        y_true_pri, y_mask_pri = self.compute_risk_target_and_mask(
            outputs["final"],
            batch["years_to_cancer_prior"],
            batch["years_to_last_followup_prior"],
        )
        return {
            "final":      (outputs.get("final"),      y_true,     y_mask),
            "current":    (outputs.get("current"),    y_true,     y_mask),
            "prior":      (outputs.get("prior"),      y_true_pri, y_mask_pri),
            "difference": (outputs.get("difference"), y_true,     y_mask),
        }
    
    def get_primary_risk_head(outputs: Dict[str, torch.Tensor], max_followup: int = 5) -> torch.Tensor:
        """Return softmax-normalized cumulative risk score."""
        risk = outputs["final"]
        if risk.dim() == 3:  # stochastic dimension
            risk = risk.mean(dim=0)
        prob = F.softmax(risk, dim=-1)
        score = prob_to_score(prob.detach().cpu().numpy(), max_followup=max_followup)
        return torch.tensor(score, device=prob.device, dtype=torch.float)
    
    @staticmethod
    def compute_risk_target_and_mask(pred, years_to_cancer,
                                     years_last_followup):
        if pred.dim() == 3:
            pred = pred.mean(dim=0)
        B, num_pred_years = pred.shape
        followup = num_pred_years - 1
        device   = years_to_cancer.device

        risk_label          = years_to_cancer.cpu().detach().numpy().copy()
        years_last_followup = years_last_followup.cpu().detach().numpy().copy()
        risk_label[risk_label > followup] = followup

        y_true = torch.zeros(B, num_pred_years, device=device)
        y_mask = torch.ones(B, num_pred_years, device=device)

        for i in range(B):
            y_true[i, int(risk_label[i])] = 1
            if risk_label[i] == followup and years_last_followup[i] < followup:
                y_mask[i, int(years_last_followup[i]) + 1:] = 0

        return y_true, y_mask