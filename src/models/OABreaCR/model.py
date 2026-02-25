import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_parts  import ContinuousPosEncoding, SpatialTransformerBlock
from .model_utils import Simple_AttentionPool, POELatent, Feedforward, BaselineModel
from utils import get_risk_loss_BCE
from utils import MeanVarianceLoss, ProbOrdiLoss


class OA_BreaCR(nn.Module):
    def __init__(self,  args):
        super(OA_BreaCR, self).__init__()
        # create model
        model = BaselineModel(arch=args.arch)  # Encoder
        num_feat = model.get_num_feat()
        self.model = model
        self.final = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
        )  # output layer

        self.pooling = Simple_AttentionPool(
            num_chan=num_feat,
            conv_pool_kernel_size=7,
            stride=1,
            num_dim=int(args.img_size[0] / 32) * int(args.img_size[1] / 32),
        )
        if args.use_poe:
            self.POE = True
            self.POELatent = POELatent(num_feat=num_feat)
        else:
            self.POE = False

        self.reg_transformer = SpatialTransformerBlock(mode='bilinear')
        self.flew = Feedforward(inplace=2, outplace=2)
        self.pos_encoding = ContinuousPosEncoding(dim=num_feat, drop=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat * 3, num_feat),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(num_feat, num_feat), )

        self.final_single = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
        )  # output layer

        self.difference_single = nn.Sequential(
            nn.Linear(num_feat, args.num_output_neurons),
        )  # output layer

        # Instantiate MV and POE losses once
        self.MV_loss = MeanVarianceLoss()
        self.POE_loss = ProbOrdiLoss()

    def forward(self, batch, **kwargs):

        target_x = batch["current_image"]
        prior_x = batch["previous_image"]
        time = batch["time_gap"]

        mask =  None
        prior_mask =  None

        if mask is None:
            x = torch.cat([target_x, target_x, target_x], dim=1)
        else:
            x = torch.cat([(mask - 0.5) * 2, target_x * mask, target_x], dim=1)

        if prior_mask is None:
            prior_x = torch.cat([prior_x, prior_x, prior_x], dim=1)
        else:
            prior_x = torch.cat([(prior_mask - 0.5) * 2, prior_x, prior_x * prior_mask], dim=1)

        x = self.model(x)  # current feature
        prior_x = self.model(prior_x)  # prior feature

        hidden_x = self.pooling(x)
        hidden_prior_x = self.pooling(prior_x)

        attention_map_x = hidden_x['attention_map']
        attention_map_prior_x = hidden_prior_x['attention_map']

        b, c, w, h = attention_map_x.shape
        x_prior_x_ = torch.cat([attention_map_x, attention_map_prior_x], dim=1)
        flow_field = self.flew(x_prior_x_)
        target_x_source_ = self.reg_transformer(attention_map_prior_x, flow_field)
        loss = self.compute_reg_loss(attention_map_x, target_x_source_)
        moved_prior_x = self.reg_transformer(prior_x, flow_field)  # aligned prior feature
        difference = torch.abs(x - moved_prior_x)  # difference feature
        hidden_difference = self.pooling(difference)

        x_hidden_feat = hidden_x['hidden']
        logit_current = self.final_single(x_hidden_feat)

        prior_x_hidden_feat = hidden_prior_x['hidden']
        logit_prior = self.final_single(prior_x_hidden_feat)

        differencehidden_feat = hidden_difference['hidden'].view(1, b, -1)
        differencehidden_feat = self.pos_encoding(differencehidden_feat, time).view(b, -1)
        logit_difference = self.difference_single(differencehidden_feat)

        x = torch.cat([x_hidden_feat, prior_x_hidden_feat, differencehidden_feat], dim=1)

        x = self.mlp(x)

        if self.POE:
            max_t = kwargs['max_t'] if 'max_t' in kwargs else 50
            use_sto = kwargs['use_sto'] if 'use_sto' in kwargs else True
            x, emb, log_var = self.POELatent(x, max_t=max_t, use_sto=use_sto)
        else:
            emb, log_var = None, None

        logit = self.final(x)
        return {
            'final': logit,
            'current': logit_current,
            'prior': logit_prior,
            'difference': logit_difference,
            'emb_final': emb,
            'log_var_final': log_var,
            'flow_field': flow_field,
            'loss': loss,
        }
    
    def compute_reg_loss(self, x, target_x_source):
        loss_t1 = torch.mean((x - target_x_source) ** 2)
        return loss_t1 * 1e-2

    def compute_risk_target_and_mask(years_to_cancer, years_last_followup, max_followup):
        """
        Converts scalar event times into cumulative binary target and mask.

        Args:
            years_to_cancer: Tensor [B]
            years_last_followup: Tensor [B]
            max_followup: int, max years

        Returns:
            y_true: [B, max_followup], 1 if event happened by year t
            y_mask: [B, max_followup], 1 if year t is observed, else 0
        """
        B = years_to_cancer.shape[0]
        y_true = torch.zeros(B, max_followup, device=years_to_cancer.device)
        y_mask = torch.ones(B, max_followup, device=years_to_cancer.device)

        years_to_cancer = years_to_cancer.clamp(0, max_followup-1)
        years_last_followup = years_last_followup.clamp(0, max_followup-1)

        for i in range(B):
            y_true[i, :years_to_cancer[i]+1] = 1
            if years_to_cancer[i] == max_followup-1 and years_last_followup[i] < max_followup-1:
                y_mask[i, years_last_followup[i]+1:] = 0

        return y_true, y_mask

    def get_risk_heads(self, outputs, batch):
        max_followup = 6
        heads = {}

        # Final/main head
        if 'final' in outputs and outputs['final'] is not None:
            y_true, y_mask = self.compute_risk_target_and_mask(
                batch['years_to_cancer'], batch['years_to_last_followup'], max_followup
            )
            heads['final'] = (outputs['final'], y_true, y_mask)

        # Current head
        if 'current' in outputs and outputs['current'] is not None:
            y_true, y_mask = self.compute_risk_target_and_mask(
                batch['years_to_cancer'], batch['years_to_last_followup'], max_followup
            )
            heads['current'] = (outputs['current'], y_true, y_mask)

        # Prior head
        if 'prior' in outputs and outputs['prior'] is not None:
            y_true, y_mask = self.compute_risk_target_and_mask(
                batch['years_to_cancer_prior'], batch['years_to_last_followup_prior'], max_followup
            )
            heads['prior'] = (outputs['prior'], y_true, y_mask)

        # Difference head
        if 'difference' in outputs and outputs['difference'] is not None:
            y_true, y_mask = self.compute_risk_target_and_mask(
                batch['years_to_cancer'], batch['years_to_last_followup'], max_followup
            )
            heads['difference'] = (outputs['difference'], y_true, y_mask)

        return heads
        

    def get_auxiliary_outputs(self, outputs):
        """
        Returns auxiliary outputs for additional losses
        (e.g., KL divergence if using POE).
        """
        return {
            "emb": outputs["emb_final"],
            "log_var": outputs["log_var_final"],
        }

    def get_primary_risk_head(self, outputs):
        """
        Returns the main prediction head used for evaluation.
        """
        return outputs["final"]
    
    def compute_total_loss(self, outputs, batch):
        total_loss = 0.0

        # --- optional extra loss ---
        if outputs.get('loss') is not None:
            total_loss += outputs['loss']

        # --- 1️⃣ BCE loss for all heads ---
        risk_heads = self.get_risk_heads(outputs, batch)
        for head_name, (logits, target, mask) in risk_heads.items():
            weight = 1.0 if head_name == 'final' else 0.2
            if logits is not None:
                total_loss += weight * get_risk_loss_BCE(logits, target, mask)

        # --- 2️⃣ MV loss for main/final head ---
        total_loss += 0.2* self.MV_loss(
            self.get_primary_risk_head(outputs),
            batch['years_to_cancer'],
            batch['years_to_last_followup'],
            weights=getattr(self.args, 'time_to_events_weights', None)
        )

        # --- 3️⃣ POE loss for main/final head ---
        aux = self.get_auxiliary_outputs(outputs)
        emb, log_var = aux['emb'], aux['log_var']
        if emb is not None:
            _, _, _, loss_POE = self.POE_loss(
                self.get_primary_risk_head(outputs),
                emb,
                log_var,
                batch['years_to_cancer'],
                batch['years_to_last_followup'],
                None,
                use_sto=self.args.use_sto,
                weights=getattr(self.args, 'time_to_events_weights', None)
            )
            total_loss += 0.2* loss_POE

        return total_loss