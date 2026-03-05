import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_parts  import ContinuousPosEncoding, SpatialTransformerBlock
from .model_utils import Simple_AttentionPool, POELatent, Feedforward, BaselineModel


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
        self.max_t = getattr(args, 'max_t', 50)
        self.use_sto = getattr(args, 'use_sto', False)

    def forward(self, batch):

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
            x, emb, log_var = self.POELatent(x, max_t=self.max_t, use_sto=self.use_sto)
    
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

    def compute_risk_target_and_mask(self, pred, years_to_cancer, years_last_followup):
        """
        Converts scalar event times into cumulative binary target and mask.

        Args:
            years_to_cancer: Tensor [B]  (event year)
            years_last_followup: Tensor [B]  (last observed year)
            max_followup: int, max years

        Returns:
            y_true: [B, max_followup], 1 if event happened by year t
            y_mask: [B, max_followup], 1 if year t is observed, else 0
        """
        if pred.dim() == 3:
            pred = pred.mean(dim=0)

        B, num_pred_years = pred.shape
        max_index = num_pred_years - 1

        device = years_to_cancer.device

        y_true = torch.zeros(B, num_pred_years, device=device)
        y_mask = torch.ones(B, num_pred_years, device=device)

        for i in range(B):

            # Clamp event year so it never exceeds prediction range
            event_year = torch.clamp(years_to_cancer[i], max=max_index)

            # ---- ONE HOT TARGET (old behavior) ----
            y_true[i, event_year] = 1

            # ---- Mask future unobserved years (censoring case) ----
            if event_year == max_index and years_last_followup[i] < max_index:
                y_mask[i, years_last_followup[i] + 1:] = 0

        return y_true, y_mask
        

    def get_risk_heads(self, outputs, batch):
        heads = {}
        y_true, y_mask = self.compute_risk_target_and_mask(outputs['final'],
                batch['years_to_cancer'], batch['years_to_last_followup']
            )
        y_true_prior, y_mask_prior = self.compute_risk_target_and_mask(outputs['final'],
                batch['years_to_cancer_prior'], batch['years_to_last_followup_prior']
            )
        
        # Final/main head
        if 'final' in outputs and outputs['final'] is not None:
            heads['final'] = (outputs['final'], y_true, y_mask)

        # Current head
        if 'current' in outputs and outputs['current'] is not None:
            heads['current'] = (outputs['current'], y_true, y_mask)

        # Prior head
        if 'prior' in outputs and outputs['prior'] is not None:
            heads['prior'] = (outputs['prior'], y_true_prior, y_mask_prior)

        # Difference head
        if 'difference' in outputs and outputs['difference'] is not None:
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
        risk = outputs["final"]
        if risk.dim()==3:
            risk = risk.mean(dim=0)    # average over stochastic dimension
        return risk
    
    