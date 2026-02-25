import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.common_parts  import ContinuousPosEncoding, SpatialTransformerBlock
from .model_utils import Simple_AttentionPool, POELatent, Feedforward, BaselineModel


class OA_BreaCR_FeatAlign(nn.Module):
    def __init__(self,  args):
        super(OA_BreaCR_FeatAlign, self).__init__()
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

        shape = [int(args.img_size[0] / 32), int(args.img_size[1] / 32)]
        self.reg_transformer = SpatialTransformerBlock(shape)
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

    def forward(self, target_x, prior_x=None, time=None, **kwargs):

        mask = kwargs['mask'] if 'mask' in kwargs else None
        prior_mask = kwargs['prior_mask'] if 'prior_mask' in kwargs else None

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
            'loss': loss,
            'flow_field': flow_field,
        }


