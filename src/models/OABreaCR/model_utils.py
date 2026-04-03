import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Feedforward(nn.Module):
    def __init__(self, inplace, outplace):
        super().__init__()

        self.conv1 = convBlock(inplace, outplace, kernel_size=3, padding=1)
        self.conv2 = convBlock(outplace, outplace, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class POELatent(nn.Module):
    """
    Adapted from: https://github.com/Li-Wanhua/POEs
    """
    def __init__(self, num_feat=2048):
        super().__init__()

        self.emd = nn.Sequential(
            nn.Linear(num_feat, num_feat),
            # nn.ReLU(True),
        )
        self.var = nn.Sequential(
            nn.Linear(num_feat, num_feat),
            # nn.BatchNorm1d(num_feat, eps=0.001, affine=False),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x, max_t=50, use_sto=True):
        emb = self.emd(x)
        log_var = self.var(x)
        sqrt_var = torch.exp(log_var * 0.5)
        if use_sto:
            rep_emb = emb.unsqueeze(0).expand(max_t, *emb.shape)
            rep_sqrt_var = sqrt_var.unsqueeze(0).expand(max_t, *sqrt_var.shape)
            norm_v = torch.randn_like(rep_emb)
            sto_emb = rep_emb + rep_sqrt_var * norm_v
            drop_emb = self.drop(sto_emb)
        else:
            drop_emb = self.drop(emb)
        return drop_emb, emb, log_var


class BaselineModel(nn.Module):
    def __init__(self, arch='resnet18'):
        super(BaselineModel, self).__init__()
        # create model
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
        # print(model)
        if 'densenet' in arch:
            num_feat = model.classifier.in_features
        elif 'resnet' in arch:
            num_feat = model.fc.in_features
        elif 'vgg' in arch:
            num_feat = model.classifier[-1].in_features
        elif 'convnext' in arch:
            num_feat = model.classifier[-1].in_features
        elif 'efficientnet' in arch:
            num_feat = model.classifier[-1].in_features

        if 'efficientnet' in arch or 'convnext' in arch:
            self.model = []
            for name, module in model.named_children():
                if name == 'avgpool':
                    continue
                if name == 'classifier':
                    continue
                self.model.append(module)
        else:
            self.model = []
            for name, module in model.named_children():
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    continue
                if isinstance(module, nn.Linear):
                    continue
                self.model.append(module)

        self.model = nn.Sequential(*self.model)
        self.num_feat = num_feat

    def forward(self, x):
        return self.model(x)

    def get_num_feat(self):
        return self.num_feat


class Simple_AttentionPool(nn.Module):
    """
    Pool to learn an attention over the slices
    Adapted from: https://github.com/reginabarzilaygroup/Sybil
    """
    def __init__(self, **kwargs):
        super(Simple_AttentionPool, self).__init__()
        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(kwargs['num_dim'])

    def forward(self, x):
        '''
        args:
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        '''
        output = {}
        B, C, W, H = x.shape

        spatially_flat_size = (B, C, -1)
        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1, 2))  # B, N, 1

        attention_map = self.norm(self.logsoftmax(attention_scores.transpose(1, 2)).view(B, -1)).view(B, 1, W, H)
        output['attention_map'] = attention_map
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # B, 1, N

        x = x * attention_scores  # B, C, N
        output['hidden'] = torch.sum(x, dim=-1)
        return output

