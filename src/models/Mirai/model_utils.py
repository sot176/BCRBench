import torch
from torch import nn
import os

from models.common_parts import CumulativeProbabilityLayer

class GlobalMaxPool(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x, _ = torch.max(x, dim=-1)
        return None, x

    def replaces_fc(self):
        return False
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class ResNet18Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(args.num_chan, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.hidden_dim = 512
        args.hidden_dim = 512

        self.pool = GlobalMaxPool()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(512, args.num_classes)

        # Optional heads
        if getattr(args, 'use_region_annotation', False):
            self.region_fc = nn.Conv2d(512, 1, 1)

        if getattr(args, 'predict_birads', False):
            self.birads_fc = nn.Linear(512, 2)

        if getattr(args, 'survival_analysis_setup', False):
            self.prob_of_failure_layer = CumulativeProbabilityLayer(
                512,  max_followup=args.max_followup
            )

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        logit, hidden = self.pool(x)

        hidden = self.dropout(hidden)
        logit = self.fc(hidden)

        if getattr(self, "prob_of_failure_layer", None):
            logit = self.prob_of_failure_layer(hidden)

        activ_dict = {"activ": x}

        if hasattr(self, "region_fc"):
            activ_dict["region_logit"] = self.region_fc(x)

        if hasattr(self, "birads_fc"):
            activ_dict["birads_logit"] = self.birads_fc(hidden)

        return logit, hidden, activ_dict
    

def load_model(path, args, model_class):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # -------------------------------
    # Case 1: Full serialized model
    # -------------------------------
    if isinstance(checkpoint, nn.Module):

        model = checkpoint

        # unwrap DataParallel if present
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        return model

    # -------------------------------
    # Case 2: state_dict checkpoint
    # -------------------------------
    model = model_class(args)

    state_dict = (
        checkpoint.get("model")
        or checkpoint.get("state_dict")
        or checkpoint
    )

    # remove DP prefix
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict, strict=False)

    return model