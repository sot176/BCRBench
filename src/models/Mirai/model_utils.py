import torch
from torch import nn
import os

from common_parts import Cumulative_Probability_Layer

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
            self.prob_of_failure_layer = Cumulative_Probability_Layer(
                512, args, max_followup=args.max_followup
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
    
class TransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(args.hidden_dim)
        self.norm2 = nn.LayerNorm(args.hidden_dim)

        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ff = self.fc2(self.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(ff))

        return x
    
class SimpleTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.projection = nn.Linear(args.precomputed_hidden_dim, args.hidden_dim)

        self.layers = nn.ModuleList(
            [TransformerLayer(args) for _ in range(args.num_layers)]
        )

        self.pool = GlobalMaxPool()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_classes)

        if getattr(args, 'survival_analysis_setup', False):
            self.prob_of_failure_layer = Cumulative_Probability_Layer(
                args.hidden_dim, args, max_followup=args.max_followup
            )

    def forward(self, x):
        x = self.projection(x)

        for layer in self.layers:
            x = layer(x)

        img_like = x.transpose(1, 2).unsqueeze(-1)

        logit, hidden = self.pool(img_like)

        hidden = self.dropout(hidden)
        logit = self.fc(hidden)

        if hasattr(self, "prob_of_failure_layer"):
            logit = self.prob_of_failure_layer(hidden)

        return logit, x, {}
    


def load_model(path, args, model_class):
    """
    Loads a model from a checkpoint path.

    Args:
        path (str): Path to checkpoint (.pt or .pth)
        args: argparse args used to instantiate the model
        model_class: Class of the model to instantiate
                     (e.g. ResNet18Backbone or SimpleTransformer)

    Returns:
        model (nn.Module)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # Case 1: Full model was saved directly
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        return model

    # Case 2: Checkpoint dict
    if isinstance(checkpoint, dict):

        # Instantiate fresh model
        model = model_class(args)

        # Common keys
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove DataParallel prefix if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        return model

    raise RuntimeError(f"Invalid checkpoint format at {path}")