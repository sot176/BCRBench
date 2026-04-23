import torch
import sys

def extract_mirai_backbone(path):
    sys.path.append('../')
    import models.Mirai.onconet as current_onconet

    # Patch legacy import paths (for old checkpoints)
    sys.modules['onconet'] = current_onconet
    sys.modules['onconet.models'] = current_onconet.models
    sys.modules['onconet.utils'] = current_onconet.utils
    sys.modules['onconet.models.custom_resnet'] = current_onconet.models.custom_resnet
    sys.modules['onconet.utils.risk_factors'] = current_onconet.utils.risk_factors
    
    # first pull mirai onto the CPU to avoid putting the whole transformer on the gpu
    mirai = torch.load(path, map_location='cpu', weights_only=False)
    embedding = []
    for l in mirai.children():
        for m in l.children():
            for name, n in m.named_children():
                embedding.append(n)
                if name == 'layer4_1':
                    break
            break
        break
    
    return torch.nn.Sequential(*embedding)


