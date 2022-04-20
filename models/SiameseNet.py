import torch.nn as nn
from torchvision.models import *



class SiameseNet(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super(SiameseNet, self).__init__()

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = globals()[backbone](pretrained=pretrained, progress=pretrained)

    def forward(self, input):
        # forward pass
        output = self.backbone(input)
        return output
