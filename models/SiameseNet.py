import torch.nn as nn
from torchvision import models


class SiameseNet(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super(SiameseNet, self).__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=pretrained, progress=pretrained)

    def forward(self, input):
        # forward pass
        output = self.backbone(input)
        return output
