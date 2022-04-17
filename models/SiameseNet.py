import torch.nn as nn
from torchvision import models


class SiameseNet(nn.Module):
    def __init__(self, backbone):
        super(SiameseNet, self).__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

    def forward_once(self, x):
        # Forward pass
        output = self.backbone(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        # returning the feature vectors of two inputs
        return output1, output2
