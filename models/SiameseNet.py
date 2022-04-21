import torch.nn as nn
import torchvision.models as tvmodels



class SiameseNet(nn.Module):
    def __init__(self, category, pretrained=True):
        super(SiameseNet, self).__init__()

        # Create a backbone network from the pretrained models provided in torchvision.models
        class_ = getattr(tvmodels, category)
        self.backbone = class_(pretrained=pretrained, progress=pretrained)

    def forward(self, input):
        # forward pass
        output = self.backbone(input)
        return output
