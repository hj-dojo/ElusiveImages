import torch.nn as nn
import torchvision.models as tvmodels


class SiameseNet(nn.Module):
    def __init__(self, category, pretrained=True, output_dim=512):
        super(SiameseNet, self).__init__()

        # Create a backbone network from the pretrained models provided in torchvision.models
        class_ = getattr(tvmodels, category)
        self.backbone = class_(pretrained=pretrained, progress=pretrained)
        self.pretrained = pretrained
        self.fc_layer = None
        if pretrained:
            out_features = None
            for layer in self.backbone.children():
                if hasattr(layer, 'out_features'):
                    out_features = layer.out_features
                elif isinstance(layer, nn.Sequential):
                    out_features = layer.head.out_features
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=out_features, out_features=output_dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(in_features=output_dim, out_features=output_dim, bias=True),
            )
            # freeze pretrained model
            for child in self.backbone.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, input):
        # forward pass
        output = self.backbone(input)
        # fully connected layer
        if self.pretrained:
            output = self.fc_layer(output)
        return output
