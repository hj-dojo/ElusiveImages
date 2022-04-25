import torch.nn as nn
import torchvision.models as tvmodels


class SiameseNet(nn.Module):
    def __init__(self, category, pretrained=True, class_dim=17):
        super(SiameseNet, self).__init__()

        # Create a backbone network from the pretrained models provided in torchvision.models
        class_ = getattr(tvmodels, category)
        self.backbone = class_(pretrained=pretrained, progress=pretrained)
        self.pretrained = pretrained

        if pretrained:
            # freeze pretrained model except for last layer
            layer_cnt = len(list(self.backbone.children()))
            layer_num = 1
            out_features = 1000
            for child in self.backbone.children():
                if hasattr(child, 'out_features'):
                    out_features = child.out_features
                elif isinstance(child, nn.Sequential):
                    if hasattr(child, 'head'):
                        out_features = child.head.out_features
                if layer_num != layer_cnt:
                    for param in child.parameters():
                        param.requires_grad = False
                layer_num += 1

            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=out_features, out_features=512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256)
            )

    def forward(self, input):
        # forward pass
        output = self.backbone(input)
        if self.pretrained:
            # fully connected layer
            output = self.fc_layer(output)
        return output
