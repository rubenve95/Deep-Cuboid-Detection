from typing import Tuple

import torchvision
from torch import nn

import backbone.base


class VGG16(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)

        features = nn.Sequential(*list(vgg16.features._modules.values())[:-1])

        vgg16.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])

        # Fix the layers before conv3:
        # for layer in range(10):
        #   for p in features[layer].parameters(): p.requires_grad = False

        num_features_out = 512

        hidden = vgg16.classifier
        num_hidden_out = 4096


        return features, hidden, num_features_out, num_hidden_out
