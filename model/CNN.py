# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch


class _ConvLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_ConvLayer, self).__init__()

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('norm', nn.BatchNorm2d(num_output_features)),

        self.drop_rate = drop_rate

    def forward(self, x):
        x = super(_ConvLayer, self).forward(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('convlayer1', _ConvLayer(3, 32, 0.0))
        self.features.add_module('maxpool', nn.MaxPool2d(2, 2))
        self.features.add_module('convlayer3', _ConvLayer(32, 64, 0.0))
        self.features.add_module('avgpool', nn.AvgPool2d(2, 2))
        self.features.add_module('convlayer5', _ConvLayer(64, 128, 0.0))

        self.classifier = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, kernel_size=8, stride=1)
        out = F.dropout(out, 0.6, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def CNN5():
    return CNN()
