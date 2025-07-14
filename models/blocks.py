import torch.nn as nn
from abc import ABC, abstractmethod

# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view(in_tensor.size()[0], -1)

# ------------------------ ClassifierBlock Class ---------------------
class ClassifierBlock(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.5, activation=None):
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()
        self.linear = nn.Linear(in_features, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        out = self.avg_pooling(x)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.activation(out)
        return out

    def get_activation(self, name):
        if name is None:
            return nn.Identity()
        name = name.lower()
        if name == 'softmax':
            return nn.Softmax(dim=1)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {name}")

# ------------------------ ConvBlock Class ---------------------
class VGG_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, maxpool=False):
        super().__init__()
        padding = int(kernel_size > 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class ResNet_ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, offset=0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels + offset, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.relu(out)

# ------------------------ ReduceBlock Class ---------------------
class ReduceBlock(nn.Module):
    def __init__(self, in_channels, red1, red2, architecture='vgg'):
        super(ReduceBlock, self).__init__()
        self.architecture = architecture.lower()

        mid_channels = in_channels // red1
        out_channels = max(1, mid_channels // red2)

        if architecture == 'vgg':
            ks1 = 1
            ks2 = 3 if red2 > 10 else 1
        elif architecture == 'resnet':
            ks1 = 1
            ks2 = 3 if red2 > 10 else 1 
            # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("Unsupported architecture for ReduceBlock")

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ks1, stride=1, padding=ks1 // 2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=ks2, stride=1, padding=ks2 // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.reduce(x)
        # if self.architecture == 'resnet': x = self.pool(x)
        return x