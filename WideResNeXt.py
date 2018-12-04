import torch.nn as nn
from torchvision.models.resnet import ResNet


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=4):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * self.expansion,
                               kernel_size=1, padding=0, stride=stride, groups=1)
        self.bn1 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes * self.expansion, planes * self.expansion,
                               kernel_size=3,  padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.conv3 = nn.Conv2d(planes * self.expansion, planes * self.expansion,
                               kernel_size=3,  padding=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_wideresnext34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
