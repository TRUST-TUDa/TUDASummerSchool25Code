import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import datetime

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out
    

class ResNet18Light(nn.Module):
    def __init__(self, name='', num_blocks=[2, 2, 2, 2], num_classes=10):
        
        super(ResNet18Light, self).__init__()
        self.created_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        self.name = f'{name}_ResNet_18'
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Head(nn.Module):
    def __init__(self, name=''):
        
        super(Head, self).__init__()
        self.created_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        self.name = f'{name}_ResNet_18_Head'

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        return out

class Tail(nn.Module):
    def __init__(self, name='', num_classes=10):
        
        super(Tail, self).__init__()
        self.created_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        self.name = f'{name}_ResNet_18_Tail'

        self.linear = nn.Linear(256*BasicBlock.expansion, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out

class Backbone(nn.Module):
    def __init__(self, name='', num_blocks=[2, 2, 2, 2]):
        
        super(Backbone, self).__init__()
        self.created_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        self.name = f'{name}_ResNet_18_Backbone'
        self.in_planes = 32

        self.layer1 = self._make_layer(32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        return out