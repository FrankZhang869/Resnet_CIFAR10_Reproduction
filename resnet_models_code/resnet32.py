import torch
import torch.nn as nn
import torch.nn.functional as F
from resnetblock import resnet_block

class resnet32 (nn.Module):
    def __init__ (self):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)

        self.first_stage = nn.Sequential(*[resnet_block(16, 16, 1) for _ in range(5)])

        down_sample = resnet_block(16, 32, 2)
        self.second_stage = nn.Sequential(down_sample, *[resnet_block(32, 32, 1) for _ in range(4)])

        down_sample2 = resnet_block(32, 64, 2)
        self.third_stage = nn.Sequential(down_sample2, *[resnet_block(64, 64, 1) for _ in range(4)])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
    def forward (self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.first_stage(x)
        x = self.second_stage(x)
        x = self.third_stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x