import torch
import torch.nn as nn
import torch.nn.functional as F
from plainblock import Conv

class plain56(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1 (16 channels)
        self.first_stage = nn.Sequential(*[Conv(16, 16, 1) for _ in range(9)])

        # Stage 2 (32 channels)
        downsample = Conv(16, 32, 2)
        self.second_stage = nn.Sequential(downsample, *[Conv(32, 32, 1) for _ in range(8)])

        # Stage 3 (64 channels)
        downsample2 = Conv(32, 64, 2)
        self.third_stage = nn.Sequential(downsample2, *[Conv(64, 64, 1) for _ in range(8)])

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.first_stage(x)
        x = self.second_stage(x)
        x = self.third_stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x