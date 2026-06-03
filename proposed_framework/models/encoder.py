# proposed_framework/models/encoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, drop2d: float = 0.0):
        super().__init__()

        groups = 8 if channels % 8 == 0 else 4

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

        self.act = nn.ReLU(inplace=True)

        if drop2d and drop2d > 0:
            self.drop = nn.Dropout2d(drop2d)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))

        return self.act(x + h)


class MicroResNetEncoder2D3Ch(nn.Module):
    """
    Micro-ResNet encoder for three-channel 5x8 pulse-response inputs.

    Input shape:
        (B, 3, 5, 8)

    Output shape:
        (B, width)
    """

    def __init__(self, width: int = 32, blocks: int = 4, drop2d: float = 0.0):
        super().__init__()

        groups = 8 if width % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[ResBlock(width, drop2d=drop2d) for _ in range(int(blocks))]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        z = self.stem(x_img)
        z = self.body(z)
        z = self.pool(z).flatten(1)

        return z