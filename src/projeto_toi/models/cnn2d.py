"""CNN 2D simples para classificar janelas POS/NEG."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class CNN2DPosNeg(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):  # type: ignore[override]
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model(in_ch: int = 4) -> CNN2DPosNeg:
    return CNN2DPosNeg(in_ch=in_ch)


__all__ = ["CNN2DPosNeg", "build_model"]
