"""
A lightweight reconstruction model for electroluminescence images.

This implementation provides a compact U-Net style autoencoder with skip
connections. It is intentionally small to keep training manageable while
serving as a functional placeholder for the broader pipeline.
"""

from typing import Any, Optional

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )


class Rec_model(nn.Module):
    """Compact U-Net style reconstruction network."""

    def __init__(self) -> None:
        super().__init__()
        self.down1 = conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = conv_block(64, 128)

        self.up2 = up_block(128, 64)
        self.up_block2 = conv_block(128, 64)

        self.up1 = up_block(64, 32)
        self.up_block1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, mode: Optional[int] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        # Bridge
        b = self.bridge(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1)

        out = self.final(u1)
        out = torch.sigmoid(out)
        return out
