"""Enhanced reconstruction model mirroring the original research code.

The repository initially shipped with a simplified placeholder model. This
module reintroduces the U-Net inspired architecture from the author's code,
including the iterative encoder/decoder passes controlled by ``count``. The
design prefers additive skip connections to keep channel sizes compact and
retains InstanceNorm for stability on small batch sizes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


def _conv_block(in_channels: int, features: int, name: str) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class _UNetStage(nn.Module):
    """U-Net like stage with additive skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, init_features: int = 32) -> None:
        super().__init__()
        features = init_features
        self.encoder1 = _conv_block(in_channels, features, name="enc1")
        self.pool1 = nn.Conv2d(features, features, kernel_size=2, stride=2)
        self.encoder2 = _conv_block(features, features * 2, name="enc2")
        self.pool2 = nn.Conv2d(features * 2, features * 2, kernel_size=2, stride=2)
        self.encoder3 = _conv_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.Conv2d(features * 4, features * 4, kernel_size=2, stride=2)
        self.encoder4 = _conv_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.Conv2d(features * 8, features * 8, kernel_size=2, stride=2)

        self.bottleneck = _conv_block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = _conv_block(features * 8, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = _conv_block(features * 4, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = _conv_block(features * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = _conv_block(features, features, name="dec1")

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck) + enc4
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4) + enc3
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3) + enc2
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2) + enc1
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))


class Rec_model(nn.Module):
    """Iterative reconstruction model with stacked U-Net stages."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, init_features: int = 32) -> None:
        super().__init__()
        self.unet_down = _UNetStage(in_channels, out_channels, init_features)
        self.unet_up = _UNetStage(out_channels, out_channels, init_features)

    def forward(self, x: torch.Tensor, count: Optional[int] = None) -> torch.Tensor:  # type: ignore[override]
        """Run ``count`` iterations of the down and up stages.

        If ``count`` is ``None`` (default) the model performs a single pass.
        If ``count`` is ``-1``, a random number of iterations in ``[1, 5]`` is
        sampled to mimic the behavior of the original script.
        """

        if count is None:
            count = 1
        if count == -1:
            count = int(torch.randint(1, 6, (1,))[0])

        for _ in range(count):
            x = self.unet_down(x)

        for _ in range(count):
            x = self.unet_up(x)

        return x
