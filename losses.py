# losses.py

from typing import Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class GradientNet(nn.Module):
    """
    Simple Sobel gradient module used for gradient-based loss.
    """

    def __init__(self) -> None:
        super().__init__()
        kernel_x = [[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0]]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

        # Buffers are moved with .to(device) on the module if needed
        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        returns gradient magnitude with shape (B, 1, H-2, W-2)
        """
        weight_x = self.weight_x.to(x.device)
        weight_y = self.weight_y.to(x.device)

        grad_x = F.conv2d(x, weight_x)
        grad_y = F.conv2d(x, weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


# Single shared instance (no trainable params, safe to reuse)
_gradient_net = GradientNet()


def compute_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradient using the shared GradientNet.
    """
    return _gradient_net(x)


def grayscale_l1_loss(output_rgb: torch.Tensor, label_rgb: torch.Tensor) -> torch.Tensor:
    """
    L1 loss between grayscale versions of two RGB images.
    """
    output_gray = TF.rgb_to_grayscale(output_rgb)
    label_gray = TF.rgb_to_grayscale(label_rgb)

    loss_fn = nn.L1Loss()
    loss = loss_fn(output_gray, label_gray)
    return loss


class ReconstructionLoss(nn.Module):
    """
    Combined L1 reconstruction loss + gradient loss (channel-wise).
    """

    def __init__(self, l1_weight: float = 1.0, gradient_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        prediction: (B, 3, H, W)
        target:     (B, 3, H, W)
        """
        # Standard L1 loss on RGB images
        l1 = self.l1_loss(prediction, target)

        # Gradient loss per channel
        gradient_loss = 0.0
        for c in range(3):
            pred_c = prediction[:, c : c + 1, :, :]
            target_c = target[:, c : c + 1, :, :]

            grad_pred = compute_gradient(pred_c)
            grad_target = compute_gradient(target_c)

            gradient_loss = gradient_loss + self.l1_loss(grad_pred, grad_target)

        gradient_loss = gradient_loss / 3.0
        total = self.l1_weight * l1 + self.gradient_weight * gradient_loss
        return total


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image from range [0,1] and shape (C, H, W)
    to a PIL image.
    """
    tensor = tensor.detach().cpu() * 255.0
    arr = tensor.numpy().astype(np.uint8)

    # (C, H, W) -> (H, W, C)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    elif arr.ndim > 3:
        # e.g. (1, C, H, W)
        arr = arr[0].transpose(1, 2, 0)

    return Image.fromarray(arr)
