"""
Dataset utilities for masked electroluminescence image reconstruction.

This module provides a lightweight dataset compatible with the training
and visualization entrypoints. Images are loaded from a directory and
optionally masked by zeroing a random square patch.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTEC_AD_dataloader_randomMasked(Dataset):
    """Simple image dataset that applies a random square mask.

    Args:
        root_dir: Directory containing image files.
        patch_size: Side length of the square mask to remove.
        coverage: Unused compatibility flag kept for parity with the
            original interface. Included in the signature for config
            compatibility.
        mask: If ``True``, apply masking; otherwise, return the clean
            image for both outputs.
        rot: If ``True``, apply a random rotation by 0/90/180/270 degrees.
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 32,
        coverage: int = -1,
        mask: bool = True,
        rot: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.coverage = coverage
        self.mask = mask
        self.rot = rot

        self.image_paths = self._gather_images()
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. Supported extensions: jpg, jpeg, png, bmp."
            )

        self.to_tensor = transforms.Compose(
            [transforms.ConvertImageDtype(torch.float32)]
        )

    def _gather_images(self) -> List[Path]:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths: List[Path] = []
        for ext in exts:
            paths.extend(self.root_dir.glob(ext))
        return sorted(paths)

    def _apply_rotation(self, image: torch.Tensor) -> torch.Tensor:
        if not self.rot:
            return image
        k = np.random.randint(0, 4)
        return torch.rot90(image, k, dims=(1, 2))

    def _apply_mask(self, image: torch.Tensor) -> torch.Tensor:
        if not self.mask:
            return image

        c, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            return image

        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        masked = image.clone()
        masked[:, top : top + self.patch_size, left : left + self.patch_size] = 0.0
        return masked

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor: torch.Tensor = transforms.functional.pil_to_tensor(pil_img)
        img_tensor = self.to_tensor(img_tensor)

        img_tensor = self._apply_rotation(img_tensor)
        masked = self._apply_mask(img_tensor)

        return {
            "input_img": img_tensor,
            "masked_image": masked,
            "img_name": img_path.name,
        }
