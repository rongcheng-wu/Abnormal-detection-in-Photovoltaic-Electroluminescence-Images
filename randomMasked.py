"""Dataset utilities for masked electroluminescence image reconstruction.

This implementation brings the richer augmentation pipeline from the original
research code back into the repository. Images are padded to a target size,
optionally rotated, brightness-adjusted, and patched with several masking
strategies. A Canny edge map is also returned for downstream visualization or
auxiliary losses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


SUPPORTED_EXTS: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class MVTEC_AD_dataloader_randomMasked(Dataset):
    """Dataset that applies random masking and augmentation to EL images.

    Args:
        root_dir: Directory containing image files.
        patch_size: Side length of square patches to alter during masking.
        coverage: Desired ratio of patches to alter. If negative, a random
            coverage value in ``[0, 1]`` is sampled for each image.
        mask: If ``True``, apply masking strategies; otherwise, return the
            clean image for both ``input_img`` and ``masked_image``.
        transform: Optional torchvision transform to apply after padding.
        rot: If ``True``, randomly rotate the image by multiples of 90°.
        brightness_range: Range from which to sample brightness factors.
        brightness_prob: Probability of applying brightness adjustment.
        target_size: Desired square side length after padding. Set to ``None``
            to skip padding.
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 64,
        coverage: float = 0.75,
        mask: bool = True,
        transform: transforms.Compose | None = None,
        rot: bool = False,
        brightness_range: Sequence[float] = (0.2, 1.3),
        brightness_prob: float = 0.8,
        target_size: int | None = 1024,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.coverage = coverage
        self.mask = mask
        self.rot = rot
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.target_size = target_size

        self.image_paths = self._gather_images()
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. Supported extensions: {', '.join(SUPPORTED_EXTS)}"
            )

        pad = self._build_padding_transform()
        to_tensor = transforms.ToTensor()
        extra = transform if transform is not None else transforms.Lambda(lambda x: x)
        self.base_transform = transforms.Compose([pad, to_tensor, extra])

    def _gather_images(self) -> List[Path]:
        paths: List[Path] = []
        for ext in SUPPORTED_EXTS:
            paths.extend(self.root_dir.glob(ext))
        return sorted(paths)

    def _build_padding_transform(self) -> Callable[[Image.Image], Image.Image]:
        if self.target_size is None:
            return transforms.Lambda(lambda x: x)

        def _pad(image: Image.Image) -> Image.Image:
            width, height = image.size
            pad_w = max(self.target_size - width, 0)
            pad_h = max(self.target_size - height, 0)
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            return TF.pad(image, [left, top, right, bottom])

        return transforms.Lambda(_pad)

    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        if not self.rot:
            return image
        transform_rot = transforms.RandomChoice(
            [
                transforms.RandomRotation(degrees=(0, 0)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.RandomRotation(degrees=(270, 270)),
            ]
        )
        return transform_rot(image)

    def _apply_brightness(self, image: Image.Image) -> Image.Image:
        if np.random.rand() >= self.brightness_prob:
            return image
        factor = np.random.uniform(*self.brightness_range)
        return TF.adjust_brightness(image, factor)

    def _compute_edges(self, image: Image.Image) -> torch.Tensor:
        gray_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, 1, 10)
        _, binary_edges = cv2.threshold(edges, 127, 1, cv2.THRESH_BINARY)
        edge_tensor = self.base_transform(Image.fromarray(binary_edges))
        return edge_tensor

    def _mask_image(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        num_horizontal_cells = max(width // self.patch_size, 1)
        num_vertical_cells = max(height // self.patch_size, 1)
        num_cells = num_horizontal_cells * num_vertical_cells

        if self.coverage > 0:
            num_patches = int(num_cells * self.coverage)
        else:
            num_patches = int(num_cells * np.random.rand())

        image_array = np.array(image)
        cell_indices = np.random.choice(num_cells, num_patches, replace=False)

        for cell_idx in cell_indices:
            cell_row = cell_idx // num_horizontal_cells
            cell_col = cell_idx % num_horizontal_cells
            x = cell_col * self.patch_size
            y = cell_row * self.patch_size

            fill_type = np.random.choice(
                ["zeros", "random_matrix", "color_block", "replace"],
                p=[0.25, 0.25, 0.25, 0.25],
            )

            if fill_type == "zeros":
                image_array[y : y + self.patch_size, x : x + self.patch_size, :] = 0
            elif fill_type == "random_matrix":
                image_array[y : y + self.patch_size, x : x + self.patch_size, :] = np.random.randint(
                    0, 256, (self.patch_size, self.patch_size, 3), dtype=np.uint8
                )
            elif fill_type == "color_block":
                random_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
                image_array[y : y + self.patch_size, x : x + self.patch_size, :] = np.tile(
                    random_color, (self.patch_size, self.patch_size, 1)
                )
            elif fill_type == "replace":
                source_idx = np.random.choice(cell_indices)
                source_row = source_idx // num_horizontal_cells
                source_col = source_idx % num_horizontal_cells
                source_x = source_col * self.patch_size
                source_y = source_row * self.patch_size
                image_array[
                    y : y + self.patch_size, x : x + self.patch_size, :
                ] = image_array[
                    source_y : source_y + self.patch_size, source_x : source_x + self.patch_size, :
                ]

        return self.base_transform(Image.fromarray(image_array))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path = self.image_paths[idx]
        input_img = Image.open(img_path).convert("RGB")
        input_img = self._apply_rotation(input_img)
        input_img = self._apply_brightness(input_img)

        binary_edges = self._compute_edges(input_img)
        masked_tensor = self._mask_image(input_img) if self.mask else self.base_transform(input_img)
        input_tensor = self.base_transform(input_img)

        return {
            "input_img": input_tensor,
            "masked_image": masked_tensor,
            "binary_edges": binary_edges,
            "img_name": img_path.name,
        }
