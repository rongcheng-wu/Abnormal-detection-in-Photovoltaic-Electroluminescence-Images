# visualization.py

import os
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def visualize_predictions(
    net: torch.nn.Module,
    dataset,
    output_dir: str = "./visualization",
    batch_size: int = 1,
    num_workers: int = 1,
    device: Union[str, torch.device] = "cuda",
) -> None:
    """
    Run inference on the whole dataset and save visualization images.

    Assumes each dataset sample is a dict with:
        - 'input_img': ground truth image tensor (C, H, W)
        - 'masked_image': masked input tensor  (C, H, W)
        - 'img_name': original image filename (string)

    The model is expected to support calls like:
        net(masked_image)          # training-style forward
        net(masked_image, mode_id) # visualization modes 1..5 (if implemented)
    """
    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    net = net.to(device)
    net.eval()

    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Visualizing", unit="batch"):
            labels = batch["input_img"].to(device)
            masked = batch["masked_image"].to(device)
            img_names = batch["img_name"]

            # You can adjust this logic depending on how Rec_model handles the second argument
            count = 0

            init_output1 = net(masked, 1)
            for _ in range(count):
                init_output1 = net(init_output1, 0)

            init_output2 = net(masked, 2)
            for _ in range(count):
                init_output2 = net(init_output2, 0)

            init_output3 = net(masked, 3)
            for _ in range(count):
                init_output3 = net(init_output3, 1)

            init_output4 = net(masked, 4)
            for _ in range(count):
                init_output4 = net(init_output4, 3)

            init_output5 = net(masked, 5)
            for _ in range(count):
                init_output5 = net(init_output5, 5)

            for i in range(init_output1.shape[0]):
                file_name = img_names[i]
                file_stem = os.path.splitext(file_name)[0]

                out1 = to_pil(init_output1[i].cpu())
                out2 = to_pil(init_output2[i].cpu())
                out3 = to_pil(init_output3[i].cpu())
                out4 = to_pil(init_output4[i].cpu())
                out5 = to_pil(init_output5[i].cpu())

                out1 = np.array(out1)
                out2 = np.array(out2)
                out3 = np.array(out3)
                out4 = np.array(out4)
                out5 = np.array(out5)

                base_path = os.path.join(output_dir, file_stem)

                Image.fromarray(out1.astype(np.uint8)).save(base_path + "_1.jpg")
                Image.fromarray(out2.astype(np.uint8)).save(base_path + "_2.jpg")
                Image.fromarray(out3.astype(np.uint8)).save(base_path + "_3.jpg")
                Image.fromarray(out4.astype(np.uint8)).save(base_path + "_4.jpg")
                Image.fromarray(out5.astype(np.uint8)).save(base_path + "_5.jpg")
