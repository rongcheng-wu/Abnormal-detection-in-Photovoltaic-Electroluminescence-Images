# train.py

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from randomMasked import MVTEC_AD_dataloader_randomMasked
from model import Rec_model
from losses import ReconstructionLoss
from visualization import visualize_predictions


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]

    train_dataset = MVTEC_AD_dataloader_randomMasked(
        data_cfg["train_root"],
        patch_size=data_cfg["patch_size"],
        coverage=data_cfg["coverage"],
        mask=data_cfg["mask_train"],
        rot=data_cfg["rotate"],
    )

    test_dataset = MVTEC_AD_dataloader_randomMasked(
        data_cfg["test_root"],
        patch_size=data_cfg["patch_size"],
        coverage=data_cfg["coverage"],
        mask=data_cfg["mask_test"],
        rot=data_cfg["rotate"],
    )

    return train_dataset, test_dataset


def build_dataloaders(cfg: dict, train_dataset):
    data_cfg = cfg["data"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )

    return train_loader


def build_model_and_optimizer(cfg: dict, device: torch.device):
    model = Rec_model()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    return model, optimizer


def maybe_load_checkpoint(cfg: dict, model: nn.Module, device: torch.device) -> Path:
    model_cfg = cfg["model"]
    ckpt_dir = Path(model_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / model_cfg["checkpoint_name"]

    if model_cfg.get("resume", False) and ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint loaded. Training from scratch.")

    return ckpt_path


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    print("Configuration:")
    print(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and dataloaders
    train_dataset, _ = build_datasets(cfg)
    train_loader = build_dataloaders(cfg, train_dataset)

    # Model and optimizer
    model, optimizer = build_model_and_optimizer(cfg, device)

    # Loss function
    loss_cfg = cfg["loss"]
    criterion = ReconstructionLoss(
        l1_weight=loss_cfg["l1_weight"],
        gradient_weight=loss_cfg["gradient_weight"],
    )

    # Checkpoint
    ckpt_path = maybe_load_checkpoint(cfg, model, device)

    # AMP scaler
    train_cfg = cfg["training"]
    use_amp = bool(train_cfg.get("use_amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    num_epochs = train_cfg["num_epochs"]
    log_interval = train_cfg.get("log_interval", 50)
    save_every_epoch = train_cfg.get("save_every_epoch", 1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            inputs = batch["masked_image"].to(device)
            labels = batch["input_img"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                progress_bar.set_postfix({"loss": f"{avg_loss:.6f}"})
                running_loss = 0.0

        if ((epoch + 1) % save_every_epoch) == 0:
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training completed.")


def run_visualization(config_path: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test dataset only
    _, test_dataset = build_datasets(cfg)

    # Build model and load checkpoint
    model = Rec_model()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for visualization.")
        model = nn.DataParallel(model)

    model = model.to(device)

    model_cfg = cfg["model"]
    ckpt_path = Path(model_cfg["checkpoint_dir"]) / model_cfg["checkpoint_name"]
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path} for visualization.")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Checkpoint {ckpt_path} not found. Visualization will run with randomly initialized weights.")

    vis_cfg = cfg.get("visualization", {})
    output_dir = vis_cfg.get("output_dir", "visualization")
    batch_size = vis_cfg.get("batch_size", 1)
    num_workers = cfg["data"]["num_workers"]

    visualize_predictions(
        net=model,
        dataset=test_dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train and visualize PV-UNet on solar cell EL images.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "visualize"],
        default="train",
        help="Run mode: 'train' or 'visualize'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args.config)
    elif args.mode == "visualize":
        run_visualization(args.config)
