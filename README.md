# PV-UNet for Solar Cell EL Image Reconstruction

This repository contains training code for a reconstruction model (`Rec_model`) applied to solar cell electroluminescence (EL) images. The code accompanies the paper **“Unsupervised and Few-Shot Segmentation in Photovoltaic Electroluminescence Images for Defects Detection via a Novel Enhanced Iterative Autoencoder with Simple Implementation.”** The implementation is still being organized and refined, so expect incremental updates. The model is trained on masked input patches and supervised by the original clean images.

> 这是论文《Unsupervised and Few-Shot Segmentation in Photovoltaic Electroluminescence Images for Defects Detection via a Novel Enhanced Iterative Autoencoder with Simple Implementation》的官方代码仓库，当前仍在陆续整理和完善中。

## Features

- Configurable training via `config.yaml`
- L1 + gradient-based reconstruction loss
- Mixed-precision training (AMP) support
- Multi-GPU support via `nn.DataParallel`
- Optional visualization script to export reconstructed images

## Project structure

```
.
├── config.yaml
├── train.py
├── losses.py
├── visualization.py
├── model.py                # your Rec_model implementation
├── randomMasked.py         # your MVTEC_AD_dataloader_randomMasked implementation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Make sure you have a compatible version of PyTorch and CUDA installed.

## Configuration

Edit `config.yaml` to set:

* Dataset paths (`data.train_root`, `data.test_root`)
* Patch size, coverage, masking settings
* Batch size, number of workers
* Learning rate, number of epochs, AMP usage
* Loss weights and checkpoint options

## Training

```bash
python train.py --config config.yaml --mode train
```

This will train the model and save checkpoints to `model.checkpoint_dir`.

## Visualization

After training (or if you already have a checkpoint), you can generate reconstructed images:

```bash
python train.py --config config.yaml --mode visualize
```

Images will be saved under the directory specified by `visualization.output_dir` in the config.

## Notes

* `model.py` must define a class `Rec_model`.
* `randomMasked.py` must define `MVTEC_AD_dataloader_randomMasked`.
* The dataset class should return a dict with keys: `input_img`, `masked_image`, `img_name`.
