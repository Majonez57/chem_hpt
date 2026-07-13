"""Precompute frozen ResNet18 image features for the PushT dataset.

WHY precompute? Running a ResNet on every image during training is expensive.
Because we keep the ResNet frozen (it is just a generic vision encoder), we run
it ONCE over all images, save the feature vectors, and afterwards training only
touches our tiny HPT network. This makes training fast and the model small.

PushT images are 96x96. ResNet18 downsamples by 32x, so its final feature map
is 3x3 with 512 channels -> 9 tokens of 512 numbers each. These 9 tokens per
frame are exactly what the VisionStem will later read from.

    INPUT : data/pusht_raw_dl/pusht_cchi_v7_replay.zarr   (raw PushT, downloaded)
    OUTPUT: data/pusht_resnet18_tokens.npy                 (float16, (N, 9, 512))

To adapt to another dataset: only the loading block changes. Get YOUR images
into a (N, H, W, 3) uint8/float array (values in 0..255) and the rest stays.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import zarr

# ? change these paths if your data lives elsewhere
ZARR_PATH = "data/pusht_raw_dl/pusht_cchi_v7_replay.zarr"
OUT_PATH = "data/pusht_resnet18_tokens.npy"

# ImageNet statistics. The published ResNet18 weights were trained with this
# normalization, so we match it to get sensible features.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_backbone(device: str) -> nn.Module:
    """ResNet18 with the classifier head removed -> a pure feature-map maker."""
    import torchvision

    resnet = torchvision.models.resnet18(weights="DEFAULT")
    # children()[:-2] drops the global AveragePool and the final Linear
    # classifier, leaving the conv stack that outputs a spatial feature map.
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def precompute(device: str, chunk: int = 512) -> None:
    backbone = build_resnet18_backbone(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    root = zarr.open(ZARR_PATH, mode="r")
    imgs = root["data/img"]            # (N, 96, 96, 3) float32, values in 0..255
    n = imgs.shape[0]
    print(f"precomputing ResNet18 features for {n} frames on {device} ...")

    feats = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch = torch.from_numpy(np.asarray(imgs[start:end])).float().to(device)  # (c,96,96,3)
        batch = batch / 255.0                     # -> 0..1
        batch = batch.permute(0, 3, 1, 2)         # -> (c, 3, 96, 96)
        batch = (batch - mean) / std              # ImageNet normalize

        fmap = backbone(batch)                    # (c, 512, h, w)
        c, ch, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)  # (c, h*w, 512)
        feats.append(tokens.cpu().numpy().astype(np.float16))
        print(f"  {end}/{n}  feature map {h}x{w} -> {h*w} tokens/frame")

    features = np.concatenate(feats, axis=0)       # (N, h*w, 512)
    print(f"final feature array: shape={features.shape} dtype={features.dtype}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, features)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    precompute(device)
