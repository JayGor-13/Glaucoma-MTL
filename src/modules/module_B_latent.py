"""Latent feature extractor using timm backbones."""

from __future__ import annotations

import timm
import torch


def get_feature_extractor(model_name: str = "efficientnet_b3") -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    return model


@torch.no_grad()
def extract_latent_features(model: torch.nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    return model(images.to(device))
