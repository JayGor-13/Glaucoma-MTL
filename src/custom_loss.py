"""Custom losses."""

import torch
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)
