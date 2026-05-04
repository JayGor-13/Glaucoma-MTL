"""Train and evaluate fusion classifier from cached features."""

from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEVICE, PROCESSED_DIR
from src.custom_loss import WeightedBCELoss
from src.metrics import compute_metrics
from src.modules.module_C_fusion import AsymmetricFusionClassifier


def main() -> None:
    x = np.load(Path(PROCESSED_DIR, "cached_master_dataset.npy"))
    y = np.load(Path(PROCESSED_DIR, "labels.npy"))

    if len(x) == 0:
        print("No cached features found. Run pipeline 01 first.")
        return

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = AsymmetricFusionClassifier(input_dim=x.shape[1]).to(DEVICE)
    crit = WeightedBCELoss(pos_weight=4.0)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(15):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_t.to(DEVICE))).cpu().numpy()
    result = compute_metrics(y_true=y.astype(int), y_prob=prob)
    print(result)


if __name__ == "__main__":
    main()
