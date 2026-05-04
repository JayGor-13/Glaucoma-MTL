"""Extract and cache geometry + latent features."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, IMAGE_SIZE, PROCESSED_DIR
from src.dataset import DualStreamGlaucomaDataset, Sample
from src.modules.module_A_geometry import extract_morphology_metrics
from src.modules.module_B_latent import extract_latent_features, get_feature_extractor


def main() -> None:
    # Placeholder sample list; replace with real CSV-based sample construction.
    samples: list[Sample] = []

    image_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.PILToTensor()])

    ds = DualStreamGlaucomaDataset(samples=samples, image_transform=image_tf, mask_transform=mask_tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = get_feature_extractor().to(DEVICE)

    all_features, all_labels = [], []
    for batch in tqdm(dl, desc="Extracting"):
        latent = extract_latent_features(model, batch["image"], DEVICE).cpu().numpy()
        masks = batch["mask"].squeeze(1).numpy()
        geom = np.stack([extract_morphology_metrics(m) for m in masks], axis=0)
        fused = np.concatenate([latent, geom], axis=1)

        all_features.append(fused)
        all_labels.append(batch["label"].numpy())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feat = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 0), dtype=np.float32)
    lab = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.float32)

    np.save(Path(PROCESSED_DIR, "cached_master_dataset.npy"), feat)
    np.save(Path(PROCESSED_DIR, "labels.npy"), lab)
    print("Features extracted, free up GPU now!")


if __name__ == "__main__":
    with torch.no_grad():
        main()
