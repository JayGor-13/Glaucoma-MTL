"""Unified dataset and CSV-to-label mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

VALID_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    label: int


def _normalize_stem(value: str) -> str:
    return Path(str(value)).stem.lower().strip()


def load_label_dict(csv_paths: list[Path], filename_col: str, label_col: str) -> dict[str, int]:
    label_map: dict[str, int] = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = _normalize_stem(row[filename_col])
            label_map[key] = int(row[label_col])
    return label_map


def find_match(root: Path, stem: str) -> Path | None:
    for ext in VALID_EXTS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_samples(image_dir: Path, mask_dir: Path, label_map: dict[str, int]) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_EXTS:
            continue
        stem = image_path.stem.lower()
        if stem not in label_map:
            continue
        mask_path = find_match(mask_dir, stem)
        if mask_path is None:
            continue
        samples.append(Sample(image_path=image_path, mask_path=mask_path, label=label_map[stem]))
    return samples


class DualStreamGlaucomaDataset(Dataset):
    def __init__(self, samples: list[Sample], image_transform: Callable | None = None, mask_transform: Callable | None = None):
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        mask = Image.open(s.mask_path).convert("L")

        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(float(s.label), dtype=torch.float32),
            "image_path": str(s.image_path),
        }
