"""Global project configuration."""

from pathlib import Path
import torch

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = 2

RAW_DATA_PATH = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
PROCESSED_FEATURE_PATH = PROCESSED_DIR / "combined_features.npy"
PROCESSED_LABEL_PATH = PROCESSED_DIR / "labels.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
