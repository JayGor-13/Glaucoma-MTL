# Glaucoma Hybrid Fusion Network

Multi-modal glaucoma screening pipeline that fuses clinical geometry descriptors (vCDR, cup/disc area ratio) with latent deep vision embeddings for high-sensitivity classification.

## Structure

- `src/config.py`: project constants and paths
- `src/dataset.py`: unified label loading + dual-stream dataset
- `src/modules/module_A_geometry.py`: morphology metrics from masks
- `src/modules/module_B_latent.py`: headless timm extractor
- `src/modules/module_C_fusion.py`: MLP late-fusion classifier
- `src/custom_loss.py`: weighted BCE loss for FN reduction
- `src/metrics.py`: sensitivity / specificity / F2 / AUC
- `pipelines/01_extract_features.py`: feature caching pass
- `pipelines/02_train_evaluate.py`: training + evaluation from cache

## Setup

```bash
pip install -r requirements.txt
```

Place Kaggle datasets under `data/raw/`, then run:

```bash
python pipelines/01_extract_features.py
python pipelines/02_train_evaluate.py
```
