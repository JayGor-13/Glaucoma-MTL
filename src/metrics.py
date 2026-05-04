"""Metrics for glaucoma classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity, "f2": f2, "auc": auc}
