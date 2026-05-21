import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def classification_metrics(y_true, logits) -> dict[str, float]:
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    aucs = []
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    for task_idx in range(y_true.shape[1]):
        task_y = y_true[:, task_idx]
        task_probs = probs[:, task_idx]
        mask = np.isfinite(task_y)
        if mask.sum() == 0:
            continue
        task_y = task_y[mask]
        task_probs = task_probs[mask]
        if len(np.unique(task_y)) < 2:
            continue
        aucs.append(float(roc_auc_score(task_y, task_probs)))
    if not aucs:
        return {"roc_auc": float("nan"), "roc_auc_tasks": 0}
    return {"roc_auc": float(np.mean(aucs)), "roc_auc_tasks": len(aucs)}


def regression_metrics(y_true, preds) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    mse = mean_squared_error(y_true, preds)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, preds)),
    }
