"""Common metrics utilities for binary/multiclass evaluation."""
import numpy as np
import torch
from typing import Dict, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=2, task_name="binary") -> Dict:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_prob is None:
        y_prob = np.eye(num_classes)[y_pred]
    y_prob = _to_numpy(y_prob)

    metrics: Dict[str, Optional[float]] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    if num_classes == 2:
        metrics["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # AUCs
        try:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, prob_pos)
            metrics["pr_auc"] = float(auc(recall_vals, precision_vals))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            metrics["mcc"] = None
    else:
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        try:
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc_macro"] = None

    # confusion matrix and per-class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    per_class = {}
    for cid in range(num_classes):
        key = str(cid)
        if key in report:
            per_class[key] = {
                "precision": float(report[key]["precision"]),
                "recall": float(report[key]["recall"]),
                "f1": float(report[key]["f1-score"]),
                "support": int(report[key]["support"]),
            }
    metrics["per_class"] = per_class
    return metrics
