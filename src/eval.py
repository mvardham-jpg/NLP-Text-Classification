# src/eval.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    """Returns accuracy + macro-F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def top_confusions(y_true, y_pred, k=10):
    """
    Returns top k confusion pairs as (count, true_class_index, pred_class_index)
    excluding the diagonal.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    pairs = []
    for i in range(cm_off.shape[0]):
        for j in range(cm_off.shape[1]):
            if cm_off[i, j] > 0:
                pairs.append((cm_off[i, j], i, j))

    pairs.sort(reverse=True)
    return pairs[:k]
