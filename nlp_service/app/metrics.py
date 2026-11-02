"""Evaluation metrics shared by TF-IDF now and LSTM / BiLSTM / BERT later, so every
model is scored by exactly the same code.

Every number is computed from real predictions with scikit-learn. Nothing is
hard-coded or estimated.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    *,
    train_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores, confusion matrix.

    ``labels`` are the classes the model can predict. If ``y_true`` contains a
    class the model never saw (possible on the PDF held-out set) it is appended
    to the label list, counted as an error, and reported in ``unseen_label_rows``.

    ``train_labels`` (optional) adds a majority-class baseline for context: the
    accuracy you would get by always predicting the most common training class.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot compute metrics on zero rows")

    model_labels = list(labels)
    extra = sorted(set(y_true) - set(model_labels))
    all_labels = model_labels + extra

    out: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=avg, zero_division=0
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )
    out["per_class"] = {
        lab: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, lab in enumerate(all_labels)
    }
    out["confusion_matrix"] = {
        "labels": all_labels,  # rows = true class, columns = predicted class
        "matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
    }
    out["unseen_label_rows"] = int(sum(1 for t in y_true if t in extra))

    if train_labels is not None and len(train_labels):
        majority = Counter(train_labels).most_common(1)[0][0]
        out["majority_class_baseline"] = {
            "class": majority,
            "accuracy": float(sum(1 for t in y_true if t == majority) / len(y_true)),
        }
    return out
