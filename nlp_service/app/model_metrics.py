"""One metrics contract for every classifier (TF-IDF, LSTM, BiLSTM and later BERT).

The model-comparison dashboard needs to put TF-IDF, LSTM, BiLSTM and BERT side by side
using the SAME fields. Every classifier already writes its per-split scores with the
shared ``app.metrics.classification_metrics``; this module turns that convention into an
explicit, testable contract:

* ``validate_metrics(metrics)``  -> list of violations (empty = conforms). It checks the
  structure and also internal consistency (confusion matrix totals, accuracy == trace / n,
  per-class supports == row sums, sorted label order shared by all models).
* ``rows_from_metrics(metrics)`` -> flat rows, one per (model, target, split), with
  identical keys for every model. This is what a dashboard table / chart consumes.
* ``collect()``                  -> reads ``artifacts/<model>/metrics.json`` for every model
  that has one, validates it, and returns the rows plus a per-model problem report. A
  broken file is reported, never silently dropped and never turned into numbers.

Only real ``metrics.json`` files are read. Nothing here computes, estimates or fills in
a score. This module has no PyTorch dependency.

Required top level:  model, trained_at, seed, data.rows{train,val,test}, data.pdf_heldout_evaluated, targets
Per trained target:  labels (sorted), n_train, splits{val|test|pdf_heldout}
Per split:           n, accuracy, precision/recall/f1 (macro + weighted), per_class,
                     confusion_matrix{labels, matrix}, majority_class_baseline, unseen_label_rows
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import settings
from .registry import model_status

MODELS = ("tfidf", "lstm", "bilstm", "bert")
TARGET_ORDER = ("subject", "topic", "difficulty")
SPLITS = ("val", "test", "pdf_heldout")
SCORE_FIELDS = (
    "accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
)
PER_CLASS_FIELDS = ("precision", "recall", "f1", "support")
ROW_FIELDS = (
    "model", "target", "split", "n", *SCORE_FIELDS, "majority_class_accuracy",
    "n_classes", "n_train", "trainable_parameters", "trained_at", "seed", "checkpoint_available",
)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_split(path: str, m: Any, labels: list[str]) -> list[str]:
    if not isinstance(m, dict):
        return [f"{path}: must be an object"]
    bad: list[str] = []
    if not _is_int(m.get("n")) or m["n"] < 1:
        bad.append(f"{path}.n: must be an integer >= 1")
    for f in SCORE_FIELDS:
        v = m.get(f)
        if not _is_num(v) or not 0.0 <= v <= 1.0:
            bad.append(f"{path}.{f}: must be a number in [0, 1]")
    if not _is_int(m.get("unseen_label_rows")) or m["unseen_label_rows"] < 0:
        bad.append(f"{path}.unseen_label_rows: must be an integer >= 0")
    mb = m.get("majority_class_baseline")
    if not (isinstance(mb, dict) and isinstance(mb.get("class"), str) and _is_num(mb.get("accuracy"))
            and 0.0 <= mb["accuracy"] <= 1.0):
        bad.append(f"{path}.majority_class_baseline: needs 'class' and 'accuracy' in [0, 1]")

    cm = m.get("confusion_matrix")
    if not (isinstance(cm, dict) and isinstance(cm.get("labels"), list) and isinstance(cm.get("matrix"), list)):
        return bad + [f"{path}.confusion_matrix: needs 'labels' and 'matrix'"]
    cm_labels, matrix = cm["labels"], cm["matrix"]
    k = len(cm_labels)
    if cm_labels[: len(labels)] != labels:
        bad.append(f"{path}.confusion_matrix.labels: must start with the target labels in sorted order")
    if len(matrix) != k or any(not isinstance(r, list) or len(r) != k for r in matrix):
        return bad + [f"{path}.confusion_matrix.matrix: must be {k}x{k}"]
    if any(not _is_int(v) or v < 0 for r in matrix for v in r):
        return bad + [f"{path}.confusion_matrix.matrix: entries must be non-negative integers"]
    total = sum(sum(r) for r in matrix)
    if _is_int(m.get("n")) and total != m["n"]:
        bad.append(f"{path}.confusion_matrix: entries sum to {total}, expected n={m['n']}")
    if total and _is_num(m.get("accuracy")) and abs(sum(matrix[i][i] for i in range(k)) / total - m["accuracy"]) > 1e-9:
        bad.append(f"{path}.accuracy: does not equal the confusion matrix trace / n")
    pc = m.get("per_class")
    if not isinstance(pc, dict) or list(pc) != cm_labels:
        bad.append(f"{path}.per_class: keys must equal confusion_matrix.labels, in the same order")
    else:
        for i, lab in enumerate(cm_labels):
            entry = pc[lab]
            if not isinstance(entry, dict) or any(f not in entry for f in PER_CLASS_FIELDS):
                bad.append(f"{path}.per_class.{lab}: needs {PER_CLASS_FIELDS}")
            elif entry["support"] != sum(matrix[i]):
                bad.append(f"{path}.per_class.{lab}.support: does not equal its confusion-matrix row sum")
    return bad


def validate_metrics(metrics: Any) -> list[str]:
    """Return every way ``metrics`` breaks the contract (empty list = it conforms)."""
    if not isinstance(metrics, dict):
        return ["metrics.json must contain a JSON object"]
    bad: list[str] = []
    if not isinstance(metrics.get("model"), str) or not metrics["model"]:
        bad.append("model: must be a non-empty string")
    if not isinstance(metrics.get("trained_at"), str):
        bad.append("trained_at: must be a string")
    if not _is_int(metrics.get("seed")):
        bad.append("seed: must be an integer")
    data = metrics.get("data")
    rows = data.get("rows") if isinstance(data, dict) else None
    if not (isinstance(rows, dict) and all(_is_int(rows.get(s)) for s in ("train", "val", "test"))):
        bad.append("data.rows: must give integer row counts for train, val and test")
    if not (isinstance(data, dict) and isinstance(data.get("pdf_heldout_evaluated"), bool)):
        bad.append("data.pdf_heldout_evaluated: must be a boolean")
    targets = metrics.get("targets")
    if not isinstance(targets, dict) or not targets:
        return bad + ["targets: must be a non-empty object"]
    for t, e in targets.items():
        p = f"targets.{t}"
        if not isinstance(e, dict) or e.get("status") not in ("trained", "skipped"):
            bad.append(f"{p}.status: must be 'trained' or 'skipped'")
            continue
        if e["status"] == "skipped":
            if not isinstance(e.get("reason"), str):
                bad.append(f"{p}.reason: a skipped target must say why")
            continue
        labels = e.get("labels")
        if not (isinstance(labels, list) and len(labels) >= 2 and all(isinstance(x, str) for x in labels)
                and labels == sorted(labels)):
            bad.append(f"{p}.labels: must be a sorted list of at least 2 strings")
            continue
        if not _is_int(e.get("n_train")) or e["n_train"] < 1:
            bad.append(f"{p}.n_train: must be an integer >= 1")
        splits = e.get("splits")
        if not isinstance(splits, dict) or not splits or set(splits) - set(SPLITS):
            bad.append(f"{p}.splits: must be a non-empty object with keys from {SPLITS}")
            continue
        for name, m in splits.items():
            bad.extend(_validate_split(f"{p}.splits.{name}", m, labels))
    return bad


def rows_from_metrics(metrics: dict[str, Any], checkpoint_available: bool | None = None) -> list[dict[str, Any]]:
    """Flat rows, one per (target, split), identical keys for every model.

    Assumes ``validate_metrics(metrics)`` returned no violations.
    """
    rows = []
    targets = sorted(metrics["targets"], key=lambda t: TARGET_ORDER.index(t) if t in TARGET_ORDER else 99)
    for target in targets:
        e = metrics["targets"][target]
        if e["status"] != "trained":
            continue
        for split in SPLITS:
            m = e["splits"].get(split)
            if m is None:
                continue
            rows.append({
                "model": metrics["model"],
                "target": target,
                "split": split,
                "n": m["n"],
                **{f: m[f] for f in SCORE_FIELDS},
                "majority_class_accuracy": m["majority_class_baseline"]["accuracy"],
                "n_classes": len(e["labels"]),
                "n_train": e["n_train"],
                "trainable_parameters": (e.get("model") or {}).get("trainable_parameters"),
                "trained_at": metrics["trained_at"],
                "seed": metrics["seed"],
                "checkpoint_available": checkpoint_available,
            })
    return rows


def collect(artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Read every available ``artifacts/<model>/metrics.json`` and return comparable rows.

    ``problems`` maps a model to its contract violations (its rows are then withheld);
    ``models_missing`` lists models with no metrics file at all (not trained yet).
    """
    base = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    trained = {m["name"]: m["trained"] for m in model_status()}
    rows: list[dict[str, Any]] = []
    found: list[str] = []
    missing: list[str] = []
    problems: dict[str, list[str]] = {}
    for model in MODELS:
        f = base / model / "metrics.json"
        if not f.is_file():
            missing.append(model)
            continue
        try:
            metrics = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems[model] = [f"could not read metrics.json: {exc}"]
            continue
        bad = validate_metrics(metrics)
        if not bad and metrics["model"] != model:
            bad.append(f"model: file lives in '{model}/' but says '{metrics['model']}'")
        if bad:
            problems[model] = bad
            continue
        found.append(model)
        rows.extend(rows_from_metrics(metrics, checkpoint_available=trained.get(model)))
    return {"models_found": found, "models_missing": missing, "problems": problems, "rows": rows}
