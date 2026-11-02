"""Builds contract-conforming metrics.json dictionaries for comparison tests.

TEST-ONLY and synthetic: the predictions are generated to have a chosen accuracy, but every
metric in the result is computed by the real ``app.metrics.classification_metrics``, so the
files satisfy the real contract (``app.model_metrics.validate_metrics``). Nothing here is, or
may be presented as, a result about a real model.
"""
import json
from pathlib import Path

from app.config import settings
from app.metrics import classification_metrics


def synth_predictions(labels, n, accuracy):
    """``n`` examples cycling through ``labels``; exactly ``round(n * (1 - accuracy))`` are wrong."""
    k = len(labels)
    y_true = [labels[i % k] for i in range(n)]
    wrong = round(n * (1 - accuracy))
    y_pred = [labels[(i + 1) % k] if i < wrong else y_true[i] for i in range(n)]
    return y_true, y_pred


def split_spec(labels, n, accuracy):
    return synth_predictions(labels, n, accuracy)


def make_metrics(model, targets, *, seed=42, sha="split-A", manifest_sha="input-A", params=None,
                 with_file_sha=True, with_manifest=True, trained_at="2026-01-01T00:00:00+00:00"):
    """``targets``: {target: "skipped" | {"labels": [...], "train": [...], "splits": {name: (y_true, y_pred)}}}"""
    entries, heldout = {}, False
    for target, spec in targets.items():
        if spec == "skipped":
            entries[target] = {"status": "skipped", "reason": "no labelled rows"}
            continue
        splits = {name: classification_metrics(yt, yp, spec["labels"], train_labels=spec["train"])
                  for name, (yt, yp) in spec["splits"].items()}
        heldout = heldout or "pdf_heldout" in splits
        entries[target] = {
            "status": "trained", "labels": spec["labels"], "n_train": len(spec["train"]), "splits": splits,
            "model": {"trainable_parameters": params},
        }
    data = {"rows": {"train": 100, "val": 20, "test": 20}, "pdf_heldout_evaluated": heldout}
    if with_file_sha:
        data["split_files_sha256"] = {"public_train": sha, "public_val": sha, "public_test": sha}
    if with_manifest:
        data["split_manifest"] = {"public_input": {"sha256": manifest_sha}, "split": {"seed": 42, "sizes": {"train": 100}}}
    return {"model": model, "trained_at": trained_at, "seed": seed, "data": data, "targets": entries}


def simple(model, *, accuracies, n=200, labels=("a", "b", "c"), target="subject", **kw):
    """One trained target whose splits have the given accuracies, e.g. {"val": 0.8, "test": 0.9}."""
    labels = list(labels)
    train = [labels[i % len(labels)] for i in range(90)]
    splits = {name: synth_predictions(labels, n, acc) for name, acc in accuracies.items()}
    return make_metrics(model, {target: {"labels": labels, "train": train, "splits": splits}}, **kw)


def write(metrics, model=None):
    d = Path(settings.artifacts_dir) / (model or metrics["model"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d
