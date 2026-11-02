"""Reports what actually exists on disk.

Nothing here is hard-coded as "trained". A model counts as trained only when its
training script has written the marker file listed below. Training scripts must
write that marker LAST, after the model is saved and evaluated, so a half-finished
run never shows up as trained.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str                       # what the component is, for display
    marker: str                     # file written last by its training script
    weights_suffix: str | None = None   # saved model files, e.g. ".joblib" / ".model"
    weights_label: str = "targets_trained"  # key under which their stems are reported


MODELS: list[ModelSpec] = [
    ModelSpec("tfidf", "classical baseline (TF-IDF + Logistic Regression)", "metrics.json", ".joblib"),
    ModelSpec(
        "word2vec",
        "word embeddings: CBOW + Skip-Gram (visualised with PCA / t-SNE)",
        "metadata.json",
        ".model",
        "variants_trained",
    ),
    ModelSpec("lstm", "deep sequence classifier (LSTM)", "metrics.json", ".pt"),
    ModelSpec("bilstm", "deep sequence classifier (BiLSTM)", "metrics.json", ".pt"),
    # Placeholder until Phase 6: BERT will save <target>.pt checkpoints (same convention as the
    # LSTM / BiLSTM). Requiring weights now means a stray metrics.json can never read as "trained".
    ModelSpec("bert", "fine-tuned transformer classifier (BERT)", "metrics.json", ".pt"),
]

# Python packages the later phases need. Probed, never imported, so /health
# stays fast and works even when none of them are installed yet.
DEPENDENCIES = [
    "numpy", "pandas", "scikit-learn", "nltk", "gensim",
    "torch", "transformers", "matplotlib",
]


def model_status() -> list[dict[str, Any]]:
    """trained = marker file exists AND (if the model has saved weight files) at
    least one of them exists. A marker alone, e.g. after a fresh git clone where
    weights are not committed, does not count as trained."""
    out = []
    for spec in MODELS:
        model_dir: Path = Path(settings.artifacts_dir) / spec.name
        weights = (
            sorted(p.stem for p in model_dir.glob(f"*{spec.weights_suffix}"))
            if spec.weights_suffix
            else None
        )
        trained = (model_dir / spec.marker).is_file() and (
            weights is None or len(weights) > 0
        )
        entry: dict[str, Any] = {
            "name": spec.name,
            "role": spec.role,
            "trained": trained,
            "marker": f"{spec.name}/{spec.marker}",
        }
        if weights is not None:
            entry[spec.weights_label] = weights if trained else []
        out.append(entry)
    return out


def dependency_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pkg in DEPENDENCIES:
        # distribution name != import name for scikit-learn
        import_name = "sklearn" if pkg == "scikit-learn" else pkg
        installed = importlib.util.find_spec(import_name) is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
        out[pkg] = {"installed": installed, "version": version}
    return out


def dataset_status() -> dict[str, dict[str, Any]]:
    """Which data files exist.

    public_train  : labelled public question dataset -> train / validation / test
    pdf_heldout   : questions/chunks from GovPrep-AI PDFs -> FINAL evaluation only,
                    never used for training or tuning.
    """
    data = settings.data_dir
    files = {
        "public_train": data / "processed" / "public_train.csv",
        "public_val": data / "processed" / "public_val.csv",
        "public_test": data / "processed" / "public_test.csv",
        "pdf_heldout": data / "processed" / "pdf_heldout.csv",
    }
    return {
        key: {"present": path.is_file(), "path": str(path.relative_to(data.parent)) if path.is_relative_to(data.parent) else str(path)}
        for key, path in files.items()
    }
