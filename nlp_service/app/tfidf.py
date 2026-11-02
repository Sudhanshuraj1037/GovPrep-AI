"""TF-IDF baseline: preprocessing -> TF-IDF vectoriser -> Logistic Regression.

One independent classifier is trained per target (subject, topic, difficulty).

Training protocol
-----------------
* fit the vectoriser and classifier on the TRAIN split only;
* choose the regularisation strength C on the VALIDATION split (macro-F1);
* report metrics on validation, on the public TEST split, and, if present, on
  the PDF held-out set. The held-out set is never used for fitting or tuning;
* artifacts are written only after everything succeeded, with ``metrics.json``
  written LAST (it is the marker the /health endpoint looks for).

If a target cannot be trained (no labels, one class only, ...) it is recorded as
skipped with the reason. If nothing can be trained, no artifact is written at all.

Security note: models are stored with joblib (pickle). Only load artifacts that
you trained yourself.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import data as data_mod
from .config import settings
from .metrics import classification_metrics
from .preprocessing import PreprocessConfig, TextPreprocessor

TARGETS = ("subject", "topic", "difficulty")
MODEL_NAME = "tfidf"
EVAL_SPLITS = ("val", "test", "pdf_heldout")


class ModelNotTrainedError(RuntimeError):
    """Raised by inference when no trained TF-IDF model exists."""


class NothingTrainedError(RuntimeError):
    """Raised by training when no target could be trained (nothing is written)."""


@dataclass(frozen=True)
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    max_features: int | None = 50_000
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    class_weight: str | None = "balanced"
    max_iter: int = 1000

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        d["c_grid"] = list(self.c_grid)
        return d


def artifacts_dir() -> Path:
    return Path(settings.artifacts_dir) / MODEL_NAME


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ training
def _top_terms(vectorizer, clf, k: int = 10) -> dict[str, list[dict[str, float]]]:
    """Highest-weight terms per class (interpretability for the report/viva)."""
    names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_
    classes = list(clf.classes_)
    if coef.shape[0] == 1:  # binary: one row, positive weights -> classes_[1]
        coef = np.vstack([-coef[0], coef[0]])
    out = {}
    for i, cls in enumerate(classes):
        idx = np.argsort(coef[i])[::-1][:k]
        out[str(cls)] = [{"term": str(names[j]), "weight": float(coef[i][j])} for j in idx]
    return out


def train_target(
    target: str,
    splits: dict[str, pd.DataFrame],
    pre: TextPreprocessor,
    cfg: TfidfConfig,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Train one target. Returns ``(bundle_or_None, metrics_entry)``."""
    frames = {
        name: df[df[target] != ""].reset_index(drop=True)
        for name, df in splits.items()
        if df is not None
    }
    train = frames["train"]
    classes = sorted(train[target].unique())
    if len(train) == 0 or len(classes) < 2:
        return None, {
            "status": "skipped",
            "reason": f"needs >= 2 classes with labelled training rows; "
            f"found {len(classes)} class(es) in {len(train)} rows",
        }

    docs = {n: [pre.to_string(t) for t in df["text"]] for n, df in frames.items()}

    vec = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        max_features=cfg.max_features,
    )
    try:
        X_train = vec.fit_transform(docs["train"])
    except ValueError as exc:
        return None, {"status": "skipped", "reason": f"TF-IDF failed: {exc}"}
    X = {"train": X_train}
    for name in frames:
        if name != "train":
            X[name] = vec.transform(docs[name])
    y = {n: df[target].tolist() for n, df in frames.items()}

    # ---- choose C on the validation split (macro-F1); ties -> smaller C
    have_val = len(frames.get("val", [])) > 0
    grid_log: list[dict[str, float]] = []
    best_c, best_score, best_clf = None, -1.0, None
    for c in sorted(cfg.c_grid):
        clf = LogisticRegression(
            C=c,
            class_weight=cfg.class_weight,
            max_iter=cfg.max_iter,
            random_state=seed,
        ).fit(X["train"], y["train"])
        if have_val:
            score = float(
                f1_score(y["val"], clf.predict(X["val"]), labels=classes,
                         average="macro", zero_division=0)
            )
            grid_log.append({"C": c, "val_f1_macro": score})
        else:
            score = 0.0
        if best_clf is None or score > best_score:
            best_c, best_score, best_clf = c, score, clf
        if not have_val:  # no validation rows: use the smallest C, say so
            break
    clf = best_clf

    entry: dict[str, Any] = {
        "status": "trained",
        "labels": [str(c) for c in clf.classes_],
        "n_train": len(train),
        "vocabulary_size": len(vec.vocabulary_),
        "chosen_C": best_c,
        "C_selected_on": "validation macro-F1" if have_val else "default (no validation rows)",
        "C_grid": grid_log,
        "splits": {},
        "top_terms": _top_terms(vec, clf),
    }
    for name in EVAL_SPLITS:
        if name not in frames or len(frames[name]) == 0:
            continue
        pred = clf.predict(X[name])
        entry["splits"][name] = classification_metrics(
            y[name], pred, list(clf.classes_), train_labels=y["train"]
        )
    bundle = {
        "target": target,
        "vectorizer": vec,
        "classifier": clf,
        "labels": [str(c) for c in clf.classes_],
        "preprocess_config": pre.config.to_dict(),
        "tfidf_config": cfg.to_dict(),
        "seed": seed,
    }
    return bundle, entry


def train_all(
    *,
    targets: Sequence[str] = TARGETS,
    cfg: TfidfConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
    seed: int = data_mod.DEFAULT_SEED,
) -> dict[str, Any]:
    """Train every requested target from ``data/processed/*.csv`` and save artifacts.

    Raises ``DataNotFoundError`` if the processed splits do not exist and
    ``NothingTrainedError`` if no target could be trained. In both cases nothing
    is written, so /health keeps reporting "not trained".
    """
    cfg = cfg or TfidfConfig()
    pre = TextPreprocessor(preprocess_cfg)
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown target(s) {unknown}; choose from {TARGETS}")

    splits: dict[str, pd.DataFrame] = {
        "train": data_mod.load_split("public_train"),
        "val": data_mod.load_split("public_val"),
        "test": data_mod.load_split("public_test"),
    }
    heldout = None
    if data_mod.processed_path("pdf_heldout").is_file():
        heldout = data_mod.load_split("pdf_heldout")
        splits["pdf_heldout"] = heldout

    bundles: dict[str, dict[str, Any]] = {}
    target_entries: dict[str, Any] = {}
    for target in targets:
        bundle, entry = train_target(target, splits, pre, cfg, seed)
        target_entries[target] = entry
        if bundle is not None:
            bundles[target] = bundle
    if not bundles:
        reasons = {t: e.get("reason") for t, e in target_entries.items()}
        raise NothingTrainedError(f"No target could be trained: {reasons}")

    manifest = None
    if data_mod.manifest_path().is_file():
        manifest = json.loads(data_mod.manifest_path().read_text(encoding="utf-8"))

    metrics = {
        "model": MODEL_NAME,
        "description": "TF-IDF (word 1-2 grams) + Logistic Regression, one classifier per target",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data": {
            "rows": {name: len(df) for name, df in splits.items()},
            "pdf_heldout_evaluated": heldout is not None,
            "split_manifest": manifest,
        },
        "preprocessing": pre.config.to_dict(),
        "tfidf": cfg.to_dict(),
        "targets": target_entries,
    }

    out = artifacts_dir()
    out.mkdir(parents=True, exist_ok=True)
    # remove models from an earlier run so inference can never mix versions
    for old in out.glob("*.joblib"):
        old.unlink()
    for target, bundle in bundles.items():
        joblib.dump(bundle, out / f"{target}.joblib")
    _write_json_atomic(out / "metrics.json", metrics)  # marker: written last
    return metrics


# ----------------------------------------------------------------- inference
class TfidfClassifier:
    """Loads the saved TF-IDF models and classifies text."""

    def __init__(self, bundles: dict[str, dict[str, Any]], metrics: dict[str, Any]):
        self.bundles = bundles
        self.metrics = metrics
        self._pre = {
            t: TextPreprocessor(PreprocessConfig.from_dict(b["preprocess_config"]))
            for t, b in bundles.items()
        }

    @property
    def targets(self) -> list[str]:
        return sorted(self.bundles)

    @property
    def trained_at(self) -> str | None:
        return self.metrics.get("trained_at")

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "TfidfClassifier":
        d = Path(directory) if directory else artifacts_dir()
        metrics_file = d / "metrics.json"
        if not metrics_file.is_file():
            raise ModelNotTrainedError(
                "TF-IDF model not trained yet (artifacts/tfidf/metrics.json is missing). "
                "Run: python -m training.prepare_data ... then python -m training.train_tfidf"
            )
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        bundles = {}
        for target, entry in metrics.get("targets", {}).items():
            if entry.get("status") != "trained":
                continue
            f = d / f"{target}.joblib"
            if not f.is_file():
                raise ModelNotTrainedError(
                    f"metrics.json lists '{target}' as trained but {f.name} is missing. "
                    "Re-run training (model files are not committed to git)."
                )
            bundles[target] = joblib.load(f)
        if not bundles:
            raise ModelNotTrainedError("metrics.json exists but contains no trained target")
        return cls(bundles, metrics)

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[dict[str, Any]]:
        """Per text and per target: label, confidence and the top-k alternatives.

        ``confidence`` is the Logistic Regression probability. It is not a
        calibrated guarantee. A text with no usable tokens after preprocessing
        gets ``None`` instead of a meaningless guess.
        """
        results: list[dict[str, Any]] = [{"predictions": {}} for _ in texts]
        for target, bundle in self.bundles.items():
            pre = self._pre[target]
            docs = [pre.to_string(t) for t in texts]
            usable = [i for i, d in enumerate(docs) if d]
            probs = None
            if usable:
                X = bundle["vectorizer"].transform([docs[i] for i in usable])
                probs = bundle["classifier"].predict_proba(X)
            classes = list(bundle["classifier"].classes_)
            pos = {i: k for k, i in enumerate(usable)}
            for i in range(len(texts)):
                if i not in pos:
                    results[i]["predictions"][target] = {
                        "label": None,
                        "confidence": None,
                        "top": [],
                        "warning": "no usable tokens after preprocessing",
                    }
                    continue
                row = probs[pos[i]]
                order = np.argsort(row)[::-1][:top_k]
                results[i]["predictions"][target] = {
                    "label": str(classes[order[0]]),
                    "confidence": float(row[order[0]]),
                    "top": [
                        {"label": str(classes[j]), "probability": float(row[j])}
                        for j in order
                    ],
                }
        return results
