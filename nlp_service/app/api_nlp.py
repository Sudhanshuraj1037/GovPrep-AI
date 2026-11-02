"""NLP endpoints (Phase 2): preprocessing demo, TF-IDF classification, TF-IDF metrics.

When no trained model exists these endpoints answer 503 / 404 with
``status: "not_trained_yet"``. They never return placeholder predictions or
made-up metrics.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from .preprocessing import PreprocessConfig, TextPreprocessor
from .tfidf import (
    MODEL_NAME,
    ModelNotTrainedError,
    TfidfClassifier,
    artifacts_dir,
)

router = APIRouter(tags=["NLP"])

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=5000)]


class PreprocessRequest(BaseModel):
    text: Annotated[str, StringConstraints(min_length=1, max_length=20000)]
    normalizer: Optional[Literal["none", "stem", "lemma"]] = None
    remove_stopwords: Optional[bool] = None


class ClassifyRequest(BaseModel):
    texts: list[ShortText] = Field(min_length=1, max_length=32)
    top_k: int = Field(default=3, ge=1, le=10)


def _not_trained(message: str) -> dict:
    return {"status": "not_trained_yet", "model": MODEL_NAME, "message": message}


# model cache keyed by (directory, metrics.json mtime) so a retrain is picked up
_cache: dict = {}


def _get_classifier() -> TfidfClassifier:
    metrics_file = artifacts_dir() / "metrics.json"
    if not metrics_file.is_file():
        _cache.clear()
        raise ModelNotTrainedError(
            "TF-IDF model not trained yet (artifacts/tfidf/metrics.json is missing)."
        )
    key = (str(metrics_file), metrics_file.stat().st_mtime_ns)
    if _cache.get("key") != key:
        _cache["clf"] = TfidfClassifier.load()
        _cache["key"] = key
    return _cache["clf"]


@router.post("/preprocess")
def preprocess(req: PreprocessRequest) -> dict:
    """Show every preprocessing stage for a piece of text."""
    overrides = {}
    if req.normalizer is not None:
        overrides["normalizer"] = req.normalizer
    if req.remove_stopwords is not None:
        overrides["remove_stopwords"] = req.remove_stopwords
    try:
        pre = TextPreprocessor(PreprocessConfig(**overrides))
        return pre.explain(req.text)
    except LookupError as exc:  # WordNet not downloaded
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("/classify/tfidf")
def classify_tfidf(req: ClassifyRequest) -> dict:
    """Classify texts with the trained TF-IDF baseline (subject / topic / difficulty)."""
    try:
        clf = _get_classifier()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=_not_trained(str(exc))) from exc
    results = clf.predict(req.texts, top_k=req.top_k)
    for text, res in zip(req.texts, results):
        res["text_preview"] = text[:120]
    all_targets = clf.metrics.get("targets", {})
    return {
        "model": MODEL_NAME,
        "trained_at": clf.trained_at,
        "targets_available": clf.targets,
        "targets_unavailable": {
            t: e.get("reason", "not trained")
            for t, e in all_targets.items()
            if e.get("status") != "trained"
        },
        "results": results,
    }


@router.get("/models/tfidf/metrics")
def tfidf_metrics() -> dict:
    """The real metrics.json written by the last training run."""
    f = artifacts_dir() / "metrics.json"
    if not f.is_file():
        raise HTTPException(
            status_code=404,
            detail=_not_trained("No metrics yet: the TF-IDF model has not been trained."),
        )
    return json.loads(f.read_text(encoding="utf-8"))
