"""Word2Vec endpoints (Phase 3).

* GET  /models/word2vec/status       trained? variants, corpus, vocabulary, coverage
* GET  /models/word2vec/similar      nearest words (cosine)
* GET  /models/word2vec/similarity   cosine similarity of two words
* GET  /models/word2vec/analogy      a : b :: c : ?
* GET  /models/word2vec/vector       the embedding vector of a word
* POST /models/word2vec/visualize    PCA / t-SNE coordinates as JSON

Words are passed as query parameters (not path segments) so spaces and Devanagari
need no special URL handling. When no model is trained every endpoint except
/status answers 503 ``not_trained_yet``; nothing is ever faked.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints

from .word2vec import (
    MODEL_NAME,
    EmbeddingService,
    UnknownVariantError,
    VisualizationError,
    Word2VecNotTrainedError,
    WordNotInVocabularyError,
    artifacts_dir,
)

router = APIRouter(prefix="/models/word2vec", tags=["Word2Vec"])

Variant = Optional[Literal["cbow", "skipgram"]]
Word = Annotated[str, StringConstraints(min_length=1, max_length=100)]

_cache: dict = {}


def _not_trained_detail(message: str) -> dict:
    return {"status": "not_trained_yet", "model": MODEL_NAME, "message": message}


def _service() -> EmbeddingService:
    meta = artifacts_dir() / "metadata.json"
    if not meta.is_file():
        _cache.clear()
        raise HTTPException(
            status_code=503,
            detail=_not_trained_detail("Word2Vec model not trained yet (artifacts/word2vec/metadata.json is missing)."),
        )
    key = (str(meta), meta.stat().st_mtime_ns)  # a retrain is picked up automatically
    if _cache.get("key") != key:
        try:
            _cache["svc"] = EmbeddingService.load()
        except Word2VecNotTrainedError as exc:
            raise HTTPException(status_code=503, detail=_not_trained_detail(str(exc))) from exc
        _cache["key"] = key
    return _cache["svc"]


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except WordNotInVocabularyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"status": "word_not_in_vocabulary", "words": exc.words, "variant": exc.variant},
        ) from exc
    except UnknownVariantError as exc:
        raise HTTPException(
            status_code=404,
            detail={"status": "variant_not_trained", "variant": exc.variant, "available": exc.available},
        ) from exc
    except VisualizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/status")
def word2vec_status() -> dict:
    """Always 200: reports whether a real trained model exists."""
    meta = artifacts_dir() / "metadata.json"
    if not meta.is_file():
        return {
            "model": MODEL_NAME,
            "trained": False,
            "status": "not_trained_yet",
            "variants_available": [],
            "message": "Word2Vec has not been trained. Run training.prepare_data then training.train_word2vec.",
        }
    return _service().status()


@router.get("/similar")
def similar_words(
    word: Annotated[str, Query(min_length=1, max_length=100)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    variant: Variant = None,
) -> dict:
    return _run(_service().most_similar, word, top_k, variant)


@router.get("/similarity")
def word_similarity(
    word_a: Annotated[str, Query(min_length=1, max_length=100)],
    word_b: Annotated[str, Query(min_length=1, max_length=100)],
    variant: Variant = None,
) -> dict:
    return _run(_service().similarity, word_a, word_b, variant)


@router.get("/analogy")
def analogy(
    a: Annotated[str, Query(min_length=1, max_length=100)],
    b: Annotated[str, Query(min_length=1, max_length=100)],
    c: Annotated[str, Query(min_length=1, max_length=100)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    variant: Variant = None,
) -> dict:
    return _run(_service().analogy, a, b, c, top_k, variant)


@router.get("/vector")
def word_vector(
    word: Annotated[str, Query(min_length=1, max_length=100)],
    variant: Variant = None,
) -> dict:
    return _run(_service().vector, word, variant)


class VisualizeRequest(BaseModel):
    method: Literal["pca", "tsne"] = "pca"
    variant: Variant = None
    # explicit words; if omitted the top_n most frequent words are plotted
    words: Optional[list[Word]] = Field(default=None, min_length=1, max_length=500)
    top_n: int = Field(default=100, ge=1, le=500)
    neighbors_per_word: int = Field(default=0, ge=0, le=20)
    perplexity: Optional[float] = Field(default=None, gt=1, le=100)


@router.post("/visualize")
def visualize(req: VisualizeRequest) -> dict:
    """2-D coordinates + metadata for a React scatter plot."""
    return _run(
        _service().visualize,
        method=req.method,
        variant=req.variant,
        words=req.words,
        top_n=req.top_n,
        neighbors_per_word=req.neighbors_per_word,
        perplexity=req.perplexity,
    )
