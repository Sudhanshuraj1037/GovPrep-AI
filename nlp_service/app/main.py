"""FastAPI entry point for the GovPrep-AI NLP/DL service.

Phase 1: service skeleton + /health.
Phase 2: /preprocess, /classify/tfidf, /models/tfidf/metrics.
Phase 3: /models/word2vec/* (status, similar, similarity, analogy, vector, visualize).
Phase 4: /models/lstm/status, /models/lstm/metrics, /classify/lstm (PyTorch is imported lazily).
Phase 5: the same three endpoints for the BiLSTM (/models/bilstm/status, /models/bilstm/metrics, /classify/bilstm).
Phase 6: the same three endpoints for BERT (/models/bert/status, /models/bert/metrics, /classify/bert).
Phase 7: /models/comparison and /models/comparison/rows (TF-IDF vs LSTM vs BiLSTM vs BERT, from real metrics only).
Later phases add preprocessing, TF-IDF, Word2Vec, LSTM, BiLSTM and BERT
endpoints as separate routers; the Node/Express backend is the only intended
client (React never calls this service directly), so CORS is deliberately off.

Run from the nlp_service/ folder:
    python -m uvicorn app.main:app --port 8001 --reload
"""
from __future__ import annotations

import logging
import platform
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from . import __version__
from .api_bert import router as bert_router
from .api_bilstm import router as bilstm_router
from .api_comparison import router as comparison_router
from .api_lstm import router as lstm_router
from .api_nlp import router as nlp_router
from .api_word2vec import router as word2vec_router
from .config import settings
from .labels import DIFFICULTIES, SUBJECTS
from .registry import dataset_status, dependency_status, model_status

logging.basicConfig(level=settings.log_level.upper())
log = logging.getLogger("govprep.nlp")

_STARTED = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("GovPrep-AI NLP service v%s starting", __version__)
    log.info("artifacts dir: %s", settings.artifacts_dir)
    log.info("data dir:      %s", settings.data_dir)
    yield
    log.info("GovPrep-AI NLP service stopped")


app = FastAPI(
    title="GovPrep-AI NLP/DL Service",
    version=__version__,
    description=(
        "Deep Learning + NLP layer for GovPrep-AI (CSE472). Classical NLP, "
        "TF-IDF, Word2Vec, LSTM, BiLSTM and BERT text classifiers. "
        "Generative features (question generation, study assistant) stay in the "
        "Node backend and use Groq; they are not part of this service."
    ),
    lifespan=lifespan,
)


app.include_router(nlp_router)
app.include_router(lstm_router)
app.include_router(bilstm_router)
app.include_router(bert_router)
app.include_router(comparison_router)
app.include_router(word2vec_router)


@app.get("/health", tags=["System"])
def health() -> dict:
    """Liveness + honest status of what is installed, trained and available."""
    models = model_status()
    return {
        "status": "ok",
        "service": "govprep-nlp-service",
        "version": __version__,
        "time": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "python": platform.python_version(),
        "random_seed": settings.random_seed,
        "labels": {"subjects": SUBJECTS, "difficulties": DIFFICULTIES},
        "models": models,
        "models_trained": sum(m["trained"] for m in models),
        "datasets": dataset_status(),
        "dependencies": dependency_status(),
    }
