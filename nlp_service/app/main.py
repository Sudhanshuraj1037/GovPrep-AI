
"""FastAPI entry point for the GovPrep-AI NLP/DL service.

Phase 1: service skeleton + /health.
Phase 2: /preprocess, /classify/tfidf, /models/tfidf/metrics.
Phase 3: /models/word2vec/* endpoints.
Phase 4: LSTM endpoints.
Phase 5: BiLSTM endpoints.
Phase 6: BERT endpoints.
Phase 7: Model comparison endpoints.

Run from the nlp_service/ folder:
    python -m uvicorn app.main:app --port 8001 --reload
"""
from __future__ import annotations

import logging
import os
import platform
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from huggingface_hub import snapshot_download

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


def download_bert_artifacts():
    """Download private BERT artifacts if they are not already available."""

    repo_id = os.getenv(
        "HF_MODEL_REPO",
        "sudhanshuraj1037/govprep-bert-artifacts",
    )

    target_dir = settings.artifacts_dir / "bert"
    target_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        target_dir / "subject.pt",
        target_dir / "metrics.json",
        target_dir / "tokenizer" / "tokenizer.json",
        target_dir / "tokenizer" / "tokenizer_config.json",
    ]

    # Local development: reuse existing artifacts without requiring a token.
    if all(file.is_file() for file in required_files):
        log.info("BERT artifacts already available.")
        return

    # Production: download missing artifacts from the private Hugging Face repo.
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Cannot access private BERT repository."
        )

    log.info("Downloading BERT artifacts from Hugging Face: %s", repo_id)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=str(target_dir),
        allow_patterns=[
            "subject.pt",
            "metrics.json",
            "tokenizer/*",
        ],
    )

    missing = [
        str(file)
        for file in required_files
        if not file.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"BERT artifact download incomplete. Missing: {missing}"
        )

    log.info("BERT artifacts downloaded and verified.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("GovPrep-AI NLP service v%s starting", __version__)
    log.info("artifacts dir: %s", settings.artifacts_dir)
    log.info("data dir:      %s", settings.data_dir)

    download_bert_artifacts()

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
        "labels": {
            "subjects": SUBJECTS,
            "difficulties": DIFFICULTIES,
        },
        "models": models,
        "models_trained": sum(m["trained"] for m in models),
        "datasets": dataset_status(),
        "dependencies": dependency_status(),
    }
