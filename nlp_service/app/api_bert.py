"""BERT endpoints (Phase 6): the same three endpoints as the LSTM / BiLSTM, built by the same factory.

* GET  /models/bert/status    always 200: trained? targets, hyper-parameters, headline scores
* GET  /models/bert/metrics   the real metrics.json of the last training run
* POST /classify/bert         subject / topic / difficulty for texts

Neither PyTorch nor Transformers is imported at load time (see ``api_lstm``): the service and
/health start without them. Only /classify/bert needs them; if either is missing it answers
503 ``dependency_missing``. With no trained model the endpoints answer 503 / 404
``not_trained_yet``. Nothing is ever faked.
"""
from __future__ import annotations

from .api_lstm import build_router

MODEL_NAME = "bert"
_cache: dict = {}


def _import_bert():
    """Import the torch + transformers dependent module on demand (separate function so tests can patch it)."""
    from . import bert

    return bert


router = build_router(
    MODEL_NAME, "BERT", lambda: _import_bert(), "BertService", _cache,
    not_trained_exc="BertNotTrainedError", packages=("torch", "transformers"),
)
