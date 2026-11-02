"""BiLSTM endpoints (Phase 5): the same three endpoints as the LSTM, built by the same factory.

* GET  /models/bilstm/status    always 200: trained? targets, hyper-parameters, headline scores
* GET  /models/bilstm/metrics   the real metrics.json of the last training run
* POST /classify/bilstm         subject / topic / difficulty for texts

PyTorch is not imported at load time (see ``api_lstm``); without it /classify/bilstm
answers 503 ``dependency_missing`` and everything else keeps working.
"""
from __future__ import annotations

from .api_lstm import build_router

MODEL_NAME = "bilstm"
_cache: dict = {}


def _import_bilstm():
    """Import the torch-dependent module on demand (separate function so tests can patch it)."""
    from . import bilstm

    return bilstm


router = build_router(MODEL_NAME, "BiLSTM", lambda: _import_bilstm(), "BiLSTMService", _cache)
