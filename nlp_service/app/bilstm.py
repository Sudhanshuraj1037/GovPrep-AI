"""BiLSTM text classifier (Phase 5): a bidirectional extension of the Phase 4 LSTM.

This is a thin façade. All the machinery (vocabulary, encoding, padding / truncation,
embedding layer, dropout, packed-sequence LSTM, classification head, training and
validation loops, early stopping, checkpoint saving / loading, inference, metrics) lives
once in ``app.lstm`` and is selected here with ``BILSTM_SPEC``. That guarantees the
LSTM and the BiLSTM are trained and evaluated by exactly the same code, on exactly the
same data, so any difference between them comes from the architecture alone.

What differs from the LSTM
--------------------------
* ``nn.LSTM(bidirectional=True)``: one LSTM reads the sequence left-to-right, another
  right-to-left. The classifier receives the concatenation of the forward final state
  (after the last real token) and the backward final state (after the first real token),
  i.e. ``2 * hidden_dim`` features. ``hidden_dim`` is per direction, so the recurrent
  parameter count is about doubled; ``metrics.json`` records the parameter counts.
* Artifacts live under ``artifacts/bilstm/`` and checkpoints carry ``arch: "bilstm"``;
  an LSTM checkpoint cannot be loaded as a BiLSTM (or the other way round).

Everything else (data protocol, seed 42, splits, preprocessing defaults, class weights,
early stopping on validation macro-F1, held-out discipline, metrics) is inherited
unchanged. PyTorch is imported through ``app.lstm`` only; ``app.main`` never imports this
module, so the service keeps working when PyTorch is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from . import data as data_mod
from .lstm import (
    BILSTM_SPEC,
    LSTMConfig,
    LSTMNotTrainedError,
    LSTMService,
    LSTMTrainingError,
    TARGETS,
    default_preprocess_config,
)
from .lstm import artifacts_dir as _artifacts_dir
from .lstm import train_all as _train_all
from .preprocessing import PreprocessConfig

MODEL_NAME = BILSTM_SPEC.name

# Same configuration class and the same errors as the LSTM: one set of hyper-parameters
# and one failure vocabulary for both recurrent models.
BiLSTMConfig = LSTMConfig
BiLSTMNotTrainedError = LSTMNotTrainedError
BiLSTMTrainingError = LSTMTrainingError

__all__ = [
    "MODEL_NAME", "TARGETS", "BiLSTMConfig", "BiLSTMNotTrainedError", "BiLSTMTrainingError",
    "BiLSTMService", "artifacts_dir", "default_preprocess_config", "train_all",
]


def artifacts_dir() -> Path:
    return _artifacts_dir(BILSTM_SPEC)


def train_all(
    *,
    targets: Sequence[str] = TARGETS,
    cfg: LSTMConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
    seed: int = data_mod.DEFAULT_SEED,
    on_epoch: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train the BiLSTM for every requested target and save ``artifacts/bilstm/``.

    Same contract as ``app.lstm.train_all``: raises ``DataNotFoundError`` if the processed
    splits are missing, ``LSTMTrainingError`` if nothing can be trained (nothing is written
    then), and writes ``metrics.json`` last.
    """
    return _train_all(
        targets=targets, cfg=cfg, preprocess_cfg=preprocess_cfg, seed=seed,
        on_epoch=on_epoch, spec=BILSTM_SPEC,
    )


class BiLSTMService(LSTMService):
    """Loads ``artifacts/bilstm/`` and classifies text."""

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "BiLSTMService":  # type: ignore[override]
        return super().load(directory, spec=BILSTM_SPEC)  # type: ignore[return-value]
