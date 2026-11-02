"""Train the PyTorch LSTM text classifiers (subject / topic / difficulty).

CPU-friendly development run (the defaults):
    python -m training.train_lstm

Real run on a GPU / Google Colab (larger model, more epochs):
    python -m training.train_lstm --device auto --embedding-dim 200 --hidden-dim 256 \
        --num-layers 2 --epochs 40 --batch-size 128 --patience 6

Start from the Word2Vec embeddings of Phase 3 (they must have been trained on the
current public_train.csv, otherwise training is refused):
    python -m training.train_lstm --init-embeddings word2vec

Reads data/processed/public_{train,val,test}.csv (the Phase 2 split, seed 42) and the
optional pdf_heldout.csv. Validation drives early stopping; test and PDF held-out
are only scored afterwards. Writes artifacts/lstm/<target>.pt and, last, metrics.json.

Exit codes: 0 trained | 2 NOT TRAINED YET (no data / nothing trainable / bad setup) |
3 PyTorch is not installed.
"""
from __future__ import annotations

from training._seq_cli import run


def main(argv: list[str] | None = None) -> int:
    return run("lstm", argv, __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
