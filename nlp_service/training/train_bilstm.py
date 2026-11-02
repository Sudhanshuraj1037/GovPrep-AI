"""Train the PyTorch BiLSTM text classifiers (subject / topic / difficulty).

The BiLSTM is the bidirectional counterpart of the LSTM: it takes exactly the same flags,
data, seed, preprocessing and evaluation, so the two models are directly comparable.

CPU-friendly development run (the defaults):
    python -m training.train_bilstm

Real run on a GPU / Google Colab (larger model, more epochs):
    python -m training.train_bilstm --device auto --embedding-dim 200 --hidden-dim 256 \\
        --num-layers 2 --epochs 40 --batch-size 128 --patience 6

Start from the Phase 3 Word2Vec embeddings (refused unless Word2Vec was trained on the
current public_train.csv):
    python -m training.train_bilstm --init-embeddings word2vec

Note: --hidden-dim is the size PER DIRECTION, so a BiLSTM has about twice the recurrent
parameters of an LSTM trained with the same flags. The parameter counts are saved in
metrics.json so a comparison can take it into account.

Reads data/processed/public_{train,val,test}.csv (the Phase 2 split, seed 42) and the
optional pdf_heldout.csv. Validation drives early stopping; test and PDF held-out are only
scored afterwards. Writes artifacts/bilstm/<target>.pt and, last, metrics.json.

Exit codes: 0 trained | 2 NOT TRAINED YET (no data / nothing trainable / bad setup) |
3 PyTorch is not installed.
"""
from __future__ import annotations

from training._seq_cli import run


def main(argv: list[str] | None = None) -> int:
    return run("bilstm", argv, __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
