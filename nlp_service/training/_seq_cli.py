"""Shared command-line trainer for the recurrent models (``train_lstm`` and ``train_bilstm``).

Both commands take exactly the same flags and run exactly the same training code
(``app.lstm.train_all``); only the architecture differs. Keeping one implementation means
the LSTM and BiLSTM runs are configured, trained and evaluated identically.

Exit codes: 0 trained | 2 NOT TRAINED YET (no data / nothing trainable / bad setup) |
3 PyTorch is not installed.
"""
from __future__ import annotations

import argparse
import sys

from app.data import DEFAULT_SEED, DataNotFoundError, DataValidationError
from app.preprocessing import NORMALIZERS, PreprocessConfig


def run(model: str, argv: list[str] | None, doc: str | None) -> int:
    """Train ``model`` ("lstm" or "bilstm") from the command line. Returns the exit code."""
    label = {"lstm": "LSTM", "bilstm": "BiLSTM"}[model]
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["subject", "topic", "difficulty"],
                    choices=["subject", "topic", "difficulty"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--embedding-dim", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-vocab", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4, help="early stopping on val macro-F1; 0 disables")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--init-embeddings", choices=["none", "word2vec"], default="none")
    ap.add_argument("--w2v-variant", choices=["cbow", "skipgram"], default="skipgram")
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--normalizer", choices=NORMALIZERS, default="none")
    ap.add_argument("--remove-stopwords", action="store_true",
                    help="drop stop-words (kept by default: word order is what the recurrent model reads)")
    a = ap.parse_args(argv)

    try:
        from app.lstm import SPECS, LSTMConfig, LSTMTrainingError, train_all
    except ImportError as exc:
        print(
            f"{label}: cannot start, PyTorch is not installed ({exc}).\n"
            "Install it (CPU build is enough): pip install torch\n"
            "  Linux CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 3
    spec = SPECS[model]

    try:
        cfg = LSTMConfig(
            embedding_dim=a.embedding_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
            dropout=a.dropout, max_len=a.max_len, min_freq=a.min_freq, max_vocab=a.max_vocab,
            batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
            grad_clip=a.grad_clip, patience=a.patience, class_weights=not a.no_class_weights,
            init_embeddings=a.init_embeddings, w2v_variant=a.w2v_variant,
            freeze_embeddings=a.freeze_embeddings, device=a.device,
        )
        pre_cfg = PreprocessConfig(normalizer=a.normalizer, remove_stopwords=a.remove_stopwords)
    except ValueError as exc:
        print(f"{label}: invalid setting: {exc}", file=sys.stderr)
        return 2

    def show(target: str, r: dict) -> None:
        if "val_f1_macro" in r:
            print(f"  [{target}] epoch {r['epoch']:>3}  train_loss {r['train_loss']:.4f}  "
                  f"val_loss {r['val_loss']:.4f}  val_acc {r['val_accuracy']:.4f}  val_macro-F1 {r['val_f1_macro']:.4f}")
        else:
            print(f"  [{target}] epoch {r['epoch']:>3}  train_loss {r['train_loss']:.4f}")

    try:
        print(f"Training {label} (real training on the processed splits)...")
        metrics = train_all(targets=a.targets, cfg=cfg, preprocess_cfg=pre_cfg, seed=a.seed, on_epoch=show,
                            spec=spec)
    except DataNotFoundError as exc:
        print(f"{label}: NOT TRAINED YET. {exc}\n"
              "Create the splits first:  python -m training.prepare_data --input <labelled.csv>",
              file=sys.stderr)
        return 2
    except (LSTMTrainingError, DataValidationError) as exc:
        print(f"{label}: NOT TRAINED YET. {exc}", file=sys.stderr)
        return 2

    emb = metrics["embeddings"]
    print(f"\n{label} training finished on {metrics['framework']['device']} "
          f"(torch {metrics['framework']['torch']}). Vocabulary: {metrics['vocabulary']['size']} words.")
    if emb["init"] == "word2vec":
        print(f"  embeddings initialised from Word2Vec/{emb['variant']}: "
              f"{emb['words_covered']}/{emb['vocabulary_words']} words covered ({emb['coverage']:.1%})")
    print("Metrics (computed on real predictions):")
    for target, e in metrics["targets"].items():
        if e["status"] != "trained":
            print(f"  {target:<11} skipped: {e['reason']}")
            continue
        t = e["training"]
        print(f"  {target:<11} classes={len(e['labels'])}  n_train={e['n_train']}  "
              f"best epoch {t['best_epoch']}/{t['epochs_run']}"
              f"{' (stopped early)' if t['stopped_early'] else ''}")
        for split, m in e["splits"].items():
            print(f"      {split:<12} n={m['n']:<6} acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}  "
                  f"weighted-F1={m['f1_weighted']:.4f}  (majority-class acc={m['majority_class_baseline']['accuracy']:.4f})")
    print(f"Saved to artifacts/{model}/ (metrics.json written last).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
