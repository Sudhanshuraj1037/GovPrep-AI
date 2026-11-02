"""Fine-tune a pretrained BERT for subject / topic / difficulty classification.

RECOMMENDED for real training: Google Colab (or any machine) with a GPU:
    python -m training.train_bert --device auto --pretrained bert-base-uncased \
        --epochs 4 --batch-size 32 --max-len 128 --lr 2e-5 --patience 2

CPU development run (works, but is slow; DistilBERT is about half the size of BERT):
    python -m training.train_bert --device cpu --pretrained distilbert-base-uncased \
        --max-len 64 --epochs 2 --batch-size 8

Hindi / multilingual text:
    python -m training.train_bert --pretrained bert-base-multilingual-cased

The checkpoint is NEVER trained from scratch: --pretrained is a Hugging Face id (downloaded on
first use, internet needed) or a local directory with a saved model + tokenizer. --freeze-encoder
trains only the classification head (a cheap CPU smoke test; it is NOT full fine-tuning and is
recorded as such in metrics.json).

Same data protocol as TF-IDF / LSTM / BiLSTM: reads data/processed/public_{train,val,test}.csv
(the Phase 2 split, seed 42) and the optional pdf_heldout.csv. Validation drives best-epoch
selection and early stopping; test and PDF held-out are only scored afterwards.
Writes artifacts/bert/<target>.pt, artifacts/bert/tokenizer/ and, last, artifacts/bert/metrics.json.
(about 440 MB per target for bert-base).

Exit codes: 0 trained | 2 NOT TRAINED YET (no data / nothing trainable / pretrained model
unavailable / bad setup) | 3 PyTorch or Transformers is not installed.
"""
from __future__ import annotations

import argparse
import sys

from app.data import DEFAULT_SEED, DataNotFoundError, DataValidationError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["subject", "topic", "difficulty"],
                    choices=["subject", "topic", "difficulty"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pretrained", default="bert-base-uncased",
                    help="Hugging Face model id or local directory (default: %(default)s)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=2, help="early stopping on val macro-F1; 0 disables")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the classification head")
    ap.add_argument("--no-clean-text", action="store_true", help="feed raw text to the tokenizer")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = ap.parse_args(argv)

    try:
        from app.bert import BertFineTuneConfig, BertTrainingError, train_all
    except ImportError as exc:
        print(
            f"BERT: cannot start, PyTorch or Transformers is not installed ({exc}).\n"
            "Install them:  pip install torch transformers\n"
            "  Linux CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 3

    try:
        cfg = BertFineTuneConfig(
            pretrained=a.pretrained, max_len=a.max_len, batch_size=a.batch_size, epochs=a.epochs,
            lr=a.lr, weight_decay=a.weight_decay, warmup_ratio=a.warmup_ratio, grad_clip=a.grad_clip,
            patience=a.patience, class_weights=not a.no_class_weights, freeze_encoder=a.freeze_encoder,
            clean_text=not a.no_clean_text, device=a.device,
        )
    except ValueError as exc:
        print(f"BERT: invalid setting: {exc}", file=sys.stderr)
        return 2

    def show(target: str, r: dict) -> None:
        if "val_f1_macro" in r:
            print(f"  [{target}] epoch {r['epoch']:>2}  train_loss {r['train_loss']:.4f}  "
                  f"val_loss {r['val_loss']:.4f}  val_acc {r['val_accuracy']:.4f}  val_macro-F1 {r['val_f1_macro']:.4f}")
        else:
            print(f"  [{target}] epoch {r['epoch']:>2}  train_loss {r['train_loss']:.4f}")

    try:
        print(f"Fine-tuning {cfg.pretrained} (real training on the processed splits)...")
        if cfg.freeze_encoder:
            print("  NOTE: --freeze-encoder is on: only the classification head is trained (not full fine-tuning).")
        metrics = train_all(targets=a.targets, cfg=cfg, seed=a.seed, on_epoch=show)
    except DataNotFoundError as exc:
        print(f"BERT: NOT TRAINED YET. {exc}\n"
              "Create the splits first:  python -m training.prepare_data --input <labelled.csv>",
              file=sys.stderr)
        return 2
    except (BertTrainingError, DataValidationError) as exc:
        print(f"BERT: NOT TRAINED YET. {exc}", file=sys.stderr)
        return 2

    fw, pre = metrics["framework"], metrics["pretrained"]
    print(f"\nBERT fine-tuning finished on {fw['device']} (torch {fw['torch']}, transformers {fw['transformers']}). "
          f"Pretrained: {pre['name']} ({pre['model_type']}).")
    print("Metrics (computed on real predictions):")
    for target, e in metrics["targets"].items():
        if e["status"] != "trained":
            print(f"  {target:<11} skipped: {e['reason']}")
            continue
        t = e["training"]
        print(f"  {target:<11} classes={len(e['labels'])}  n_train={e['n_train']}  "
              f"trainable params={e['model']['trainable_parameters']:,}  "
              f"best epoch {t['best_epoch']}/{t['epochs_run']}{' (stopped early)' if t['stopped_early'] else ''}")
        for split, m in e["splits"].items():
            print(f"      {split:<12} n={m['n']:<6} acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}  "
                  f"weighted-F1={m['f1_weighted']:.4f}  (majority-class acc={m['majority_class_baseline']['accuracy']:.4f})")
    print("Saved to artifacts/bert/ (metrics.json written last).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
