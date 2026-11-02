"""Train Word2Vec (CBOW and Skip-Gram) on the public TRAIN split.

    python -m training.train_word2vec
    python -m training.train_word2vec --variants skipgram --vector-size 200 --epochs 50

Reads ONLY data/processed/public_train.csv for training (validation, test and the PDF
held-out set are never used for training; token coverage is measured on them
afterwards). Writes artifacts/word2vec/{cbow,skipgram}.model, vocab_stats.json and,
last, metadata.json.

If the data is missing or no model can be trained it prints "NOT TRAINED YET",
writes nothing and exits with code 2.
"""
from __future__ import annotations

import argparse
import sys

from app.data import DEFAULT_SEED, DataNotFoundError, DataValidationError
from app.preprocessing import NORMALIZERS, PreprocessConfig
from app.word2vec import (
    VARIANTS,
    EmbeddingService,
    Word2VecConfig,
    Word2VecTrainingError,
    train_all,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--vector-size", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--normalizer", choices=NORMALIZERS, default="none")
    ap.add_argument("--keep-stopwords", action="store_true")
    a = ap.parse_args(argv)

    cfg = Word2VecConfig(
        vector_size=a.vector_size,
        window=a.window,
        min_count=a.min_count,
        negative=a.negative,
        epochs=a.epochs,
    )
    pre_cfg = PreprocessConfig(normalizer=a.normalizer, remove_stopwords=not a.keep_stopwords)
    try:
        meta = train_all(variants=a.variants, cfg=cfg, preprocess_cfg=pre_cfg, seed=a.seed)
    except DataNotFoundError as exc:
        print(
            f"WORD2VEC: NOT TRAINED YET. {exc}\n"
            "Create the splits first:  python -m training.prepare_data --input <labelled.csv>",
            file=sys.stderr,
        )
        return 2
    except (Word2VecTrainingError, DataValidationError) as exc:
        print(f"WORD2VEC: NOT TRAINED YET. {exc}", file=sys.stderr)
        return 2

    c = meta["corpus"]
    print(f"Word2Vec training finished on public_train: {c['documents']} documents, {c['tokens']} tokens.")
    for v, info in meta["variants"].items():
        print(f"  {v:<9} vocabulary={info['vocab_size']}  ({info['train_seconds']}s)")
    for split, cov in meta["vocabulary_coverage"].items():
        if cov["coverage"] is not None:
            print(f"  token coverage on {split:<12}: {cov['coverage']:.1%} of {cov['tokens']} tokens")
    cmp_ = meta["variant_comparison"]
    if cmp_:
        print(f"  CBOW vs Skip-Gram top-{cmp_['k']} neighbour overlap (Jaccard, {cmp_['words_compared']} words): "
              f"{cmp_['mean_jaccard']:.3f}")

    svc = EmbeddingService.load()
    print("Nearest neighbours of the most frequent words (real model output):")
    for variant in svc.variants:
        wv = svc.vectors[variant]
        for w in wv.index_to_key[:3]:
            near = ", ".join(x["word"] for x in svc.most_similar(w, 5, variant)["neighbors"])
            print(f"  [{variant}] {w}: {near}")
    print("Saved to artifacts/word2vec/ (metadata.json written last).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
