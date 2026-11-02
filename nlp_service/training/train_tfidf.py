"""Train the TF-IDF + Logistic Regression baseline from data/processed/*.csv.

    python -m training.train_tfidf
    python -m training.train_tfidf --targets subject difficulty --normalizer none

Writes artifacts/tfidf/<target>.joblib and, last, artifacts/tfidf/metrics.json.
If the data is missing or nothing can be trained it prints
"NOT TRAINED YET", writes nothing and exits with code 2.
"""
from __future__ import annotations

import argparse
import sys

from app.data import DEFAULT_SEED, DataNotFoundError, DataValidationError
from app.preprocessing import NORMALIZERS, PreprocessConfig
from app.tfidf import TARGETS, NothingTrainedError, TfidfConfig, train_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--normalizer", choices=NORMALIZERS, default="stem")
    ap.add_argument("--keep-stopwords", action="store_true")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=50_000)
    ap.add_argument("--ngram-max", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    a = ap.parse_args(argv)

    pre_cfg = PreprocessConfig(normalizer=a.normalizer, remove_stopwords=not a.keep_stopwords)
    cfg = TfidfConfig(
        ngram_range=(1, a.ngram_max),
        min_df=a.min_df,
        max_features=a.max_features,
        c_grid=tuple(a.c_grid),
    )
    try:
        metrics = train_all(targets=a.targets, cfg=cfg, preprocess_cfg=pre_cfg, seed=a.seed)
    except DataNotFoundError as exc:
        print(f"TF-IDF: NOT TRAINED YET. {exc}\n"
              "Create the splits first:  python -m training.prepare_data --input <labelled.csv>",
              file=sys.stderr)
        return 2
    except (NothingTrainedError, DataValidationError) as exc:
        print(f"TF-IDF: NOT TRAINED YET. {exc}", file=sys.stderr)
        return 2

    print("TF-IDF training finished. Metrics (computed on real predictions):")
    for target, entry in metrics["targets"].items():
        if entry["status"] != "trained":
            print(f"  {target:<11} skipped: {entry['reason']}")
            continue
        print(f"  {target:<11} classes={len(entry['labels'])}  n_train={entry['n_train']}  C={entry['chosen_C']}")
        for split, m in entry["splits"].items():
            print(f"      {split:<12} n={m['n']:<6} acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}"
                  f"  (majority-class acc={m['majority_class_baseline']['accuracy']:.4f})")
    print("Saved to artifacts/tfidf/ (metrics.json written last).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
