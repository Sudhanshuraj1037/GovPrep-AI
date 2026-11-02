"""Validate a labelled CSV and create the reproducible train/val/test split.

    python -m training.prepare_data --input data/public/questions.csv
    python -m training.prepare_data --input data/public/questions.csv \
        --heldout data/pdf_heldout/govprep_pdfs.csv

CSV columns: text, subject, topic, difficulty  (topic / difficulty may be blank).
Writes data/processed/public_{train,val,test}.csv, pdf_heldout.csv (if given)
and split_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys

from app.data import DEFAULT_SEED, DataNotFoundError, DataValidationError, prepare_splits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="public labelled CSV")
    ap.add_argument("--heldout", help="PDF-derived CSV, used for final evaluation only")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--min-class-count", type=int, default=5)
    a = ap.parse_args(argv)
    try:
        manifest = prepare_splits(
            a.input,
            heldout_csv=a.heldout,
            seed=a.seed,
            val_size=a.val_size,
            test_size=a.test_size,
            min_class_count=a.min_class_count,
        )
    except (DataNotFoundError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
