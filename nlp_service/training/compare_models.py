"""Write the model-comparison report from the REAL metrics.json files.

    python -m training.compare_models                  # 1000 bootstrap resamples
    python -m training.compare_models --bootstrap 0    # no intervals

Reads artifacts/{tfidf,lstm,bilstm,bert}/metrics.json (whichever exist and satisfy the shared
metrics contract) and writes, under artifacts/comparison/:
    comparison.json   everything (what GET /models/comparison returns)
    comparison.md     a readable report with tables, intervals and caveats
    comparison.csv    one row per target / split / model

Needs no PyTorch or Transformers. With fewer than two comparable models it prints why, writes
nothing and exits with code 2. Nothing is ever estimated or filled in for a missing model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from app.comparison import DEFAULT_BOOTSTRAP, DEFAULT_SEED, comparison_rows, compare, to_markdown
from app.config import settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="resamples for the intervals (0 = none)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", help="output folder (default: artifacts/comparison)")
    a = ap.parse_args(argv)
    if a.bootstrap < 0:
        print("COMPARISON: --bootstrap must be >= 0", file=sys.stderr)
        return 2

    c = compare(n_bootstrap=a.bootstrap, seed=a.seed)
    if c["status"] != "ok":
        print(f"COMPARISON: NOT AVAILABLE YET. Models with valid metrics: {c['models_found'] or 'none'}; "
              f"not trained: {c['models_missing']}.", file=sys.stderr)
        for model, problems in c["problems"].items():
            print(f"  {model}: metrics.json does not satisfy the contract: {problems[:2]}", file=sys.stderr)
        print("At least two trained models are needed (see the README for the training commands).", file=sys.stderr)
        return 2

    out = Path(a.out) if a.out else Path(settings.artifacts_dir) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "comparison.md").write_text(to_markdown(c), encoding="utf-8")
    rows = comparison_rows(c)
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Comparison of {', '.join(c['models_found'])} written to {out}/ (comparison.json, .md, .csv).")
    for w_ in c["warnings"]:
        print(f"  WARNING: {w_}")
    for s in c["summary"]:
        sep = {None: "n/a", True: "separated from runner-up", False: "intervals overlap"}[s["separated_from_runner_up"]]
        if len(s["tied_for_best"]) > 1:
            print(f"  {s['target']:<11} test macro-F1: TIE between {', '.join(s['tied_for_best'])} ({s['f1_macro']:.4f})")
        else:
            print(f"  {s['target']:<11} best on test by macro-F1: {s['best_model']:<7} ({s['f1_macro']:.4f}; {sep})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
