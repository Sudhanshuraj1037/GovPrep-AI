"""Dataset statistics, quality checks and class-distribution charts (Phase 10, step 3).

    python -m training.dataset_report            # reads data/processed/, writes data/processed/report/

Everything is computed from the CSVs on disk; nothing is hard-coded. Checks performed:
  * class counts per split, imbalance ratio, classes too small to evaluate (<30 test rows is flagged);
  * text-length statistics;
  * exact-duplicate leakage between splits (after whitespace/case normalisation);
  * NEAR-duplicate leakage: test/val rows whose TF-IDF cosine similarity to a training row is >= 0.9;
  * group leakage: any ``source`` value present in more than one split (when a source column exists
    and the split was made with --group-by source this must be zero).
The result is ``dataset_statistics.json``, ``class_distribution.png`` and a plain verdict list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from app.data import DataNotFoundError, load_split, manifest_path, processed_dir, text_key  # noqa: E402

NEAR_DUP_THRESHOLD = 0.9
SMALL_TEST_CLASS = 30


def near_duplicates(train: pd.DataFrame, other: pd.DataFrame) -> int:
    if train.empty or other.empty:
        return 0
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1).fit(train["text"])
    a, b = vec.transform(other["text"]), vec.transform(train["text"])
    best = np.zeros(a.shape[0])
    for start in range(0, a.shape[0], 500):          # chunked: no dense n x m matrix in memory
        sims = (a[start:start + 500] @ b.T).toarray()
        best[start:start + 500] = sims.max(axis=1)
    return int((best >= NEAR_DUP_THRESHOLD).sum())


def build_report() -> dict:
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = load_split(f"public_{name}")
    try:
        splits["pdf_heldout"] = load_split("pdf_heldout")
    except DataNotFoundError:
        pass

    stats: dict = {"splits": {}, "checks": [], "warnings": []}
    for name, df in splits.items():
        lens = df["text"].str.len()
        stats["splits"][name] = {
            "rows": len(df),
            "subject_counts": {k: int(v) for k, v in df["subject"].value_counts().sort_index().items()},
            "difficulty_counts": {k or "(blank)": int(v) for k, v in df["difficulty"].value_counts().items()},
            "topic_labelled_rows": int((df["topic"] != "").sum()),
            "text_chars": {"min": int(lens.min()), "median": float(lens.median()),
                           "mean": round(float(lens.mean()), 1), "max": int(lens.max())} if len(df) else {},
        }
        if "source" in df.columns:
            stats["splits"][name]["source_counts"] = {k: int(v) for k, v in df["source"].value_counts().items()}

    tr = splits["train"]
    counts = tr["subject"].value_counts()
    stats["imbalance_ratio_train"] = round(float(counts.max() / counts.min()), 2)
    for s in ("val", "test"):
        for cls, n in splits[s]["subject"].value_counts().items():
            if n < SMALL_TEST_CLASS and s == "test":
                stats["warnings"].append(
                    f"class '{cls}' has only {n} test rows (<{SMALL_TEST_CLASS}): its per-class metrics will be unreliable")
    if stats["imbalance_ratio_train"] > 5:
        stats["warnings"].append(
            f"train imbalance ratio {stats['imbalance_ratio_train']}:1 - report macro-F1, not accuracy alone")

    # Leakage must be checked across every pair of partitions, not only train-vs-heldout.
    # In particular, a validation/test duplicate can still contaminate final model selection.
    split_names = list(splits)
    for i, left_name in enumerate(split_names):
        left = splits[left_name]
        left_keys = set(left["text"].map(text_key))
        for right_name in split_names[i + 1:]:
            right = splits[right_name]
            exact = int(right["text"].map(text_key).isin(left_keys).sum())
            near = near_duplicates(left, right)
            stats["checks"].append({
                "check": f"exact duplicates {left_name} vs {right_name}",
                "count": exact,
                "pass": exact == 0,
            })
            stats["checks"].append({
                "check": f"near duplicates (cosine>={NEAR_DUP_THRESHOLD}) {left_name} vs {right_name}",
                "count": near,
                "pass": near == 0,
            })
            if near:
                stats["warnings"].append(
                    f"{near} {right_name} row(s) are near-duplicates of {left_name}: paraphrased copies may inflate scores"
                )

    # `source` is provenance (which corpus/file supplied a row), not automatically
    # a document-level grouping key. A single corpus is expected to appear in all
    # random stratified splits. Enforce disjointness only for the grouping column
    # explicitly recorded in the split manifest.
    split_meta = {}
    if manifest_path().exists():
        try:
            split_meta = json.loads(manifest_path().read_text(encoding="utf-8")).get("split", {})
        except (OSError, ValueError, TypeError):
            split_meta = {}
    group_col = split_meta.get("grouped_by")
    if group_col:
        missing_group = [name for name, frame in splits.items() if group_col not in frame.columns]
        if missing_group:
            stats["checks"].append({
                "check": f"group column '{group_col}' present in all splits",
                "count": len(missing_group),
                "missing_splits": missing_group,
                "pass": False,
            })
        else:
            group_sets = {name: set(frame[group_col].dropna().astype(str))
                          for name, frame in splits.items()}
            shared_groups = set()
            for i, left_name in enumerate(split_names):
                for right_name in split_names[i + 1:]:
                    shared_groups.update(group_sets[left_name] & group_sets[right_name])
            stats["checks"].append({
                "check": f"groups shared across {group_col} partitions",
                "count": len(shared_groups),
                "values": sorted(shared_groups)[:20],
                "pass": len(shared_groups) == 0,
            })

    stats["verdict"] = "PASS" if all(c["pass"] for c in stats["checks"]) else "FAIL: leakage detected"
    return stats


def plot(stats: dict, out: Path) -> None:
    names = [n for n in stats["splits"]]
    subjects = sorted({s for n in names for s in stats["splits"][n]["subject_counts"]})
    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.1), 5))
    w = 0.8 / max(1, len(names))
    x = np.arange(len(subjects))
    for i, n in enumerate(names):
        vals = [stats["splits"][n]["subject_counts"].get(s, 0) for s in subjects]
        ax.bar(x + i * w, vals, w, label=f"{n} (n={stats['splits'][n]['rows']})")
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("questions")
    ax.set_title("Class distribution per split (subject)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    try:
        stats = build_report()
    except DataNotFoundError as exc:
        print(f"ERROR: {exc}\nRun training.prepare_data first.", file=sys.stderr)
        return 2
    out = processed_dir() / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(stats, out / "class_distribution.png")
    print(json.dumps({k: stats[k] for k in ("verdict", "warnings", "checks", "imbalance_ratio_train")},
                     indent=2, ensure_ascii=False))
    print(f"\nWrote {out}", file=sys.stderr)
    return 0 if stats["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
