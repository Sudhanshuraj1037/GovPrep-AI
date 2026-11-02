"""Data loading, validation and reproducible splitting.

Schema (every CSV, public or PDF-derived):  text, subject, topic, difficulty

* ``text`` and ``subject`` are required on every row.
* ``topic`` and ``difficulty`` may be blank; a classifier for that target is
  simply trained on the rows that have a label (or skipped if there are too few).
* ``subject`` must be one of ``labels.SUBJECTS`` and ``difficulty`` one of
  ``labels.DIFFICULTIES`` (matching is case-insensitive). ``topic`` is free text
  until a controlled taxonomy is defined.

Splits are stratified on ``subject``, use a fixed seed (default 42), and are
independent of the row order of the input file.

Nothing in this module reads ``settings`` at import time, so tests (and
``NLP_DATA_DIR``) can redirect the data directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings
from .labels import DATA_COLUMNS, DIFFICULTIES, SUBJECTS

DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


class DataNotFoundError(FileNotFoundError):
    """A required data file does not exist (e.g. no training data yet)."""


class DataValidationError(ValueError):
    """The data exists but is malformed."""


# --------------------------------------------------------------------- paths
def processed_dir() -> Path:
    return Path(settings.data_dir) / "processed"


def processed_path(name: str) -> Path:
    """name in {public_train, public_val, public_test, pdf_heldout}."""
    return processed_dir() / f"{name}.csv"


def manifest_path() -> Path:
    return processed_dir() / "split_manifest.json"


# --------------------------------------------------------------------- utils
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_WS = re.compile(r"\s+")


def text_key(text: str) -> str:
    """Normalised form used for duplicate / leakage detection."""
    return _WS.sub(" ", text).strip().casefold()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------- loading
def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate a CSV with columns text, subject, topic, difficulty.

    Returns ``(frame, report)``. Raises ``DataNotFoundError`` if the file is
    missing and ``DataValidationError`` for schema / label problems.
    """
    path = Path(path)
    if not path.is_file():
        raise DataNotFoundError(f"Data file not found: {path}")

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:  # malformed CSV
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in DATA_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(
            f"{path.name} is missing column(s) {missing}. "
            f"Required schema: {DATA_COLUMNS}"
        )

    df = raw[DATA_COLUMNS].copy()
    for col in DATA_COLUMNS:
        df[col] = df[col].map(lambda v: _WS.sub(" ", v).strip())

    rows_read = len(df)
    empty_text = int((df["text"] == "").sum())
    df = df[df["text"] != ""]
    missing_subject = int((df["subject"] == "").sum())
    df = df[df["subject"] != ""]

    subj_map = {s.casefold(): s for s in SUBJECTS}
    unknown = sorted({v for v in df["subject"] if v.casefold() not in subj_map})
    if unknown:
        raise DataValidationError(
            f"{path.name}: unknown subject label(s) {unknown}. Allowed: {SUBJECTS}. "
            "Map them in your CSV first."
        )
    df["subject"] = df["subject"].map(lambda v: subj_map[v.casefold()])

    diff_map = {d.casefold(): d for d in DIFFICULTIES}
    unknown_d = sorted(
        {v for v in df["difficulty"] if v and v.casefold() not in diff_map}
    )
    if unknown_d:
        raise DataValidationError(
            f"{path.name}: unknown difficulty label(s) {unknown_d}. "
            f"Allowed: {DIFFICULTIES} (or blank)."
        )
    df["difficulty"] = df["difficulty"].map(
        lambda v: diff_map[v.casefold()] if v else ""
    )

    df = df.reset_index(drop=True)
    report = {
        "file": path.name,
        "rows_read": rows_read,
        "dropped_empty_text": empty_text,
        "dropped_missing_subject": missing_subject,
        "rows_kept": len(df),
    }
    return df, report


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicate texts. If copies of one text carry different subjects the
    label is unreliable, so ALL copies are dropped (counted separately)."""
    df = df.copy()
    df["_key"] = df["text"].map(text_key)
    conflict_keys = set(
        df.groupby("_key")["subject"].nunique().loc[lambda s: s > 1].index
    )
    n_conflict_rows = int(df["_key"].isin(conflict_keys).sum())
    df = df[~df["_key"].isin(conflict_keys)]
    before = len(df)
    df = df.drop_duplicates("_key", keep="first")
    stats = {
        "duplicate_rows_dropped": before - len(df),
        "conflicting_label_rows_dropped": n_conflict_rows,
    }
    return df.reset_index(drop=True), stats


# ----------------------------------------------------------------- splitting
def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
    min_class_count: int = 5,
    stratify_on: str = "subject",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Seeded stratified train/val/test split.

    * Classes with fewer than ``min_class_count`` rows cannot be split three ways
      and are dropped (and reported, never silently).
    * Rows are sorted by normalised text first, so the result does not depend on
      the order of the input file.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise DataValidationError("val_size and test_size must be in (0,1) and sum to < 1")

    df = df.copy()
    df["_key"] = df["text"].map(text_key)
    df = df.sort_values("_key", kind="mergesort").reset_index(drop=True)

    counts = df[stratify_on].value_counts()
    dropped = {str(k): int(v) for k, v in counts[counts < min_class_count].items()}
    df = df[~df[stratify_on].isin(dropped)].reset_index(drop=True)
    if df[stratify_on].nunique() < 2:
        raise DataValidationError(
            f"Need at least 2 '{stratify_on}' classes with >= {min_class_count} rows "
            f"each after filtering; got {df[stratify_on].nunique()}. "
            f"Dropped small classes: {dropped or 'none'}"
        )

    try:
        rest, test = train_test_split(
            df, test_size=test_size, stratify=df[stratify_on], random_state=seed
        )
        train, val = train_test_split(
            rest,
            test_size=val_size / (1.0 - test_size),
            stratify=rest[stratify_on],
            random_state=seed,
        )
    except ValueError as exc:
        raise DataValidationError(
            f"Dataset too small to split with val_size={val_size}, test_size={test_size}: {exc}"
        ) from exc

    splits = {
        name: part.drop(columns="_key").sort_index().reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    report = {
        "seed": seed,
        "stratified_on": stratify_on,
        "val_size": val_size,
        "test_size": test_size,
        "min_class_count": min_class_count,
        "dropped_small_classes": dropped,
        "sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": {
            k: {str(c): int(n) for c, n in v[stratify_on].value_counts().sort_index().items()}
            for k, v in splits.items()
        },
    }
    return splits, report


def drop_leaked(
    heldout: pd.DataFrame, public_splits: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove held-out rows whose text also appears in any public split."""
    seen: set[str] = set()
    for part in public_splits.values():
        seen.update(part["text"].map(text_key))
    keys = heldout["text"].map(text_key)
    leaked = keys.isin(seen)
    return heldout[~leaked].reset_index(drop=True), {
        "heldout_rows_removed_as_leaks": int(leaked.sum())
    }


# ------------------------------------------------------------ orchestration
def prepare_splits(
    input_csv: str | Path,
    *,
    heldout_csv: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
    min_class_count: int = 5,
) -> dict[str, Any]:
    """Validate the public CSV, split it, write ``processed/*.csv`` and a manifest.

    The optional PDF-derived held-out CSV is validated, de-duplicated against the
    public data, and written to ``processed/pdf_heldout.csv``. It is never split
    and never used for training.
    """
    public, load_report = load_dataset(input_csv)
    public, dedup_report = deduplicate(public)
    splits, split_report = stratified_split(
        public,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        min_class_count=min_class_count,
    )

    out = processed_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(processed_path(f"public_{name}"), index=False, encoding="utf-8")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "public_input": {
            "file": Path(input_csv).name,
            "sha256": file_sha256(input_csv),
            **load_report,
            **dedup_report,
        },
        "split": split_report,
        "pdf_heldout": None,
    }

    heldout_path = processed_path("pdf_heldout")
    if heldout_csv is not None:
        held, held_load = load_dataset(heldout_csv)
        held, held_dedup = deduplicate(held)
        held, leak = drop_leaked(held, splits)
        held.to_csv(heldout_path, index=False, encoding="utf-8")
        manifest["pdf_heldout"] = {
            "file": Path(heldout_csv).name,
            "sha256": file_sha256(heldout_csv),
            **held_load,
            **held_dedup,
            **leak,
            "rows_written": len(held),
            "subject_counts": {
                str(k): int(v) for k, v in held["subject"].value_counts().sort_index().items()
            },
        }
    elif heldout_path.exists():
        # an old held-out file must not silently survive a re-split of the public data
        heldout_path.unlink()

    _write_json_atomic(manifest_path(), manifest)
    return manifest


def load_split(name: str) -> pd.DataFrame:
    """Load one processed split (public_train / public_val / public_test / pdf_heldout)."""
    path = processed_path(name)
    if not path.is_file():
        raise DataNotFoundError(f"{path} not found")
    df, _ = load_dataset(path)
    return df
