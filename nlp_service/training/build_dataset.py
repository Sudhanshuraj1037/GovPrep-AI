"""Turn raw downloaded files into the project's canonical labelled CSV (Phase 10, step 1).

    python -m training.build_dataset --config data/raw/sources.json --out data/public/questions.csv

``sources.json`` lists every raw file you downloaded. It is YOUR record of provenance, so every entry
needs a name, URL and licence that you checked yourself:

    {"sources": [
      {"name": "ExampleQA", "path": "data/raw/example.csv", "format": "csv",
       "url": "https://...", "license": "<as stated by the publisher>",
       "columns": {"text": "question", "subject": "category", "topic": "", "difficulty": ""}}
    ]}

What it does (and what it refuses to do):
* renames columns to text / subject / topic / difficulty and adds ``source`` = the entry's name;
* maps each raw subject through ``labels.SUBJECT_ALIASES``; a raw label with no mapping is COUNTED AND
  DROPPED, never guessed, and listed in the report so you can decide;
* cleans text (HTML tags, control characters, whitespace) and drops rows that are empty or too short;
* never invents labels: if a file has no subject column, its rows are dropped, not labelled by a heuristic.
Exact-duplicate removal, splitting and leakage checks are done afterwards by ``training.prepare_data``.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

from app.labels import DIFFICULTIES, SUBJECT_ALIASES, SUBJECTS

_TAG = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
MIN_CHARS = 10


def clean_text(value: object) -> str:
    s = html.unescape(str(value if value is not None else ""))
    s = _TAG.sub(" ", s)
    s = _CTRL.sub(" ", s)
    return _WS.sub(" ", s).strip()


def map_subject(raw: object) -> str | None:
    key = _WS.sub(" ", str(raw)).strip().casefold()
    for s in SUBJECTS:
        if key == s.casefold():
            return s
    return SUBJECT_ALIASES.get(key)


def read_raw(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if fmt == "json":
        return pd.read_json(path, dtype=False)
    raise ValueError(f"unsupported format '{fmt}' (use csv, tsv, json or jsonl)")


def build(config: dict, base: Path) -> tuple[pd.DataFrame, dict]:
    frames, report = [], {"sources": []}
    for src in config["sources"]:
        for key in ("name", "path", "format", "url", "license", "columns"):
            if not src.get(key) and key != "columns":
                raise SystemExit(f"source entry is missing '{key}': {src}")
        cols = src["columns"]
        if not cols.get("text") or not cols.get("subject"):
            raise SystemExit(f"{src['name']}: 'columns' must name a text column and a subject column")
        raw = read_raw(base / src["path"], src["format"])
        for want in (cols["text"], cols["subject"]):
            if want not in raw.columns:
                raise SystemExit(f"{src['name']}: column '{want}' not found; file has {list(raw.columns)}")
        df = pd.DataFrame({
            "text": raw[cols["text"]].map(clean_text),
            "subject_raw": raw[cols["subject"]],
            "topic": raw[cols["topic"]].map(clean_text) if cols.get("topic") in raw.columns else "",
            "difficulty": raw[cols["difficulty"]] if cols.get("difficulty") in raw.columns else "",
        })
        df["difficulty"] = df["difficulty"].map(lambda v: str(v).strip().casefold()).where(
            lambda s: s.isin(DIFFICULTIES), "")
        df["subject"] = df["subject_raw"].map(map_subject)
        n_read = len(df)
        too_short = df["text"].str.len() < MIN_CHARS
        unmapped = df["subject"].isna() & ~too_short
        entry = {
            "name": src["name"], "url": src["url"], "license": src["license"],
            "rows_read": n_read,
            "dropped_too_short_or_empty": int(too_short.sum()),
            "dropped_unmapped_subject": int(unmapped.sum()),
            "unmapped_subject_labels": {str(k): int(v) for k, v in
                                        df.loc[unmapped, "subject_raw"].value_counts().items()},
        }
        df = df[~too_short & df["subject"].notna()].copy()
        df["source"] = src["name"]
        entry["rows_kept"] = len(df)
        report["sources"].append(entry)
        frames.append(df[["text", "subject", "topic", "difficulty", "source"]])
    if not frames:
        raise SystemExit("no sources listed")
    return pd.concat(frames, ignore_index=True), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg_path = Path(a.config)
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    df, report = build(config, base=Path.cwd())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    report["rows_written"] = len(df)
    rp = out.with_suffix(".build_report.json")
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(df)} rows -> {out}\nBuild report -> {rp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
