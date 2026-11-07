"""Phase 10 tooling: raw -> canonical CSV, source-grouped split, quality report.

All data here is SYNTHETIC and lives in temp folders; it only exercises the code.
"""
import json

import pandas as pd
import pytest

from app import data as D
from tests.conftest import make_toy_frame
from training import build_dataset as B
from training import dataset_report as R


def test_alias_mapping_and_unmapped_are_reported_not_guessed():
    assert B.map_subject("Indian Polity") == "Polity"
    assert B.map_subject("  general   science ") == "Science"
    assert B.map_subject("History") == "History"
    assert B.map_subject("Computer Knowledge") is None      # no such app label: must NOT be guessed
    assert B.map_subject("Astrology") is None


def test_clean_text_strips_html_entities_control_chars():
    assert B.clean_text("<p>Who  wrote&nbsp;it?\x00</p>") == "Who wrote it?"


def test_build_maps_drops_and_records_provenance(env):
    raw = pd.DataFrame({
        "q": ["Who founded the Maurya empire in ancient India?", "Which article deals with the President?",
              "What is the speed of light in vacuum?", "How many bytes are in a kilobyte?", "short"],
        "cat": ["Indian History", "Indian Polity", "General Science", "Computer Knowledge", "History"],
    })
    (env / "raw.csv").write_text(raw.to_csv(index=False), encoding="utf-8")
    cfg = {"sources": [{"name": "SynthSrc", "path": "raw.csv", "format": "csv", "url": "https://example.invalid",
                        "license": "test-only", "columns": {"text": "q", "subject": "cat"}}]}
    df, rep = B.build(cfg, base=env)
    assert list(df.columns) == ["text", "subject", "topic", "difficulty", "source"]
    assert sorted(df["subject"]) == ["History", "Polity", "Science"]
    assert set(df["source"]) == {"SynthSrc"}
    e = rep["sources"][0]
    assert e["dropped_unmapped_subject"] == 1 and e["unmapped_subject_labels"] == {"Computer Knowledge": 1}
    assert e["dropped_too_short_or_empty"] == 1 and e["rows_kept"] == 3


def test_build_requires_provenance_fields(env):
    (env / "r.csv").write_text("q,c\nsome long enough question text,History\n")
    cfg = {"sources": [{"name": "X", "path": "r.csv", "format": "csv", "url": "", "license": "",
                        "columns": {"text": "q", "subject": "c"}}]}
    with pytest.raises(SystemExit):
        B.build(cfg, base=env)


def _multi_source_frame(n_sources=12, per=20):
    base = make_toy_frame(per * n_sources // 4 + 1)
    base = base.iloc[: n_sources * per].copy()
    base["source"] = [f"doc{i % n_sources}" for i in range(len(base))]
    return base


def test_grouped_split_never_straddles_a_source(env):
    df = _multi_source_frame()
    p = env / "q.csv"; df.to_csv(p, index=False)
    m = D.prepare_splits(p, group_col="source", seed=42)
    parts = {n: D.load_split(f"public_{n}") for n in ("train", "val", "test")}
    sets = {n: set(d["source"]) for n, d in parts.items()}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    assert m["split"]["grouped_by"] == "source"
    assert sum(len(d) for d in parts.values()) == m["split"]["sizes"]["train"] + m["split"]["sizes"]["val"] + m["split"]["sizes"]["test"]


def test_grouped_split_refuses_when_too_few_groups(env):
    df = make_toy_frame(30); df["source"] = "only_one_pdf"
    p = env / "q.csv"; df.to_csv(p, index=False)
    with pytest.raises(D.DataValidationError, match="distinct 'source'"):
        D.prepare_splits(p, group_col="source")


def test_grouped_split_is_deterministic(env):
    df = _multi_source_frame()
    p = env / "q.csv"; df.to_csv(p, index=False)
    D.prepare_splits(p, group_col="source", seed=7)
    a = D.load_split("public_test")["text"].tolist()
    D.prepare_splits(p, group_col="source", seed=7)
    assert a == D.load_split("public_test")["text"].tolist()


def test_default_split_unchanged_without_source_column(env):
    p = env / "q.csv"; make_toy_frame(30).to_csv(p, index=False)
    D.prepare_splits(p)
    assert list(D.load_split("public_train").columns) == ["text", "subject", "topic", "difficulty"]


def test_report_passes_on_clean_split_and_writes_chart(env):
    p = env / "q.csv"; make_toy_frame(40).to_csv(p, index=False)
    D.prepare_splits(p)
    stats = R.build_report()
    assert stats["verdict"] == "PASS"
    R.plot(stats, env / "c.png")
    assert (env / "c.png").stat().st_size > 1000


def test_report_detects_exact_and_near_duplicate_leakage(env):
    p = env / "q.csv"; make_toy_frame(40).to_csv(p, index=False)
    D.prepare_splits(p)
    tr = D.load_split("public_train")
    te = D.load_split("public_test")
    leaked = te.copy()
    leaked.loc[0, "text"] = tr.loc[0, "text"]                       # exact copy
    leaked.loc[1, "text"] = tr.loc[1, "text"] + " ?"                 # near copy
    leaked.to_csv(D.processed_path("public_test"), index=False)
    stats = R.build_report()
    assert stats["verdict"].startswith("FAIL")
    counts = {c["check"]: c["count"] for c in stats["checks"]}
    assert counts["exact duplicates train vs test"] >= 1
    assert counts["near duplicates (cosine>=0.9) train vs test"] >= 2


def test_provenance_source_can_appear_in_all_random_splits(env):
    # `source` identifies the originating dataset, not an individual document.
    df = make_toy_frame(60)
    df["source"] = "one-large-corpus"
    p = env / "q.csv"; df.to_csv(p, index=False)
    D.prepare_splits(p)
    stats = R.build_report()
    assert stats["verdict"] == "PASS"
    assert not any("sources shared" in c["check"] for c in stats["checks"])


def test_report_checks_validation_vs_test_duplicates(env):
    p = env / "q.csv"; make_toy_frame(60).to_csv(p, index=False)
    D.prepare_splits(p)
    val = D.load_split("public_val")
    test = D.load_split("public_test")
    test.loc[0, "text"] = val.loc[0, "text"]
    test.to_csv(D.processed_path("public_test"), index=False)
    stats = R.build_report()
    assert stats["verdict"].startswith("FAIL")
    checks = {c["check"]: c for c in stats["checks"]}
    assert checks["exact duplicates val vs test"]["count"] >= 1


def test_grouped_split_report_enforces_disjoint_group_column(env):
    df = _multi_source_frame()
    p = env / "q.csv"; df.to_csv(p, index=False)
    D.prepare_splits(p, group_col="source")
    stats = R.build_report()
    checks = {c["check"]: c for c in stats["checks"]}
    assert checks["groups shared across source partitions"]["pass"] is True
