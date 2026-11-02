import json

import pandas as pd
import pytest

from app import data as D
from tests.conftest import make_toy_frame


def _write(path, df):
    df.to_csv(path, index=False)
    return path


def test_missing_file_raises_not_found(env):
    with pytest.raises(D.DataNotFoundError):
        D.load_dataset(env / "nope.csv")


def test_missing_column_rejected(env):
    p = env / "bad.csv"
    pd.DataFrame({"text": ["a"], "subject": ["History"]}).to_csv(p, index=False)
    with pytest.raises(D.DataValidationError, match="missing column"):
        D.load_dataset(p)


def test_unknown_labels_rejected(env):
    df = make_toy_frame(5)
    df.loc[0, "subject"] = "Astrology"
    with pytest.raises(D.DataValidationError, match="Astrology"):
        D.load_dataset(_write(env / "x.csv", df))
    df = make_toy_frame(5)
    df.loc[0, "difficulty"] = "impossible"
    with pytest.raises(D.DataValidationError, match="impossible"):
        D.load_dataset(_write(env / "y.csv", df))


def test_normalisation_and_blank_handling(env):
    df = pd.DataFrame(
        {
            "Text": ["  A  question ", "", "another one", "third"],
            "SUBJECT": ["history", "History", "POLITY", ""],
            "topic": ["", "", "x", "y"],
            "difficulty": ["EASY", "easy", "", "hard"],
        }
    )
    out, rep = D.load_dataset(_write(env / "n.csv", df))
    assert list(out.columns) == ["text", "subject", "topic", "difficulty"]
    assert out["text"].tolist() == ["A question", "another one"]
    assert out["subject"].tolist() == ["History", "Polity"]
    assert out["difficulty"].tolist() == ["easy", ""]
    assert rep["dropped_empty_text"] == 1 and rep["dropped_missing_subject"] == 1


def test_utf8_bom_and_hindi(env):
    p = env / "h.csv"
    p.write_text("text,subject,topic,difficulty\nभारत की राजधानी क्या है,Polity,,easy\n",
                 encoding="utf-8-sig")
    out, _ = D.load_dataset(p)
    assert out.loc[0, "text"].startswith("भारत")


def test_deduplicate_and_conflicts():
    df = pd.DataFrame(
        {
            "text": ["Same text", "same  TEXT", "Other", "Clash", "clash", "Unique"],
            "subject": ["History", "History", "Polity", "History", "Polity", "Science"],
            "topic": [""] * 6,
            "difficulty": [""] * 6,
        }
    )
    out, st = D.deduplicate(df)
    assert sorted(out["text"]) == ["Other", "Same text", "Unique"]
    assert st == {"duplicate_rows_dropped": 1, "conflicting_label_rows_dropped": 2}


def test_split_is_reproducible_and_seed_sensitive():
    df = make_toy_frame(40)
    a, _ = D.stratified_split(df, seed=42)
    b, _ = D.stratified_split(df, seed=42)
    c, _ = D.stratified_split(df, seed=7)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    assert not a["test"]["text"].equals(c["test"]["text"])


def test_split_independent_of_input_row_order():
    df = make_toy_frame(40)
    a, _ = D.stratified_split(df, seed=42)
    b, _ = D.stratified_split(df.sample(frac=1, random_state=3).reset_index(drop=True), seed=42)
    for k in a:
        assert sorted(a[k]["text"]) == sorted(b[k]["text"])


def test_split_disjoint_complete_and_stratified():
    df = make_toy_frame(40)  # 4 classes x 40
    s, rep = D.stratified_split(df, seed=42, val_size=0.15, test_size=0.15)
    sets = {k: set(v["text"]) for k, v in s.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"])
    assert sum(len(v) for v in s.values()) == len(df)
    assert rep["sizes"] == {k: len(v) for k, v in s.items()}
    for name, share in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        counts = s[name]["subject"].value_counts()
        assert counts.nunique() == 1 or counts.max() - counts.min() <= 1
        assert abs(len(s[name]) / len(df) - share) < 0.03


def test_small_classes_dropped_and_reported():
    df = make_toy_frame(30)
    extra = pd.DataFrame(
        {"text": [f"rare {i}" for i in range(3)], "subject": ["Economics"] * 3,
         "topic": [""] * 3, "difficulty": [""] * 3}
    )
    s, rep = D.stratified_split(pd.concat([df, extra]), seed=42, min_class_count=5)
    assert rep["dropped_small_classes"] == {"Economics": 3}
    assert all("Economics" not in set(v["subject"]) for v in s.values())


def test_too_few_classes_rejected():
    df = make_toy_frame(20)
    df = df[df["subject"] == "History"]
    with pytest.raises(D.DataValidationError):
        D.stratified_split(df)


def test_prepare_splits_writes_files_manifest_and_removes_leaks(toy_csv, env):
    held = make_toy_frame(10, seed=5).head(12).copy()
    leaked = pd.read_csv(toy_csv).head(3)
    held = pd.concat([held, leaked], ignore_index=True)
    held_csv = env / "held.csv"
    held.to_csv(held_csv, index=False)

    m = D.prepare_splits(toy_csv, heldout_csv=held_csv, seed=42)
    for name in ("public_train", "public_val", "public_test", "pdf_heldout"):
        assert D.processed_path(name).is_file()
    man = json.loads(D.manifest_path().read_text(encoding="utf-8"))
    assert man["split"]["seed"] == 42
    assert man["public_input"]["sha256"] == D.file_sha256(toy_csv)
    assert m["pdf_heldout"]["heldout_rows_removed_as_leaks"] == 3

    # the held-out set shares no text with any public split
    keys = set()
    for n in ("public_train", "public_val", "public_test"):
        keys |= set(D.load_split(n)["text"].map(D.text_key))
    assert not (set(D.load_split("pdf_heldout")["text"].map(D.text_key)) & keys)


def test_prepare_twice_gives_identical_files(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    first = {n: D.processed_path(n).read_bytes() for n in ("public_train", "public_val", "public_test")}
    D.prepare_splits(toy_csv, seed=42)
    for n, content in first.items():
        assert D.processed_path(n).read_bytes() == content


def test_resplit_without_heldout_removes_stale_heldout(toy_csv, env):
    held_csv = env / "held.csv"
    make_toy_frame(10, seed=5).head(10).to_csv(held_csv, index=False)
    D.prepare_splits(toy_csv, heldout_csv=held_csv)
    assert D.processed_path("pdf_heldout").is_file()
    D.prepare_splits(toy_csv)
    assert not D.processed_path("pdf_heldout").exists()
