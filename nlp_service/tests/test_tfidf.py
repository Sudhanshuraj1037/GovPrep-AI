import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app import data as D
from app import tfidf as T
from app.config import settings
from app.preprocessing import PreprocessConfig
from tests.conftest import make_toy_frame


@pytest.fixture
def prepared(toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    return toy_csv


def _out():
    return T.artifacts_dir()


# ---------------------------------------------------------------- "not trained"
def test_no_data_means_not_trained_and_nothing_written(env):
    with pytest.raises(D.DataNotFoundError):
        T.train_all()
    assert not _out().exists() or not any(_out().iterdir())
    with pytest.raises(T.ModelNotTrainedError, match="not trained yet"):
        T.TfidfClassifier.load()


def test_cli_reports_not_trained_yet(env, capsys):
    from training.train_tfidf import main
    code = main([])
    err = capsys.readouterr().err
    assert code == 2 and "NOT TRAINED YET" in err
    assert not _out().exists() or not any(_out().iterdir())


def test_nothing_trainable_writes_nothing(prepared):
    toy = pd.read_csv(D.processed_path("public_train"), dtype=str, keep_default_na=False)
    assert (toy["topic"] != "").any()
    for n in ("public_train", "public_val", "public_test"):
        df = pd.read_csv(D.processed_path(n), dtype=str, keep_default_na=False)
        df["topic"] = ""
        df.to_csv(D.processed_path(n), index=False)
    with pytest.raises(T.NothingTrainedError):
        T.train_all(targets=("topic",))
    assert not _out().exists() or not any(_out().iterdir())


# ------------------------------------------------------------------- training
def test_training_writes_artifacts_with_marker(prepared):
    m = T.train_all(seed=42)
    assert (_out() / "metrics.json").is_file()
    for t in ("subject", "topic", "difficulty"):
        assert (_out() / f"{t}.joblib").is_file()
    saved = json.loads((_out() / "metrics.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 42 and saved["model"] == "tfidf"
    assert saved["preprocessing"] == PreprocessConfig().to_dict()
    assert saved["data"]["pdf_heldout_evaluated"] is False
    assert saved["targets"]["subject"]["status"] == "trained"
    assert set(saved["targets"]["subject"]["splits"]) == {"val", "test"}
    assert m["targets"]["subject"]["labels"] == sorted(m["targets"]["subject"]["labels"])


def test_reported_metrics_match_independent_recomputation(prepared):
    """The numbers in metrics.json must equal what you get by reloading the saved
    model and scoring the saved test CSV yourself."""
    T.train_all(seed=42)
    saved = json.loads((_out() / "metrics.json").read_text(encoding="utf-8"))
    test = D.load_split("public_test")
    train = D.load_split("public_train")

    clf = T.TfidfClassifier.load()
    preds = clf.predict(test["text"].tolist(), top_k=1)
    y_pred = [p["predictions"]["subject"]["label"] for p in preds]
    y_true = test["subject"].tolist()
    reported = saved["targets"]["subject"]["splits"]["test"]

    assert reported["n"] == len(test)
    assert reported["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    labels = saved["targets"]["subject"]["labels"]
    assert reported["f1_macro"] == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    cm = np.array(reported["confusion_matrix"]["matrix"])
    assert reported["confusion_matrix"]["labels"] == labels
    assert (cm == confusion_matrix(y_true, y_pred, labels=labels)).all()
    assert cm.sum() == len(test)
    for i, lab in enumerate(labels):
        assert cm[i].sum() == reported["per_class"][lab]["support"]
    # majority baseline is computed from TRAIN labels
    maj = train["subject"].value_counts().idxmax()
    assert reported["majority_class_baseline"]["class"] == maj


def test_training_is_reproducible(prepared):
    a = T.train_all(seed=42)
    b = T.train_all(seed=42)
    for t in ("subject", "difficulty"):
        assert a["targets"][t]["chosen_C"] == b["targets"][t]["chosen_C"]
        for split in ("val", "test"):
            assert a["targets"][t]["splits"][split]["accuracy"] == b["targets"][t]["splits"][split]["accuracy"]
            assert a["targets"][t]["splits"][split]["confusion_matrix"] == b["targets"][t]["splits"][split]["confusion_matrix"]


def test_c_is_chosen_on_validation_and_grid_logged(prepared):
    m = T.train_all(cfg=T.TfidfConfig(c_grid=(0.01, 1.0, 100.0)))
    e = m["targets"]["subject"]
    assert e["C_selected_on"] == "validation macro-F1"
    assert [g["C"] for g in e["C_grid"]] == [0.01, 1.0, 100.0]
    best = max(e["C_grid"], key=lambda g: (g["val_f1_macro"], -g["C"]))
    assert e["chosen_C"] == best["C"]


def test_pdf_heldout_is_evaluated_only_when_present_and_not_trained_on(prepared, env):
    held = make_toy_frame(8, seed=11)
    held_csv = env / "held.csv"
    held.to_csv(held_csv, index=False)
    D.prepare_splits(prepared, heldout_csv=held_csv, seed=42)
    m = T.train_all()
    assert m["data"]["pdf_heldout_evaluated"] is True
    n_held = len(D.load_split("pdf_heldout"))
    assert m["targets"]["subject"]["splits"]["pdf_heldout"]["n"] == n_held
    # held-out rows never reach the fitted vocabulary
    assert m["data"]["rows"]["train"] == len(D.load_split("public_train"))


def test_unseen_heldout_label_counted_as_error(prepared, env):
    held = make_toy_frame(6, seed=12)
    held.loc[:2, "subject"] = "Economics"  # class absent from training data
    held_csv = env / "held.csv"
    held.to_csv(held_csv, index=False)
    D.prepare_splits(prepared, heldout_csv=held_csv, seed=42)
    m = T.train_all(targets=("subject",))
    r = m["targets"]["subject"]["splits"]["pdf_heldout"]
    assert r["unseen_label_rows"] == 3
    assert "Economics" in r["confusion_matrix"]["labels"]
    assert r["per_class"]["Economics"]["recall"] == 0.0


def test_target_without_labels_is_skipped_with_reason(env):
    path = env / "data" / "public" / "t.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path)
    m = T.train_all()
    assert m["targets"]["topic"]["status"] == "skipped"
    assert "needs >= 2 classes" in m["targets"]["topic"]["reason"]
    assert m["targets"]["subject"]["status"] == "trained"
    assert not (_out() / "topic.joblib").exists()


def test_stale_models_removed_on_retrain(prepared):
    T.train_all()
    assert (_out() / "topic.joblib").exists()
    T.train_all(targets=("subject",))
    assert (_out() / "subject.joblib").exists()
    assert not (_out() / "topic.joblib").exists()


def test_marker_is_written_last(prepared, monkeypatch):
    """If saving metrics fails, /health must not show the model as trained."""
    from app.registry import model_status

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(T, "_write_json_atomic", boom)
    with pytest.raises(OSError):
        T.train_all()
    assert (_out() / "subject.joblib").exists()
    assert not (_out() / "metrics.json").exists()
    tfidf = next(m for m in model_status() if m["name"] == "tfidf")
    assert tfidf["trained"] is False


def test_top_terms_are_real_vocabulary(prepared):
    m = T.train_all()
    terms = m["targets"]["subject"]["top_terms"]
    assert set(terms) == set(m["targets"]["subject"]["labels"])
    history = {t["term"] for t in terms["History"]}
    assert history & {"empir", "dynasti", "emperor", "battl", "ancient", "kingdom", "mediev", "treati", "coloni", "revolt"}


# ------------------------------------------------------------------ inference
def test_inference_shapes_and_probabilities(prepared):
    T.train_all()
    clf = T.TfidfClassifier.load()
    assert clf.targets == ["difficulty", "subject", "topic"]
    out = clf.predict(["the river delta and monsoon climate", "ancient empire battle"], top_k=10)
    s0 = out[0]["predictions"]["subject"]
    assert s0["label"] == "Geography"
    assert out[1]["predictions"]["subject"]["label"] == "History"
    assert sum(t["probability"] for t in s0["top"]) == pytest.approx(1.0)
    assert 0 <= s0["confidence"] <= 1
    assert s0["top"][0]["label"] == s0["label"]


def test_inference_no_usable_tokens_returns_none(prepared):
    T.train_all()
    out = T.TfidfClassifier.load().predict(["the of and", "   ", "river delta"])
    assert out[0]["predictions"]["subject"]["label"] is None
    assert "warning" in out[0]["predictions"]["subject"]
    assert out[1]["predictions"]["subject"]["label"] is None
    assert out[2]["predictions"]["subject"]["label"] == "Geography"


def test_model_file_missing_is_a_clear_error(prepared):
    T.train_all()
    (_out() / "subject.joblib").unlink()
    with pytest.raises(T.ModelNotTrainedError, match="missing"):
        T.TfidfClassifier.load()


def test_cli_prepare_then_train(toy_csv, env, capsys):
    from training.prepare_data import main as prep
    from training.train_tfidf import main as train
    assert prep(["--input", str(toy_csv)]) == 0
    capsys.readouterr()
    assert train(["--targets", "subject"]) == 0
    out = capsys.readouterr().out
    assert "TF-IDF training finished" in out and "macro-F1" in out
    assert (_out() / "metrics.json").is_file()


def test_fitting_and_tuning_never_see_test_or_heldout(prepared):
    """Replace the test and held-out data with different data: the fitted model,
    vocabulary and chosen C must be bit-identical (only the reported test
    metrics may change). Guards against test-set leakage into training or tuning."""
    from app.preprocessing import TextPreprocessor

    splits = {n: D.load_split(f"public_{n}") for n in ("train", "val", "test")}
    splits["pdf_heldout"] = make_toy_frame(8, seed=11)
    pre, cfg = TextPreprocessor(), T.TfidfConfig()
    b1, e1 = T.train_target("subject", splits, pre, cfg, 42)

    test = splits["test"]
    altered = dict(splits)
    altered["test"] = test.assign(
        subject=test["subject"].sample(frac=1, random_state=1).values,
        text=test["text"].str[::-1],
    )
    altered["pdf_heldout"] = make_toy_frame(8, seed=99)
    b2, e2 = T.train_target("subject", altered, pre, cfg, 42)

    assert e1["chosen_C"] == e2["chosen_C"]
    assert e1["C_grid"] == e2["C_grid"]
    assert e1["vocabulary_size"] == e2["vocabulary_size"]
    assert b1["vectorizer"].vocabulary_ == b2["vectorizer"].vocabulary_
    assert np.array_equal(b1["classifier"].coef_, b2["classifier"].coef_)
    # ...while the reported test numbers do react to the changed test data
    assert e1["splits"]["test"]["accuracy"] != e2["splits"]["test"]["accuracy"]
