"""Tests for the model-agnostic metrics contract (app/model_metrics.py) that the comparison
dashboard will use to put TF-IDF, LSTM, BiLSTM and BERT side by side.

The contract / tampering tests need no PyTorch. The cross-model tests (LSTM + BiLSTM) use the
session-trained tiny models and are skipped when PyTorch is not installed. Synthetic data only.
"""
import copy
import json
import math
import shutil
from pathlib import Path

import pytest

from app import data as D
from app import model_metrics as MM
from app import tfidf as T
from app.config import settings
from tests.conftest import make_toy_frame


@pytest.fixture
def tfidf_metrics(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    return T.train_all(seed=42)


def _sub(m):
    """The 'subject' / 'test' metric block of a metrics dict."""
    return m["targets"]["subject"]["splits"]["test"]


# ============================================================ validator
def test_real_tfidf_metrics_conform(tfidf_metrics):
    assert MM.validate_metrics(tfidf_metrics) == []
    on_disk = json.loads((T.artifacts_dir() / "metrics.json").read_text(encoding="utf-8"))
    assert MM.validate_metrics(on_disk) == []


def test_heldout_split_conforms_too(env, toy_csv):
    held = env / "held.csv"
    make_toy_frame(8, seed=11).to_csv(held, index=False)
    D.prepare_splits(toy_csv, heldout_csv=held, seed=42)
    m = T.train_all()
    assert "pdf_heldout" in m["targets"]["subject"]["splits"] and MM.validate_metrics(m) == []


def _tamper(tfidf_metrics, edit):
    m = copy.deepcopy(tfidf_metrics)
    edit(m)
    return MM.validate_metrics(m)


def _set(path, value):
    def edit(m):
        node = m
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return edit


def _delete(path):
    def edit(m):
        node = m
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return edit


T0 = ("targets", "subject", "splits", "test")


@pytest.mark.parametrize("edit,expected", [
    (_delete(("trained_at",)), "trained_at"),
    (_set(("seed",), "42"), "seed"),
    (_set(("model",), ""), "model"),
    (_delete(("data", "rows")), "data.rows"),
    (_set(("data", "pdf_heldout_evaluated"), 1), "pdf_heldout_evaluated"),
    (_set(("targets",), {}), "targets"),
    (_set(("targets", "subject", "status"), "weird"), "status"),
    (_set(("targets", "subject", "labels"), ["b", "a", "c", "d"]), "sorted"),
    (_set(("targets", "subject", "labels"), ["only"]), "at least 2"),
    (_set(("targets", "subject", "n_train"), 0), "n_train"),
    (_set(("targets", "subject", "splits"), {}), "splits"),
    (_set(("targets", "subject", "splits"), {"train": {}}), "splits"),
    (_delete(T0 + ("f1_macro",)), "f1_macro"),
    (_delete(T0 + ("f1_weighted",)), "f1_weighted"),
    (_set(T0 + ("accuracy",), 1.5), "accuracy"),
    (_set(T0 + ("recall_macro",), float("nan")), "recall_macro"),
    (_set(T0 + ("precision_weighted",), "0.9"), "precision_weighted"),
    (_set(T0 + ("n",), True), ".n"),
    (_set(T0 + ("n",), 0), ".n"),
    (_set(T0 + ("unseen_label_rows",), -1), "unseen_label_rows"),
    (_delete(T0 + ("majority_class_baseline",)), "majority_class_baseline"),
    (_delete(T0 + ("confusion_matrix",)), "confusion_matrix"),
])
def test_structural_violations_are_detected(tfidf_metrics, edit, expected):
    bad = _tamper(tfidf_metrics, edit)
    assert bad and any(expected in b for b in bad), bad


def test_consistency_violations_are_detected(tfidf_metrics):
    def accuracy_not_trace(m):                       # a plausible but inconsistent accuracy
        _sub(m)["accuracy"] = min(1.0, _sub(m)["accuracy"] + 0.1) if _sub(m)["accuracy"] < 0.9 else 0.5

    def matrix_total(m):
        _sub(m)["confusion_matrix"]["matrix"][0][0] += 1

    def matrix_not_square(m):
        _sub(m)["confusion_matrix"]["matrix"][0].append(0)

    def matrix_negative(m):
        _sub(m)["confusion_matrix"]["matrix"][0][0] = -1

    def support(m):
        first = next(iter(_sub(m)["per_class"]))
        _sub(m)["per_class"][first]["support"] += 1

    def per_class_order(m):
        pc = _sub(m)["per_class"]
        _sub(m)["per_class"] = dict(reversed(list(pc.items())))

    def labels_prefix(m):
        _sub(m)["confusion_matrix"]["labels"] = list(reversed(_sub(m)["confusion_matrix"]["labels"]))

    for edit, expected in [
        (accuracy_not_trace, "trace / n"), (matrix_total, "sum to"), (matrix_not_square, "x"),
        (matrix_negative, "non-negative"), (support, "support"), (per_class_order, "per_class"),
        (labels_prefix, "start with the target labels"),
    ]:
        bad = _tamper(tfidf_metrics, edit)
        assert bad and any(expected in b for b in bad), (expected, bad)


def test_skipped_target_needs_a_reason_and_non_objects_are_rejected(tfidf_metrics):
    bad = _tamper(tfidf_metrics, _set(("targets", "topic"), {"status": "skipped"}))
    assert any("must say why" in b for b in bad)
    assert MM.validate_metrics(_tamper.__name__) and MM.validate_metrics([]) and MM.validate_metrics(None)


def test_skipped_target_with_reason_conforms(tfidf_metrics):
    m = copy.deepcopy(tfidf_metrics)
    m["targets"]["topic"] = {"status": "skipped", "reason": "no labels"}
    assert MM.validate_metrics(m) == []
    assert not [r for r in MM.rows_from_metrics(m) if r["target"] == "topic"]


# ============================================================ rows
def test_rows_have_the_same_fields_and_the_real_values(tfidf_metrics):
    rows = MM.rows_from_metrics(tfidf_metrics, checkpoint_available=True)
    assert len(rows) == 3 * 2                                           # 3 targets x (val, test)
    assert all(tuple(r) == MM.ROW_FIELDS for r in rows)
    assert [r["target"] for r in rows] == ["subject"] * 2 + ["topic"] * 2 + ["difficulty"] * 2
    assert [r["split"] for r in rows[:2]] == ["val", "test"]
    r = next(x for x in rows if x["target"] == "subject" and x["split"] == "test")
    src = _sub(tfidf_metrics)
    assert r["n"] == src["n"] and r["accuracy"] == src["accuracy"] and r["f1_macro"] == src["f1_macro"]
    assert r["majority_class_accuracy"] == src["majority_class_baseline"]["accuracy"]
    assert r["n_classes"] == len(tfidf_metrics["targets"]["subject"]["labels"])
    assert r["model"] == "tfidf" and r["seed"] == 42 and r["checkpoint_available"] is True
    assert r["trainable_parameters"] is None                           # not a notion for TF-IDF


def test_rows_never_invent_numbers(tfidf_metrics):
    rows = MM.rows_from_metrics(tfidf_metrics)
    for r in rows:
        src = tfidf_metrics["targets"][r["target"]]["splits"][r["split"]]
        for f in MM.SCORE_FIELDS:
            assert r[f] == src[f] and not math.isnan(r[f])


# ============================================================ collect()
def test_collect_with_nothing_trained(env):
    res = MM.collect()
    assert res == {"models_found": [], "models_missing": list(MM.MODELS), "problems": {}, "rows": []}


def test_collect_reads_the_real_file(tfidf_metrics):
    res = MM.collect()
    assert res["models_found"] == ["tfidf"] and res["models_missing"] == ["lstm", "bilstm", "bert"]
    assert res["problems"] == {} and len(res["rows"]) == 6
    assert {r["checkpoint_available"] for r in res["rows"]} == {True}
    assert MM.collect(settings.artifacts_dir) == res                   # explicit directory gives the same


def test_a_broken_file_is_reported_not_turned_into_numbers(tfidf_metrics):
    f = T.artifacts_dir() / "metrics.json"
    m = json.loads(f.read_text(encoding="utf-8"))
    m["targets"]["subject"]["splits"]["test"]["accuracy"] = 7
    f.write_text(json.dumps(m), encoding="utf-8")
    res = MM.collect()
    assert res["rows"] == [] and "tfidf" in res["problems"] and res["models_found"] == []
    f.write_text("{ not json", encoding="utf-8")
    assert "could not read" in MM.collect()["problems"]["tfidf"][0]


def test_a_file_in_the_wrong_folder_is_rejected(tfidf_metrics):
    d = Path(settings.artifacts_dir)
    shutil.copytree(d / "tfidf", d / "bert")                            # TF-IDF metrics filed under bert/
    res = MM.collect()
    assert "bert" in res["problems"] and "says 'tfidf'" in res["problems"]["bert"][0]
    assert res["models_found"] == ["tfidf"] and all(r["model"] == "tfidf" for r in res["rows"])


def test_a_new_model_plugs_in_by_conforming_to_the_contract(tfidf_metrics):
    """BERT (Phase 6) only has to write a metrics.json that satisfies the contract."""
    d = Path(settings.artifacts_dir)
    m = json.loads((d / "tfidf" / "metrics.json").read_text(encoding="utf-8"))
    m["model"] = "bert"                                                 # test fixture: a renamed copy
    (d / "bert").mkdir()
    (d / "bert" / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    res = MM.collect()
    assert res["models_found"] == ["tfidf", "bert"] and res["problems"] == {}
    assert {r["model"] for r in res["rows"]} == {"tfidf", "bert"}
    assert next(r for r in res["rows"] if r["model"] == "bert")["checkpoint_available"] is False   # no weights


# ============================================================ cross-model (LSTM + BiLSTM + TF-IDF)
def _load(name):
    return json.loads((Path(settings.artifacts_dir) / name / "metrics.json").read_text(encoding="utf-8"))


def test_all_three_models_conform(all_models_trained):
    for name in ("tfidf", "lstm", "bilstm"):
        assert MM.validate_metrics(_load(name)) == [], name


def test_collect_returns_identical_fields_for_every_model(all_models_trained):
    res = MM.collect()
    assert res["models_found"] == ["tfidf", "lstm", "bilstm"] and res["models_missing"] == ["bert"]
    assert res["problems"] == {}
    assert len(res["rows"]) == 3 * 3 * 2                                # 3 models x 3 targets x (val, test)
    assert {tuple(r) for r in res["rows"]} == {MM.ROW_FIELDS}
    for m in ("tfidf", "lstm", "bilstm"):
        assert [(r["target"], r["split"]) for r in res["rows"] if r["model"] == m] == \
            [(t, s) for t in ("subject", "topic", "difficulty") for s in ("val", "test")]


def test_models_are_scored_on_exactly_the_same_rows_and_classes(all_models_trained):
    rows = {(r["model"], r["target"], r["split"]): r for r in MM.collect()["rows"]}
    raw = {n: _load(n) for n in ("tfidf", "lstm", "bilstm")}
    for target in ("subject", "topic", "difficulty"):
        for split in ("val", "test"):
            trio = [rows[(m, target, split)] for m in ("tfidf", "lstm", "bilstm")]
            for field in ("n", "n_classes", "n_train", "majority_class_accuracy", "seed"):
                assert len({r[field] for r in trio}) == 1, (target, split, field)
            labels = {json.dumps(raw[m]["targets"][target]["splits"][split]["confusion_matrix"]["labels"]) for m in raw}
            assert len(labels) == 1                                    # same class order -> matrices comparable
            assert len({json.dumps(raw[m]["targets"][target]["labels"]) for m in raw}) == 1


def test_the_recurrent_models_were_trained_on_the_same_split_files(all_models_trained):
    lm, bm = _load("lstm"), _load("bilstm")
    assert lm["data"]["split_files_sha256"] == bm["data"]["split_files_sha256"]
    assert bm["data"]["split_files_sha256"]["public_train"] == D.file_sha256(D.processed_path("public_train"))
    assert lm["data"]["split_manifest"]["split"]["seed"] == bm["data"]["split_manifest"]["split"]["seed"] == 42


def test_parameter_counts_are_available_for_the_neural_models_only(all_models_trained):
    rows = {(r["model"], r["target"]): r["trainable_parameters"] for r in MM.collect()["rows"] if r["split"] == "test"}
    for target in ("subject", "topic", "difficulty"):
        assert rows[("tfidf", target)] is None
        assert isinstance(rows[("lstm", target)], int) and rows[("bilstm", target)] > rows[("lstm", target)] > 0


def test_rows_equal_the_raw_metrics_files(all_models_trained):
    for r in MM.collect()["rows"]:
        src = _load(r["model"])["targets"][r["target"]]["splits"][r["split"]]
        assert all(r[f] == src[f] for f in MM.SCORE_FIELDS) and r["n"] == src["n"]


def test_checkpoint_availability_follows_the_disk_but_the_real_metrics_stay(all_models_trained):
    for f in (Path(settings.artifacts_dir) / "bilstm").glob("*.pt"):
        f.unlink()
    rows = MM.collect()["rows"]
    by_model = {m: {r["checkpoint_available"] for r in rows if r["model"] == m} for m in ("tfidf", "lstm", "bilstm")}
    assert by_model == {"tfidf": {True}, "lstm": {True}, "bilstm": {False}}
    assert any(r["model"] == "bilstm" for r in rows)                   # past run's metrics are still real


def test_one_broken_model_does_not_hide_the_others(all_models_trained):
    f = Path(settings.artifacts_dir) / "lstm" / "metrics.json"
    m = json.loads(f.read_text(encoding="utf-8"))
    m["targets"]["subject"]["splits"]["val"]["confusion_matrix"]["matrix"][0][0] += 5
    f.write_text(json.dumps(m), encoding="utf-8")
    res = MM.collect()
    assert list(res["problems"]) == ["lstm"] and res["models_found"] == ["tfidf", "bilstm"]
    assert {r["model"] for r in res["rows"]} == {"tfidf", "bilstm"}


# ============================================================ all four models (Phase 7 comparison)
ALL = ("tfidf", "lstm", "bilstm", "bert")


def test_all_four_models_conform(four_models_trained):
    for name in ALL:
        assert MM.validate_metrics(_load(name)) == [], name


def test_collect_returns_identical_fields_for_all_four_models(four_models_trained):
    res = MM.collect()
    assert res["models_found"] == list(ALL) and res["models_missing"] == [] and res["problems"] == {}
    assert len(res["rows"]) == 4 * 3 * 2                                  # 4 models x 3 targets x (val, test)
    assert {tuple(r) for r in res["rows"]} == {MM.ROW_FIELDS}
    for m in ALL:
        assert [(r["target"], r["split"]) for r in res["rows"] if r["model"] == m] == \
            [(t, s) for t in ("subject", "topic", "difficulty") for s in ("val", "test")]


def test_all_four_models_are_scored_on_exactly_the_same_rows_and_classes(four_models_trained):
    rows = {(r["model"], r["target"], r["split"]): r for r in MM.collect()["rows"]}
    raw = {n: _load(n) for n in ALL}
    for target in ("subject", "topic", "difficulty"):
        for split in ("val", "test"):
            group = [rows[(m, target, split)] for m in ALL]
            for field in ("n", "n_classes", "n_train", "majority_class_accuracy", "seed"):
                assert len({r[field] for r in group}) == 1, (target, split, field)
            assert len({json.dumps(raw[m]["targets"][target]["splits"][split]["confusion_matrix"]["labels"]) for m in ALL}) == 1
            assert len({json.dumps(raw[m]["targets"][target]["labels"]) for m in ALL}) == 1


def test_bert_was_trained_on_the_same_split_files_and_seed_as_the_recurrent_models(four_models_trained):
    bert, lstm, bilstm = _load("bert"), _load("lstm"), _load("bilstm")
    assert bert["data"]["split_files_sha256"] == lstm["data"]["split_files_sha256"] == bilstm["data"]["split_files_sha256"]
    assert bert["data"]["split_files_sha256"]["public_train"] == D.file_sha256(D.processed_path("public_train"))
    assert bert["seed"] == lstm["seed"] == bilstm["seed"] == 42
    assert bert["data"]["split_manifest"]["split"]["seed"] == 42
    assert bert["data"]["rows"] == lstm["data"]["rows"]


def test_parameter_counts_for_every_model(four_models_trained):
    rows = {(r["model"], r["target"]): r["trainable_parameters"] for r in MM.collect()["rows"] if r["split"] == "test"}
    for target in ("subject", "topic", "difficulty"):
        assert rows[("tfidf", target)] is None
        assert all(isinstance(rows[(m, target)], int) and rows[(m, target)] > 0 for m in ("lstm", "bilstm", "bert"))


def test_bert_rows_equal_the_raw_metrics_file(four_models_trained):
    for r in (x for x in MM.collect()["rows"] if x["model"] == "bert"):
        src = _load("bert")["targets"][r["target"]]["splits"][r["split"]]
        assert all(r[f] == src[f] for f in MM.SCORE_FIELDS) and r["n"] == src["n"]


def test_bert_checkpoint_availability_follows_the_disk_but_real_metrics_stay(four_models_trained):
    for f in (Path(settings.artifacts_dir) / "bert").glob("*.pt"):
        f.unlink()
    rows = MM.collect()["rows"]
    by_model = {m: {r["checkpoint_available"] for r in rows if r["model"] == m} for m in ALL}
    assert by_model == {"tfidf": {True}, "lstm": {True}, "bilstm": {True}, "bert": {False}}
    assert any(r["model"] == "bert" for r in rows)


def test_a_broken_bert_file_does_not_hide_the_other_three(four_models_trained):
    f = Path(settings.artifacts_dir) / "bert" / "metrics.json"
    m = json.loads(f.read_text(encoding="utf-8"))
    m["targets"]["subject"]["splits"]["test"]["accuracy"] = 7
    f.write_text(json.dumps(m), encoding="utf-8")
    res = MM.collect()
    assert list(res["problems"]) == ["bert"] and res["models_found"] == ["tfidf", "lstm", "bilstm"]
    assert {r["model"] for r in res["rows"]} == {"tfidf", "lstm", "bilstm"}
