"""Phase 7 tests: the TF-IDF vs LSTM vs BiLSTM vs BERT comparison (app/comparison.py).

Most tests need no PyTorch: they use metrics files built by the real shared metrics function
from synthetic predictions (tests/metrics_factory.py), so exact scenarios (clear winner, tie,
mismatched rows, broken file...) can be constructed. The final section uses the real
session-trained tiny models. Nothing here is a result about a real model.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app import comparison as C
from app import model_metrics as MM
from app.config import settings
from app.metrics import classification_metrics
from tests.metrics_factory import make_metrics, simple, synth_predictions, write


# ============================================================ matrix -> metrics
def test_metrics_from_a_matrix_equal_the_reported_ones_including_unseen_classes():
    rng = np.random.default_rng(0)
    for trial in range(120):
        k, n = int(rng.integers(2, 9)), int(rng.integers(5, 200))
        labels = [f"c{i}" for i in range(k)]
        y_true = list(rng.choice(labels, n, p=rng.dirichlet(np.ones(k))))
        y_pred = [t if rng.random() < 0.6 else rng.choice(labels) for t in y_true]
        if trial % 3 == 0:
            y_true[0] = "unseen"                                       # a class the model never predicts
        m = classification_metrics(y_true, y_pred, labels)
        got = C.metrics_from_matrices(np.array(m["confusion_matrix"]["matrix"]))
        assert got["accuracy"][0] == pytest.approx(m["accuracy"], abs=1e-12)
        assert got["f1_macro"][0] == pytest.approx(m["f1_macro"], abs=1e-12)
        assert got["f1_weighted"][0] == pytest.approx(m["f1_weighted"], abs=1e-12)


def test_a_stack_of_matrices_matches_one_by_one():
    a, b = np.array([[5, 1], [2, 4]]), np.array([[9, 0], [0, 3]])
    stack = C.metrics_from_matrices(np.stack([a, b]))
    for i, m in enumerate((a, b)):
        one = C.metrics_from_matrices(m)
        assert all(stack[k][i] == pytest.approx(one[k][0]) for k in one)
    assert C.metrics_from_matrices(b)["accuracy"][0] == 1.0 and C.metrics_from_matrices(b)["f1_macro"][0] == 1.0


# ============================================================ bootstrap intervals
def _cm(accuracy, n, k=3):
    yt, yp = synth_predictions([f"c{i}" for i in range(k)], n, accuracy)
    labels = [f"c{i}" for i in range(k)]
    return np.array(classification_metrics(yt, yp, labels)["confusion_matrix"]["matrix"])


def test_bootstrap_is_deterministic_and_seed_dependent():
    cm = _cm(0.8, 200)
    a = C.bootstrap_intervals(cm, n_bootstrap=300, seed=1)
    assert a == C.bootstrap_intervals(cm, n_bootstrap=300, seed=1)
    assert a != C.bootstrap_intervals(cm, n_bootstrap=300, seed=2)
    assert set(a) == {"accuracy", "f1_macro", "f1_weighted"} and C.bootstrap_intervals(cm, n_bootstrap=0, seed=1) is None


def test_the_interval_contains_the_point_estimate():
    for seed in range(8):
        cm = _cm(0.7, 300)
        point = C.metrics_from_matrices(cm)
        ci = C.bootstrap_intervals(cm, n_bootstrap=400, seed=seed)
        for name in ci:
            assert ci[name][0] <= point[name][0] <= ci[name][1]


def test_the_interval_matches_the_binomial_standard_error():
    n, p = 500, 0.8
    cm = np.array([[int(n * p), n - int(n * p)], [0, 0]])               # accuracy = correct / n
    lo, hi = C.bootstrap_intervals(cm, n_bootstrap=3000, seed=0)["accuracy"]
    approx_se = (hi - lo) / (2 * 1.96)
    assert approx_se == pytest.approx(np.sqrt(p * (1 - p) / n), rel=0.12)


def test_more_data_gives_a_narrower_interval_and_a_perfect_model_a_degenerate_one():
    narrow = C.bootstrap_intervals(_cm(0.8, 1000), n_bootstrap=400, seed=0)["accuracy"]
    wide = C.bootstrap_intervals(_cm(0.8, 60), n_bootstrap=400, seed=0)["accuracy"]
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert C.bootstrap_intervals(_cm(1.0, 90), n_bootstrap=100, seed=0)["f1_macro"] == [1.0, 1.0]


def test_the_95_percent_interval_covers_the_truth_about_95_percent_of_the_time():
    rng, hits, n, trials = np.random.default_rng(7), 0, 150, 150
    for t in range(trials):
        correct = int(rng.binomial(n, 0.8))
        lo, hi = C.bootstrap_intervals(np.array([[correct, n - correct], [0, 0]]), n_bootstrap=400, seed=t)["accuracy"]
        hits += lo <= 0.8 <= hi
    assert 0.88 <= hits / trials <= 0.99


# ============================================================ nothing / one model
def test_no_models_trained_is_an_honest_empty_state(env):
    c = C.compare()
    assert c["status"] == "no_models_trained" and c["models_found"] == [] and c["targets"] == {} and c["summary"] == []
    assert c["models_missing"] == list(MM.MODELS) and any("nothing to compare" in w for w in c["warnings"])


def test_a_single_model_is_reported_but_not_compared(env):
    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8}))
    c = C.compare()
    assert c["status"] == "single_model" and c["models_found"] == ["tfidf"]
    blk = c["targets"]["subject"]["test"]
    assert [e["model"] for e in blk["models"]] == ["tfidf"] and blk["separated_from_runner_up"] is None


# ============================================================ ranking and intervals
def _three(env_unused=None, n=400):
    write(simple("tfidf", accuracies={"val": 0.70, "test": 0.70}, n=n))
    write(simple("lstm", accuracies={"val": 0.80, "test": 0.80}, n=n))
    write(simple("bilstm", accuracies={"val": 0.92, "test": 0.92}, n=n))


def test_models_are_ranked_by_macro_f1_with_real_numbers(env):
    _three()
    c = C.compare(n_bootstrap=200)
    blk = c["targets"]["subject"]["test"]
    assert [e["model"] for e in blk["models"]] == ["bilstm", "lstm", "tfidf"]
    assert [e["rank_f1_macro"] for e in blk["models"]] == [1, 2, 3]
    assert blk["best_by_f1_macro"] == "bilstm" and blk["tied_for_best"] == ["bilstm"]
    raw = {m: json.loads((Path(settings.artifacts_dir) / m / "metrics.json").read_text()) for m in ("tfidf", "lstm", "bilstm")}
    for e in blk["models"]:
        src = raw[e["model"]]["targets"]["subject"]["splits"]["test"]
        assert e["f1_macro"] == src["f1_macro"] and e["accuracy"] == src["accuracy"] and e["n"] == src["n"]
    assert c["summary"][0]["best_model"] == "bilstm" and c["summary"][0]["target"] == "subject"


def test_a_clear_winner_is_separated_and_a_close_one_is_not(env):
    _three(n=400)
    blk = C.compare(n_bootstrap=400)["targets"]["subject"]["test"]
    assert blk["separated_from_runner_up"] is True                      # 0.92 vs 0.80 with n=400
    write(simple("bilstm", accuracies={"val": 0.82, "test": 0.82}, n=400))
    blk = C.compare(n_bootstrap=400)["targets"]["subject"]["test"]
    assert blk["best_by_f1_macro"] == "bilstm" and blk["separated_from_runner_up"] is False   # 0.82 vs 0.80: not established
    write(simple("bilstm", accuracies={"val": 0.92, "test": 0.92}, n=24))
    write(simple("lstm", accuracies={"val": 0.80, "test": 0.80}, n=24))
    write(simple("tfidf", accuracies={"val": 0.70, "test": 0.70}, n=24))
    assert C.compare(n_bootstrap=400)["targets"]["subject"]["test"]["separated_from_runner_up"] is False   # tiny n


def test_ranking_uses_macro_f1_not_accuracy(env):
    """On imbalanced classes a model that mostly predicts the majority class has the higher accuracy but
    the lower macro-F1; the ranking must follow macro-F1."""
    labels = ["a", "b", "c"]
    y_true = ["a"] * 80 + ["b"] * 10 + ["c"] * 10
    majority_only = ["a"] * 100                                           # accuracy 0.80, never finds b or c
    balanced = (["a"] * 60 + ["b"] * 20) + (["b"] * 8 + ["c"] * 2) + (["c"] * 8 + ["a"] * 2)
    train = ["a"] * 80 + ["b"] * 10 + ["c"] * 10
    spec = lambda pred: {"labels": labels, "train": train, "splits": {"test": (y_true, pred)}}   # noqa: E731
    write(make_metrics("lstm", {"subject": spec(majority_only)}))
    write(make_metrics("bilstm", {"subject": spec(balanced)}))
    blk = C.compare(n_bootstrap=0)["targets"]["subject"]["test"]
    by = {e["model"]: e for e in blk["models"]}
    assert by["lstm"]["accuracy"] > by["bilstm"]["accuracy"]              # the trap: accuracy says lstm
    assert by["bilstm"]["f1_macro"] > by["lstm"]["f1_macro"]              # macro-F1 says bilstm
    assert [e["model"] for e in blk["models"]] == ["bilstm", "lstm"] and blk["best_by_f1_macro"] == "bilstm"


def test_ties_share_a_rank_and_are_reported_as_a_tie(env):
    write(simple("tfidf", accuracies={"val": 0.9, "test": 0.9}))
    write(simple("bert", accuracies={"val": 0.9, "test": 0.9}))
    write(simple("lstm", accuracies={"val": 0.7, "test": 0.7}))
    blk = C.compare(n_bootstrap=100)["targets"]["subject"]["test"]
    assert [e["rank_f1_macro"] for e in blk["models"]] == [1, 1, 3]
    assert blk["tied_for_best"] == ["tfidf", "bert"] and blk["separated_from_runner_up"] is False
    assert C.compare(n_bootstrap=100)["summary"][0]["tied_for_best"] == ["tfidf", "bert"]


def test_deltas_against_tfidf_and_the_majority_baseline(env):
    _three()
    blk = C.compare(n_bootstrap=0)["targets"]["subject"]["test"]
    by = {e["model"]: e for e in blk["models"]}
    assert by["tfidf"]["f1_macro_minus_tfidf"] is None
    assert by["lstm"]["f1_macro_minus_tfidf"] == pytest.approx(by["lstm"]["f1_macro"] - by["tfidf"]["f1_macro"])
    assert by["bilstm"]["accuracy_minus_majority_baseline"] == pytest.approx(by["bilstm"]["accuracy"] - blk["majority_class_accuracy"])
    assert all(e["ci"] is None for e in blk["models"]) and blk["separated_from_runner_up"] is None   # intervals disabled


def test_intervals_are_deterministic_across_calls_and_per_model(env):
    _three()
    a, b = C.compare(n_bootstrap=200), C.compare(n_bootstrap=200)
    assert a == b
    cis = {e["model"]: e["ci"] for e in a["targets"]["subject"]["test"]["models"]}
    assert cis["tfidf"] != cis["lstm"]
    assert C.compare(n_bootstrap=200, seed=7)["targets"]["subject"]["test"]["models"][0]["ci"] != cis["bilstm"]


def test_point_estimates_inside_their_intervals_for_every_entry(env):
    _three()
    for blk in (b for t in C.compare(n_bootstrap=300)["targets"].values() for b in t.values()):
        for e in blk["models"]:
            for name, (lo, hi) in e["ci"].items():
                assert lo <= e[name] <= hi


# ============================================================ comparability
def test_models_scored_on_different_rows_are_excluded_with_a_reason(env):
    write(simple("tfidf", accuracies={"val": 0.7, "test": 0.7}, n=200))
    write(simple("lstm", accuracies={"val": 0.8, "test": 0.8}, n=200))
    write(simple("bilstm", accuracies={"val": 0.9, "test": 0.9}, n=150))      # scored on fewer rows
    blk = C.compare(n_bootstrap=0)["targets"]["subject"]["test"]
    assert [e["model"] for e in blk["models"]] == ["lstm", "tfidf"] and blk["n"] == 200
    assert list(blk["excluded"]) == ["bilstm"] and "n=150" in blk["excluded"]["bilstm"]


def test_models_with_different_classes_are_excluded(env):
    write(simple("tfidf", accuracies={"val": 0.7, "test": 0.7}, labels=("a", "b", "c")))
    write(simple("lstm", accuracies={"val": 0.8, "test": 0.8}, labels=("a", "b", "c")))
    write(simple("bert", accuracies={"val": 0.9, "test": 0.9}, labels=("a", "b", "c", "d")))
    blk = C.compare(n_bootstrap=0)["targets"]["subject"]["test"]
    assert {e["model"] for e in blk["models"]} == {"tfidf", "lstm"} and "bert" in blk["excluded"]
    assert "4 classes" in blk["excluded"]["bert"]


def test_different_split_files_raise_a_warning(env):
    write(simple("tfidf", accuracies={"val": 0.7, "test": 0.7}, sha="split-A"))
    write(simple("lstm", accuracies={"val": 0.8, "test": 0.8}, sha="split-A"))
    assert C.compare(n_bootstrap=0)["split_identity"]["files_sha256_equal"] is True      # same checksums: fine
    write(simple("bilstm", accuracies={"val": 0.8, "test": 0.8}, sha="split-B"))
    c = C.compare(n_bootstrap=0)
    assert c["split_identity"]["files_sha256_equal"] is False
    assert any("DIFFERENT processed split files" in w for w in c["warnings"])


def test_different_split_manifests_raise_a_warning(env):
    write(simple("tfidf", accuracies={"val": 0.7, "test": 0.7}, manifest_sha="input-A"))
    write(simple("lstm", accuracies={"val": 0.8, "test": 0.8}, manifest_sha="input-B"))
    c = C.compare(n_bootstrap=0)
    assert c["split_identity"]["manifest_equal"] is False and any("DIFFERENT split manifests" in w for w in c["warnings"])


def test_identical_splits_raise_no_warning(env):
    _three()
    c = C.compare(n_bootstrap=0)
    assert c["warnings"] == [] and c["split_identity"]["files_sha256_equal"] is True and c["split_identity"]["manifest_equal"] is True


def test_missing_split_evidence_is_flagged(env):
    write(simple("tfidf", accuracies={"val": 0.7, "test": 0.7}, with_file_sha=False, with_manifest=False))
    write(simple("lstm", accuracies={"val": 0.8, "test": 0.8}))
    c = C.compare(n_bootstrap=0)
    assert c["split_identity"]["models_without_evidence"] == ["tfidf"]
    assert any("cannot be verified" in w for w in c["warnings"])


def test_a_broken_metrics_file_is_reported_not_compared(env):
    _three()
    bad = json.loads((Path(settings.artifacts_dir) / "bilstm" / "metrics.json").read_text())
    bad["targets"]["subject"]["splits"]["test"]["accuracy"] = 9
    write(bad)
    c = C.compare(n_bootstrap=0)
    assert list(c["problems"]) == ["bilstm"] and c["models_found"] == ["tfidf", "lstm"]
    assert [e["model"] for e in c["targets"]["subject"]["test"]["models"]] == ["lstm", "tfidf"]


def test_a_skipped_target_is_listed_and_the_others_are_still_compared(env):
    labels = ["a", "b", "c"]
    train = [labels[i % 3] for i in range(90)]
    trained = {"labels": labels, "train": train, "splits": {"test": synth_predictions(labels, 200, 0.8)}}
    write(make_metrics("tfidf", {"subject": trained, "topic": trained}))
    write(make_metrics("lstm", {"subject": trained, "topic": "skipped"}))
    c = C.compare(n_bootstrap=0)
    assert [e["model"] for e in c["targets"]["subject"]["test"]["models"]] == ["tfidf", "lstm"]
    topic = c["targets"]["topic"]["test"]
    assert [e["model"] for e in topic["models"]] == ["tfidf"] and topic["skipped_target"] == {"lstm": "no labelled rows"}


# ============================================================ test vs pdf_heldout
def test_generalisation_gap_between_test_and_pdf_heldout(env):
    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8, "pdf_heldout": 0.5}))
    write(simple("lstm", accuracies={"val": 0.9, "test": 0.9, "pdf_heldout": 0.6}))
    c = C.compare(n_bootstrap=0)
    g = c["generalisation_gap"]["subject"]
    for model in ("tfidf", "lstm"):
        t = c["targets"]["subject"]["test"]["models"]
        h = c["targets"]["subject"]["pdf_heldout"]["models"]
        tf = next(e for e in t if e["model"] == model)["f1_macro"]
        hf = next(e for e in h if e["model"] == model)["f1_macro"]
        assert g[model]["f1_macro_gap"] == pytest.approx(tf - hf) and g[model]["f1_macro_gap"] > 0.1
    assert "pdf_heldout" in c["targets"]["subject"]


def test_no_gap_is_reported_without_heldout_data(env):
    _three()
    assert C.compare(n_bootstrap=0)["generalisation_gap"] == {}


# ============================================================ exports
def test_flat_rows_equal_the_comparison(env):
    _three()
    c = C.compare(n_bootstrap=200)
    rows = C.comparison_rows(c)
    assert len(rows) == 2 * 3                                           # val + test, 3 models
    for r in rows:
        e = next(x for x in c["targets"][r["target"]][r["split"]]["models"] if x["model"] == r["model"])
        assert r["f1_macro"] == e["f1_macro"] and r["accuracy"] == e["accuracy"] and r["n"] == e["n"]
        assert (r["f1_macro_ci_low"], r["f1_macro_ci_high"]) == tuple(e["ci"]["f1_macro"])
    assert sum(r["best_in_group"] for r in rows if r["split"] == "test") == 1


def test_markdown_report_states_the_numbers_and_the_caveats(env):
    _three()
    c = C.compare(n_bootstrap=100)
    md = C.to_markdown(c)
    for e in c["targets"]["subject"]["test"]["models"]:
        assert f"{e['f1_macro']:.4f}" in md and f"{e['accuracy']:.4f}" in md
    assert "Best by macro-F1: **bilstm**" in md and "unpaired" in md and "Not trained (absent): bert" in md
    assert "no paired test is claimed" in md


def test_markdown_flags_ties_and_warnings(env):
    write(simple("tfidf", accuracies={"val": 0.9, "test": 0.9}, sha="A"))
    write(simple("bert", accuracies={"val": 0.9, "test": 0.9}, sha="B"))
    md = C.to_markdown(C.compare(n_bootstrap=0))
    assert "Tied for best by macro-F1 (identical point estimates): **tfidf, bert**" in md
    assert "> WARNING: The models were trained on DIFFERENT processed split files" in md


# ============================================================ CLI
def test_cli_writes_the_three_files_from_real_metrics(env, capsys):
    _three()
    from training.compare_models import main

    assert main(["--bootstrap", "50"]) == 0
    out = Path(settings.artifacts_dir) / "comparison"
    assert sorted(p.name for p in out.iterdir()) == ["comparison.csv", "comparison.json", "comparison.md"]
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert data["status"] == "ok" and data["models_found"] == ["tfidf", "lstm", "bilstm"]
    with open(out / "comparison.csv", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 6
    assert "best on test by macro-F1: bilstm" in capsys.readouterr().out


def test_cli_with_fewer_than_two_models_writes_nothing(env, capsys):
    from training.compare_models import main

    assert main([]) == 2
    assert "NOT AVAILABLE YET" in capsys.readouterr().err and not (Path(settings.artifacts_dir) / "comparison").exists()
    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8}))
    assert main([]) == 2 and not (Path(settings.artifacts_dir) / "comparison").exists()


def test_cli_reports_contract_problems_and_rejects_bad_options(env, capsys):
    from training.compare_models import main

    write(simple("tfidf", accuracies={"val": 0.8, "test": 0.8}))
    bad = simple("lstm", accuracies={"val": 0.8, "test": 0.8})
    bad["targets"]["subject"]["splits"]["test"]["accuracy"] = 4
    write(bad)
    assert main([]) == 2
    assert "lstm: metrics.json does not satisfy the contract" in capsys.readouterr().err
    assert main(["--bootstrap", "-1"]) == 2


def test_cli_prints_ties_and_warnings(env, capsys):
    from training.compare_models import main

    write(simple("tfidf", accuracies={"val": 0.9, "test": 0.9}))
    write(simple("bert", accuracies={"val": 0.9, "test": 0.9}))
    assert main(["--bootstrap", "0", "--out", str(env / "elsewhere")]) == 0
    assert "TIE between tfidf, bert" in capsys.readouterr().out and (env / "elsewhere" / "comparison.md").is_file()


# ============================================================ the real (tiny, synthetic-data) models
def test_real_metrics_files_reproduce_from_their_confusion_matrices(four_models_trained):
    """For every model, split and target in the REAL metrics files, recomputing accuracy, macro-F1 and
    weighted-F1 from the stored confusion matrix gives exactly the reported numbers: the basis of the intervals."""
    checked = 0
    for model in MM.MODELS:
        f = Path(settings.artifacts_dir) / model / "metrics.json"
        if not f.is_file():
            continue
        for e in json.loads(f.read_text())["targets"].values():
            for s in e.get("splits", {}).values():
                got = C.metrics_from_matrices(np.array(s["confusion_matrix"]["matrix"]))
                assert got["accuracy"][0] == pytest.approx(s["accuracy"], abs=1e-12)
                assert got["f1_macro"][0] == pytest.approx(s["f1_macro"], abs=1e-12)
                assert got["f1_weighted"][0] == pytest.approx(s["f1_weighted"], abs=1e-12)
                checked += 1
    assert checked == 4 * 3 * 2


def test_the_four_real_models_are_compared_on_identical_rows(four_models_trained):
    c = C.compare(n_bootstrap=100)
    assert c["status"] == "ok" and c["models_found"] == list(MM.MODELS) and c["models_missing"] == []
    assert c["warnings"] == [] and c["problems"] == {}
    assert c["split_identity"]["files_sha256_equal"] is True and c["split_identity"]["manifest_equal"] is True
    assert c["split_identity"]["models_without_evidence"] == []
    for target in ("subject", "topic", "difficulty"):
        for split in ("val", "test"):
            blk = c["targets"][target][split]
            assert [e["model"] for e in blk["models"]] != [] and len(blk["models"]) == 4 and blk["excluded"] == {}
            assert len({e["n"] for e in blk["models"]}) == 1
    rows = {(r["model"], r["target"], r["split"]): r for r in MM.collect()["rows"]}
    for blk_target, splits in c["targets"].items():
        for split, blk in splits.items():
            for e in blk["models"]:
                assert e["f1_macro"] == rows[(e["model"], blk_target, split)]["f1_macro"]
    assert len(c["summary"]) == 3 and all(s["models_compared"] == 4 for s in c["summary"])


def test_the_real_comparison_exports_are_consistent(four_models_trained, capsys):
    from training.compare_models import main

    assert main(["--bootstrap", "50"]) == 0
    out = Path(settings.artifacts_dir) / "comparison"
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    md = (out / "comparison.md").read_text(encoding="utf-8")
    for model in MM.MODELS:
        assert model in md
    assert data["models_found"] == list(MM.MODELS) and "Not trained (absent): none" in md
