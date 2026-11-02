"""Phase 5 tests: PyTorch BiLSTM text classifier.

Covers (1) the bidirectional architecture itself, (2) the full training / evaluation /
checkpoint / inference protocol, and (3) the guarantees that tie the BiLSTM to the LSTM
(same data, splits, seed 42, preprocessing and vocabulary; no cross-loading of checkpoints).

All data is SYNTHETIC (see conftest): it exercises the code and proves properties
(determinism, no leakage, correct metrics, safe checkpoints). Nothing computed on it is a
result about real performance. Skipped automatically if PyTorch is not installed.
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from app import bilstm as B  # noqa: E402
from app import data as D  # noqa: E402
from app import lstm as L  # noqa: E402
from app import word2vec as W  # noqa: E402
from app.config import settings  # noqa: E402
from app.registry import model_status  # noqa: E402
from tests.conftest import TINY_LSTM, make_toy_frame  # noqa: E402


def tiny(**kw):
    return B.BiLSTMConfig(**{**TINY_LSTM, **kw})


def _out():
    return B.artifacts_dir()


def _ckpt(target="subject"):
    return torch.load(_out() / f"{target}.pt", map_location="cpu", weights_only=True)


def _append_rows(split, text, n=10):
    p = D.processed_path(split)
    df = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=D.DATA_COLUMNS)
    extra = pd.DataFrame([{"text": text, "subject": "History", "topic": "", "difficulty": ""}] * n)
    pd.concat([df, extra], ignore_index=True).to_csv(p, index=False)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    return toy_csv


def _bi(**kw):
    cfg = dict(vocab_size=30, embedding_dim=8, hidden_dim=12, num_layers=1, num_classes=4, dropout=0.5,
               bidirectional=True)
    cfg.update(kw)
    torch.manual_seed(0)
    return L.LSTMTextClassifier(**cfg)


# ============================================================ architecture
def test_bidirectional_layout_and_shapes():
    m = _bi().eval()
    out = m(torch.randint(2, 30, (5, 7)), torch.tensor([7, 5, 3, 2, 1]))
    assert tuple(out.shape) == (5, 4)
    assert m.bidirectional and m.lstm.bidirectional
    assert m.fc.in_features == 2 * 12                       # forward + backward state
    assert L.LSTMTextClassifier(30, 8, 12, 1, 4, 0.5).fc.in_features == 12   # the LSTM is unchanged


def test_parameter_count_matches_the_closed_form():
    """Independent of PyTorch's own counting: embedding + recurrent weights + classifier."""
    V, E, H, C = 30, 8, 12, 4
    recurrent_per_direction = 4 * H * (E + H) + 8 * H       # W_ih + W_hh + two bias vectors
    expected_bi = V * E + 2 * recurrent_per_direction + (2 * H * C + C)
    expected_uni = V * E + recurrent_per_direction + (H * C + C)
    assert sum(p.numel() for p in _bi().parameters()) == expected_bi
    assert sum(p.numel() for p in _bi(bidirectional=False).parameters()) == expected_uni
    assert expected_bi - expected_uni == recurrent_per_direction + H * C


@pytest.mark.parametrize("layers", [1, 2])
def test_representation_is_forward_last_and_backward_first_state(layers):
    """Recompute the model by hand from the LSTM's per-step outputs: the classifier must read
    [forward state after the LAST token ; backward state after the FIRST token]."""
    m = _bi(num_layers=layers).eval()
    H = 12
    ids = torch.randint(2, 30, (3, 6))
    with torch.no_grad():
        out, _ = m.lstm(m.embedding(ids))                              # (B, T, 2H), no padding
        manual = m.fc(torch.cat([out[:, -1, :H], out[:, 0, H:]], dim=1))
        got = m(ids, torch.full((3,), 6))
    assert torch.allclose(manual, got, atol=1e-6)


def test_padding_content_never_affects_the_output():
    """With packed sequences the backward direction must start at each sequence's true last
    token, not at a padding position, whatever the padding contains."""
    m = _bi().eval()
    ids = torch.randint(2, 30, (4, 9))
    lens = torch.tensor([9, 6, 4, 2])
    base = m(ids, lens)
    garbage = ids.clone()
    for i, n in enumerate(lens.tolist()):
        garbage[i, n:] = torch.randint(2, 30, (9 - n,))
    assert torch.allclose(base, m(garbage, lens), atol=1e-6)


def test_each_sequence_is_independent_of_its_batch():
    m = _bi().eval()
    ids = torch.randint(2, 30, (6, 8))
    lens = torch.tensor([8, 7, 5, 4, 3, 1])
    batch = m(ids, lens)
    for i in range(6):
        single = m(ids[i : i + 1, : lens[i]], lens[i : i + 1])
        assert torch.allclose(batch[i : i + 1], single, atol=1e-5)


def test_the_output_depends_on_token_order():
    """A recurrent model is not a bag of words: reversing the sequence changes the logits."""
    m = _bi().eval()
    ids = torch.randint(2, 30, (1, 6))
    a = m(ids, torch.tensor([6]))
    b = m(ids.flip(1), torch.tensor([6]))
    assert not torch.allclose(a, b, atol=1e-6)


def test_gradients_reach_both_directions_and_padding_row_stays_zero():
    m = _bi(num_layers=2)
    ids, lens = torch.randint(2, 30, (4, 6)), torch.full((4,), 6)
    torch.nn.functional.cross_entropy(m(ids, lens), torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for name in ("weight_ih_l0", "weight_ih_l0_reverse", "weight_ih_l1", "weight_ih_l1_reverse"):
        assert torch.count_nonzero(getattr(m.lstm, name).grad) > 0, name   # both directions learn
    assert torch.count_nonzero(m.embedding.weight.grad[L.PAD_IDX]) == 0


def test_dropout_only_active_in_training_mode():
    ids, lens = torch.randint(2, 30, (4, 6)), torch.full((4,), 6)
    m = _bi(dropout=0.5)
    m.eval()
    assert torch.equal(m(ids, lens), m(ids, lens))
    m.train()
    torch.manual_seed(1)
    a = m(ids, lens)
    torch.manual_seed(2)
    assert not torch.allclose(a, m(ids, lens))
    assert m.lstm.dropout == 0.0 and _bi(num_layers=2).lstm.dropout == 0.5   # inter-layer only when stacked


# ============================================================ façade and config
def test_facade_reuses_the_lstm_implementation():
    assert B.BiLSTMConfig is L.LSTMConfig and B.BiLSTMNotTrainedError is L.LSTMNotTrainedError
    assert B.BiLSTMTrainingError is L.LSTMTrainingError and issubclass(B.BiLSTMService, L.LSTMService)
    assert B.MODEL_NAME == "bilstm" and B.artifacts_dir().name == "bilstm"
    assert B.artifacts_dir() != L.artifacts_dir() and L.artifacts_dir().name == "lstm"
    assert L.BILSTM_SPEC.bidirectional and not L.LSTM_SPEC.bidirectional
    assert set(L.SPECS) == {"lstm", "bilstm"}


# ============================================================ not trained
def test_no_data_means_not_trained_and_nothing_written(env):
    with pytest.raises(D.DataNotFoundError):
        B.train_all(cfg=tiny())
    assert not _out().exists() or not any(_out().iterdir())
    with pytest.raises(L.LSTMNotTrainedError, match="BiLSTM not trained yet"):
        B.BiLSTMService.load()


def test_cli_reports_not_trained_yet(env, capsys):
    from training.train_bilstm import main

    assert main(["--epochs", "1"]) == 2
    assert "BiLSTM: NOT TRAINED YET" in capsys.readouterr().err
    assert not _out().exists() or not any(_out().iterdir())


def test_cli_without_pytorch_exits_with_clear_message(env, capsys, monkeypatch):
    from training.train_bilstm import main

    monkeypatch.setitem(sys.modules, "app.lstm", None)       # makes `import app.lstm` raise ImportError
    assert main([]) == 3
    assert "BiLSTM: cannot start, PyTorch is not installed" in capsys.readouterr().err


def test_cli_rejects_invalid_setting(env, capsys):
    from training.train_bilstm import main

    assert main(["--dropout", "1.5"]) == 2
    assert "BiLSTM: invalid setting" in capsys.readouterr().err


def test_cli_trains_the_bilstm_into_its_own_folder(prepared, capsys):
    from training.train_bilstm import main

    assert main(["--targets", "subject", "--epochs", "3", "--embedding-dim", "8", "--hidden-dim", "8"]) == 0
    out = capsys.readouterr().out
    assert "Training BiLSTM" in out and "BiLSTM training finished" in out and "macro-F1" in out
    assert "artifacts/bilstm/" in out
    assert (_out() / "subject.pt").is_file() and (_out() / "metrics.json").is_file()
    assert not L.artifacts_dir().exists() or not any(L.artifacts_dir().iterdir())   # LSTM folder untouched


def test_nothing_trainable_writes_nothing(env):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    with pytest.raises(L.LSTMTrainingError, match="No target could be trained"):
        B.train_all(cfg=tiny(epochs=2), targets=("topic",))
    assert not _out().exists() or not any(_out().iterdir())


def test_invalid_targets_rejected(prepared):
    with pytest.raises(ValueError):
        B.train_all(cfg=tiny(), targets=("subject", "colour"))
    with pytest.raises(ValueError):
        B.train_all(cfg=tiny(), targets=())


# ============================================================ artifacts
def test_artifacts_and_metrics_structure(bilstm_trained):
    m = bilstm_trained
    for t in ("subject", "topic", "difficulty"):
        assert (_out() / f"{t}.pt").is_file()
    assert (_out() / "metrics.json").is_file()
    assert not L.artifacts_dir().exists() or not any(L.artifacts_dir().iterdir())   # nothing leaked into lstm/
    assert m["model"] == "bilstm" and "idirectional" in m["description"] and m["seed"] == 42
    assert m["framework"]["torch"] == torch.__version__ and m["framework"]["device"] == "cpu"
    assert m["hyperparameters"] == tiny().to_dict()
    e = m["targets"]["subject"]
    assert e["status"] == "trained" and e["labels"] == sorted(e["labels"])
    assert set(e["splits"]) == {"val", "test"} and m["data"]["pdf_heldout_evaluated"] is False
    assert e["model"]["bidirectional"] is True
    assert e["training"]["epochs_run"] == len(e["training"]["history"])
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                "precision_weighted", "recall_weighted", "confusion_matrix", "per_class"):
        assert key in e["splits"]["test"]


def test_checkpoint_is_safe_plain_data_and_bidirectional(bilstm_trained):
    ck = _ckpt()
    assert ck["arch"] == "bilstm" and ck["target"] == "subject" and ck["model_config"]["bidirectional"] is True
    assert ck["vocab"][:2] == ["<pad>", "<unk>"]
    assert all(isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in ck["state_dict"].values())
    sd = ck["state_dict"]
    assert "lstm.weight_ih_l0_reverse" in sd                                  # the backward LSTM exists
    H, C = ck["model_config"]["hidden_dim"], len(ck["labels"])
    assert tuple(sd["fc.weight"].shape) == (C, 2 * H)


def test_recorded_parameter_count_matches_the_checkpoint(bilstm_trained):
    for target in ("subject", "topic", "difficulty"):
        ck = _ckpt(target)
        total = sum(v.numel() for v in ck["state_dict"].values())
        rec = bilstm_trained["targets"][target]["model"]
        assert rec["total_parameters"] == total and rec["trainable_parameters"] == total


def test_class_weights_recorded(bilstm_trained):
    counts = D.load_split("public_train")["subject"].value_counts().sort_index().tolist()
    recorded = bilstm_trained["targets"]["subject"]["training"]["class_weights"]
    assert np.allclose(recorded, L.balanced_class_weights(counts).numpy(), rtol=1e-6)


# ============================================================ same conventions as the LSTM
def test_same_data_seed_preprocessing_and_vocabulary_as_the_lstm(bilstm_trained, _lstm_baseline):
    lroot, lm = _lstm_baseline
    bm = bilstm_trained
    assert bm["seed"] == lm["seed"] == 42
    assert bm["preprocessing"] == lm["preprocessing"]
    assert bm["hyperparameters"] == lm["hyperparameters"]
    assert bm["vocabulary"] == lm["vocabulary"]
    assert bm["data"]["split_files_sha256"] == lm["data"]["split_files_sha256"]      # byte-identical splits
    assert bm["data"]["rows"] == lm["data"]["rows"] and bm["data"]["encoding"] == lm["data"]["encoding"]
    for t in ("subject", "topic", "difficulty"):
        lc = torch.load(lroot / "artifacts" / "lstm" / f"{t}.pt", map_location="cpu", weights_only=True)
        bc = _ckpt(t)
        assert bc["vocab"] == lc["vocab"] and bc["labels"] == lc["labels"]
        assert bc["preprocess_config"] == lc["preprocess_config"] and bc["max_len"] == lc["max_len"]
        assert bm["targets"][t]["labels"] == lm["targets"][t]["labels"]
        assert bm["targets"][t]["n_train"] == lm["targets"][t]["n_train"]
        for split in ("val", "test"):                                   # the very same rows are scored
            assert bm["targets"][t]["splits"][split]["n"] == lm["targets"][t]["splits"][split]["n"]
            assert bm["targets"][t]["splits"][split]["confusion_matrix"]["labels"] == \
                lm["targets"][t]["splits"][split]["confusion_matrix"]["labels"]


def test_the_bilstm_has_more_parameters_than_the_lstm_with_equal_settings(bilstm_trained, _lstm_baseline):
    _, lm = _lstm_baseline
    H = TINY_LSTM["hidden_dim"]
    for t, C in (("subject", 4), ("topic", 8), ("difficulty", 3)):
        b = bilstm_trained["targets"][t]["model"]["trainable_parameters"]
        l = lm["targets"][t]["model"]["trainable_parameters"]
        E = TINY_LSTM["embedding_dim"]
        assert b - l == (4 * H * (E + H) + 8 * H) + H * C                # the extra direction + wider classifier


# ============================================================ metrics are real
@pytest.mark.parametrize("split", ["val", "test"])
def test_reported_metrics_match_independent_recomputation(bilstm_trained, split):
    """Reload the SAVED BiLSTM checkpoint, score the SAVED split yourself, compare every
    reported number. Also proves save -> load reproduces the trained model exactly."""
    svc = B.BiLSTMService.load()
    df = D.load_split(f"public_{split}")
    preds = svc.predict(df["text"].tolist(), top_k=1)
    y_pred = [p["predictions"]["subject"]["label"] for p in preds]
    assert None not in y_pred
    y_true = df["subject"].tolist()
    rep = bilstm_trained["targets"]["subject"]["splits"][split]
    labels = bilstm_trained["targets"]["subject"]["labels"]

    assert rep["n"] == len(df)
    assert rep["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        assert rep[f"precision_{avg}"] == pytest.approx(p)
        assert rep[f"recall_{avg}"] == pytest.approx(r)
        assert rep[f"f1_{avg}"] == pytest.approx(f)
    cm = np.array(rep["confusion_matrix"]["matrix"])
    assert rep["confusion_matrix"]["labels"] == labels
    assert (cm == confusion_matrix(y_true, y_pred, labels=labels)).all() and cm.sum() == len(df)
    assert rep["majority_class_baseline"]["class"] == D.load_split("public_train")["subject"].value_counts().idxmax()


def test_the_model_learns_the_synthetic_structure(bilstm_trained):
    """Sanity check of the whole chain on a task with known structure (chance is 0.25).
    Not a performance claim."""
    assert bilstm_trained["targets"]["subject"]["splits"]["test"]["accuracy"] > 0.9


# ============================================================ training protocol
def test_best_epoch_is_selected_on_validation(bilstm_trained):
    t = bilstm_trained["targets"]["subject"]["training"]
    f1s = [h["val_f1_macro"] for h in t["history"]]
    assert t["best_epoch"] == 1 + int(np.argmax(f1s))
    assert t["model_selection"] == "best validation macro-F1"
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in t["history"])
    assert t["history"][-1]["train_loss"] < t["history"][0]["train_loss"]


def test_restored_weights_are_exactly_those_of_the_best_epoch(prepared):
    full = B.train_all(cfg=tiny(epochs=40, patience=4), targets=("subject",))
    t = full["targets"]["subject"]["training"]
    assert t["stopped_early"] and t["best_epoch"] < t["epochs_run"]
    restored = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    B.train_all(cfg=tiny(epochs=t["best_epoch"], patience=0), targets=("subject",))
    assert all(torch.equal(restored[k], _ckpt()["state_dict"][k]) for k in restored)


def test_early_stopping_and_disabling_it(prepared):
    m = B.train_all(cfg=tiny(epochs=60, patience=2), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is True and t["epochs_run"] == t["best_epoch"] + 2 < 60
    m = B.train_all(cfg=tiny(epochs=6, patience=0), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is False and t["epochs_run"] == 6


def test_training_is_reproducible_and_seed_dependent(prepared):
    cfg = tiny(epochs=6, patience=0)
    a = B.train_all(cfg=cfg, targets=("subject",), seed=42)
    wa = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    b = B.train_all(cfg=cfg, targets=("subject",), seed=42)
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert all(torch.equal(wa[k], _ckpt()["state_dict"][k]) for k in wa)
    B.train_all(cfg=cfg, targets=("subject",), seed=7)
    assert not all(torch.equal(wa[k], _ckpt()["state_dict"][k]) for k in wa)


def test_each_target_is_independent_of_the_others(prepared):
    cfg = tiny(epochs=5, patience=0)
    B.train_all(cfg=cfg, targets=("subject",))
    alone = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    B.train_all(cfg=cfg, targets=("subject", "difficulty"))
    assert all(torch.equal(alone[k], _ckpt()["state_dict"][k]) for k in alone)


def test_test_and_heldout_data_never_influence_training(prepared):
    """Replace the TEST and PDF held-out data completely: weights, vocabulary, chosen epoch and
    history must be identical; only the reported test numbers may change."""
    cfg = tiny(epochs=8, patience=0)
    a = B.train_all(cfg=cfg, targets=("subject",))
    w_a = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    vocab_a = _ckpt()["vocab"]
    test = D.load_split("public_test")
    test.assign(subject=test["subject"].sample(frac=1, random_state=1).values,
                text=test["text"].str[::-1]).to_csv(D.processed_path("public_test"), index=False)
    make_toy_frame(8, seed=99).to_csv(D.processed_path("pdf_heldout"), index=False)
    b = B.train_all(cfg=cfg, targets=("subject",))
    assert _ckpt()["vocab"] == vocab_a
    assert all(torch.equal(w_a[k], _ckpt()["state_dict"][k]) for k in w_a)
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert a["targets"]["subject"]["splits"]["test"]["accuracy"] != b["targets"]["subject"]["splits"]["test"]["accuracy"]
    assert "pdf_heldout" in b["targets"]["subject"]["splits"]


def test_vocabulary_comes_from_train_only(prepared):
    for split in ("public_val", "public_test", "pdf_heldout"):
        _append_rows(split, "zzsentinel zzsentinel zzsentinel zzsentinel history")
    B.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert "zzsentinel" not in _ckpt()["vocab"]


def test_sentinel_control_enters_vocab_when_in_train(prepared):
    _append_rows("public_train", "zzsentinel zzsentinel zzsentinel zzsentinel history")
    B.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert "zzsentinel" in _ckpt()["vocab"]


def test_heldout_evaluated_only_when_present_with_unseen_labels_as_errors(prepared):
    m = B.train_all(cfg=tiny(epochs=3), targets=("subject",))
    assert "pdf_heldout" not in m["targets"]["subject"]["splits"]
    held = make_toy_frame(6, seed=12)
    held.loc[:2, "subject"] = "Economics"                          # a class the model never saw
    held.to_csv(D.processed_path("pdf_heldout"), index=False)
    m = B.train_all(cfg=tiny(epochs=3), targets=("subject",))
    r = m["targets"]["subject"]["splits"]["pdf_heldout"]
    assert m["data"]["pdf_heldout_evaluated"] is True and r["n"] == len(held)
    assert r["unseen_label_rows"] == 3 and "Economics" in r["confusion_matrix"]["labels"]
    assert r["per_class"]["Economics"]["recall"] == 0.0
    assert m["data"]["split_files_sha256"]["pdf_heldout"] == D.file_sha256(D.processed_path("pdf_heldout"))


def test_target_without_labels_is_skipped_with_reason(env):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    m = B.train_all(cfg=tiny(epochs=2))
    assert m["targets"]["topic"]["status"] == "skipped"
    assert "needs >= 2 classes" in m["targets"]["topic"]["reason"]
    assert m["targets"]["subject"]["status"] == "trained" and not (_out() / "topic.pt").exists()


def test_no_validation_rows_falls_back_to_last_epoch(prepared):
    D.load_split("public_val").assign(difficulty="").to_csv(D.processed_path("public_val"), index=False)
    m = B.train_all(cfg=tiny(epochs=4, patience=2), targets=("difficulty",))
    t = m["targets"]["difficulty"]["training"]
    assert t["model_selection"].startswith("last epoch") and t["epochs_run"] == 4 and t["best_epoch"] == 4


def test_stale_checkpoints_removed_on_retrain(prepared):
    B.train_all(cfg=tiny(epochs=2))
    assert (_out() / "topic.pt").exists()
    B.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "topic.pt").exists()


def test_marker_is_written_last_and_health_stays_honest(prepared, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(L, "_write_json_atomic", boom)       # the shared implementation writes the marker
    with pytest.raises(OSError):
        B.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "metrics.json").exists()
    st = next(m for m in model_status() if m["name"] == "bilstm")
    assert st["trained"] is False and st["targets_trained"] == []


def test_progress_callback_reports_real_epochs(prepared):
    seen = []
    B.train_all(cfg=tiny(epochs=3, patience=0), targets=("subject",), on_epoch=lambda t, r: seen.append((t, r)))
    assert [r["epoch"] for _, r in seen] == [1, 2, 3] and all("val_f1_macro" in r for _, r in seen)


# ============================================================ isolation from the LSTM
def test_training_the_bilstm_never_touches_the_lstm_artifacts(lstm_trained, _bilstm_baseline):
    lstm_dir = L.artifacts_dir()
    before = {f.name: _sha(f) for f in lstm_dir.iterdir() if f.suffix == ".pt"}
    B.train_all(cfg=tiny(epochs=3), targets=("subject",))
    assert {f.name: _sha(f) for f in lstm_dir.iterdir() if f.suffix == ".pt"} == before
    assert (lstm_dir / "metrics.json").is_file() and (_out() / "subject.pt").is_file()


def test_registry_reports_each_model_independently(lstm_trained):
    status = lambda n: next(m for m in model_status() if m["name"] == n)  # noqa: E731
    assert status("lstm")["trained"] is True and status("bilstm")["trained"] is False
    assert status("bilstm")["targets_trained"] == []
    B.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert status("bilstm")["trained"] is True and status("bilstm")["targets_trained"] == ["subject"]
    assert status("lstm")["targets_trained"] == ["difficulty", "subject", "topic"]
    for f in L.artifacts_dir().glob("*.pt"):
        f.unlink()
    assert status("lstm")["trained"] is False and status("bilstm")["trained"] is True


# ============================================================ Word2Vec init
def test_word2vec_embeddings_initialise_and_freeze(w2v_trained):
    m = B.train_all(cfg=tiny(epochs=3, patience=0, init_embeddings="word2vec", freeze_embeddings=True),
                    targets=("subject",))
    emb = m["embeddings"]
    assert emb["init"] == "word2vec" and emb["dimension"] == 50 and 0 < emb["coverage"] < 1
    ck = _ckpt()
    assert ck["model_config"]["embedding_dim"] == 50 and ck["model_config"]["bidirectional"] is True
    assert tuple(ck["state_dict"]["fc.weight"].shape)[1] == 2 * TINY_LSTM["hidden_dim"]
    wv = W.EmbeddingService.load().vectors["skipgram"]
    weights, covered = ck["state_dict"]["embedding.weight"], 0
    for i, word in enumerate(ck["vocab"]):
        if i > 1 and word in wv.key_to_index:
            assert np.allclose(weights[i].numpy(), wv[word])
            covered += 1
    assert covered == emb["words_covered"] > 0


def test_word2vec_init_refused_if_trained_on_a_different_split(w2v_trained):
    train = D.load_split("public_train")
    pd.concat([train, train.head(1).assign(text="a brand new training sentence")]).to_csv(
        D.processed_path("public_train"), index=False)
    with pytest.raises(L.LSTMTrainingError, match="different public_train"):
        B.train_all(cfg=tiny(init_embeddings="word2vec"), targets=("subject",))


# ============================================================ inference
def test_inference_shapes_probabilities_and_order(bilstm_trained):
    svc = B.BiLSTMService.load()
    assert isinstance(svc, B.BiLSTMService) and svc.spec is L.BILSTM_SPEC
    assert svc.targets == ["difficulty", "subject", "topic"]
    out = svc.predict(["the river delta and monsoon climate", "an ancient empire battle"], top_k=10)
    s = out[0]["predictions"]["subject"]
    assert s["label"] == "Geography" and out[1]["predictions"]["subject"]["label"] == "History"
    probs = [t["probability"] for t in s["top"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-5) and probs == sorted(probs, reverse=True)
    assert s["top"][0]["label"] == s["label"] and s["confidence"] == pytest.approx(probs[0])
    assert len(svc.predict(["river delta"], top_k=2)[0]["predictions"]["subject"]["top"]) == 2


def test_batched_and_single_predictions_agree(bilstm_trained):
    svc = B.BiLSTMService.load()
    texts = ["river delta", "an ancient empire battle dynasty", "atom",
             "the constitution article of the federal parliament"]
    batched = svc.predict(texts, top_k=4)
    for text, b in zip(texts, batched):
        single = svc.predict([text], top_k=4)[0]
        for target in svc.targets:
            assert single["predictions"][target]["label"] == b["predictions"][target]["label"]
            assert single["predictions"][target]["confidence"] == pytest.approx(
                b["predictions"][target]["confidence"], abs=1e-5)


def test_unreadable_text_gets_none_and_a_reason(bilstm_trained):
    out = B.BiLSTMService.load().predict(["zzz qqq xxx", "   ", "!!!", "river delta"])
    for i in (0, 1, 2):
        p = out[i]["predictions"]["subject"]
        assert p["label"] is None and p["confidence"] is None and p["top"] == []
    assert "out of the model's vocabulary" in out[0]["predictions"]["subject"]["warning"]
    assert "no usable tokens" in out[1]["predictions"]["subject"]["warning"]
    assert out[3]["predictions"]["subject"]["label"] == "Geography"


def test_long_text_is_truncated_and_inference_is_deterministic(bilstm_trained):
    svc = B.BiLSTMService.load()
    long_text = "river delta monsoon " * 200
    a = svc.predict([long_text])[0]["predictions"]["subject"]
    assert a["label"] == "Geography" and a == svc.predict([long_text])[0]["predictions"]["subject"]
    assert all(not m["model"].training for m in svc.models.values())


# ============================================================ loading errors and isolation
def test_missing_and_corrupt_checkpoints_are_clear_errors(bilstm_trained):
    (_out() / "subject.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(L.LSTMNotTrainedError, match="could not load checkpoint"):
        B.BiLSTMService.load()
    (_out() / "subject.pt").unlink()
    with pytest.raises(L.LSTMNotTrainedError, match="missing"):
        B.BiLSTMService.load()


class _Evil:
    def __reduce__(self):                                    # would run code if unpickled
        return (print, ("PWNED",))


def test_loading_never_executes_pickled_code(bilstm_trained, capsys):
    ck = _ckpt()
    ck["evil"] = _Evil()
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(L.LSTMNotTrainedError, match="could not load checkpoint"):
        B.BiLSTMService.load()
    assert "PWNED" not in capsys.readouterr().out


def test_an_lstm_checkpoint_cannot_be_loaded_as_a_bilstm(lstm_trained):
    ck = torch.load(L.artifacts_dir() / "subject.pt", map_location="cpu", weights_only=True)
    _out().mkdir(parents=True, exist_ok=True)
    for f in L.artifacts_dir().iterdir():                    # metrics + all checkpoints, but filed under bilstm/
        (_out() / f.name).write_bytes(f.read_bytes())
    assert ck["arch"] == "lstm"
    with pytest.raises(L.LSTMNotTrainedError, match="'lstm' checkpoint, expected 'bilstm'"):
        B.BiLSTMService.load()


def test_a_bilstm_checkpoint_cannot_be_loaded_as_an_lstm(bilstm_trained):
    lstm_dir = L.artifacts_dir()
    lstm_dir.mkdir(parents=True, exist_ok=True)
    for f in _out().iterdir():
        (lstm_dir / f.name).write_bytes(f.read_bytes())
    with pytest.raises(L.LSTMNotTrainedError, match="'bilstm' checkpoint, expected 'lstm'"):
        L.LSTMService.load()


def test_directionality_flag_must_match_the_architecture(bilstm_trained):
    ck = _ckpt()
    ck["model_config"] = {**ck["model_config"], "bidirectional": False}   # claims bilstm, says unidirectional
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(L.LSTMNotTrainedError, match="directionality"):
        B.BiLSTMService.load()
