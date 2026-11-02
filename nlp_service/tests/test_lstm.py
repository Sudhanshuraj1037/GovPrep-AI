"""Phase 4 tests: PyTorch LSTM text classifier.

All data is SYNTHETIC (see conftest): it exercises the code and proves properties
(determinism, no leakage, correct metrics, safe checkpoints). Nothing computed on it
is a result about real performance. Skipped automatically if PyTorch is not installed.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from app import data as D  # noqa: E402
from app import lstm as L  # noqa: E402
from app import word2vec as W  # noqa: E402
from app.preprocessing import TextPreprocessor  # noqa: E402
from app.registry import model_status  # noqa: E402
from tests.conftest import TINY_LSTM, make_toy_frame  # noqa: E402


def tiny(**kw):
    return L.LSTMConfig(**{**TINY_LSTM, **kw})


def _out():
    return L.artifacts_dir()


def _ckpt(target="subject"):
    return torch.load(_out() / f"{target}.pt", map_location="cpu", weights_only=True)


def _append_rows(split, text, n=10):
    p = D.processed_path(split)
    df = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=D.DATA_COLUMNS)
    extra = pd.DataFrame([{"text": text, "subject": "History", "topic": "", "difficulty": ""}] * n)
    pd.concat([df, extra], ignore_index=True).to_csv(p, index=False)


@pytest.fixture
def prepared(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    return toy_csv


# ============================================================ vocabulary
def test_vocab_specials_frequency_order_and_caps():
    docs = [["a", "b", "b"], ["b", "c", "c", "c"], ["d"]]       # b=3 c=3 a=1 d=1
    assert L.build_vocab(docs, 1, 100) == ["<pad>", "<unk>", "b", "c", "a", "d"]   # ties alphabetical
    assert L.build_vocab(docs, 2, 100) == ["<pad>", "<unk>", "b", "c"]            # min_freq
    assert L.build_vocab(docs, 1, 4) == ["<pad>", "<unk>", "b", "c"]              # max_vocab incl. specials
    assert L.build_vocab(docs, 1, 3) == ["<pad>", "<unk>", "b"]
    assert L.build_vocab(list(reversed(docs)), 1, 100) == L.build_vocab(docs, 1, 100)  # order-independent
    assert (L.PAD_IDX, L.UNK_IDX) == (0, 1)


# ============================================================ encoding
def test_encode_pad_truncate_unk_and_stats():
    index = {w: i for i, w in enumerate(["<pad>", "<unk>", "a", "b"])}
    ids, lens, st = L.encode_docs([["a", "b", "zzz"], [], ["b"] * 10], index, max_len=4)
    assert ids.dtype == torch.int64 and tuple(ids.shape) == (3, 4)
    assert ids.tolist() == [[2, 3, 1, 0], [1, 0, 0, 0], [3, 3, 3, 3]]    # OOV->unk, empty->[unk], pad=0
    assert lens.tolist() == [3, 1, 4]
    assert st["empty_after_preprocessing"] == 1 and st["truncated_rows"] == 1
    assert st["tokens"] == 13 and st["unk_rate"] == pytest.approx(1 / 13)


def test_truncation_keeps_the_first_tokens():
    index = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
    ids, lens, _ = L.encode_docs([["a", "b", "a", "b", "b"]], index, max_len=3)
    assert ids.tolist() == [[2, 3, 2]] and lens.tolist() == [3]


def test_balanced_class_weights_match_sklearn():
    counts = [10, 30, 60]
    got = L.balanced_class_weights(counts).numpy()
    ref = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.repeat([0, 1, 2], counts))
    assert np.allclose(got, ref, rtol=1e-6)


# ============================================================ model
def _model(**kw):
    cfg = dict(vocab_size=30, embedding_dim=8, hidden_dim=12, num_layers=1, num_classes=4, dropout=0.5)
    cfg.update(kw)
    torch.manual_seed(0)
    return L.LSTMTextClassifier(**cfg)


def test_forward_shape_and_parameter_layout():
    m = _model().eval()
    ids = torch.randint(2, 30, (5, 7))
    out = m(ids, torch.tensor([7, 5, 3, 2, 1]))
    assert tuple(out.shape) == (5, 4)
    assert m.embedding.weight.shape == (30, 8) and m.fc.out_features == 4
    assert isinstance(m.lstm, torch.nn.LSTM) and m.lstm.num_layers == 1 and not m.lstm.bidirectional


def test_padding_content_never_affects_the_output():
    """Packed sequences: whatever sits in the padded positions must not change the logits."""
    m = _model().eval()
    ids = torch.randint(2, 30, (4, 9))
    lens = torch.tensor([9, 6, 4, 2])
    base = m(ids, lens)
    garbage = ids.clone()
    for i, n in enumerate(lens.tolist()):
        garbage[i, n:] = torch.randint(2, 30, (9 - n,))
    assert torch.allclose(base, m(garbage, lens), atol=1e-6)


def test_each_sequence_is_independent_of_its_batch():
    m = _model().eval()
    ids = torch.randint(2, 30, (6, 8))
    lens = torch.tensor([8, 7, 5, 4, 3, 1])
    batch = m(ids, lens)
    for i in range(6):
        single = m(ids[i : i + 1, : lens[i]], lens[i : i + 1])
        assert torch.allclose(batch[i : i + 1], single, atol=1e-5)


def test_dropout_only_active_in_training_mode():
    ids, lens = torch.randint(2, 30, (4, 6)), torch.full((4,), 6)
    m = _model(dropout=0.5)
    m.eval()
    assert torch.equal(m(ids, lens), m(ids, lens))
    m.train()
    torch.manual_seed(1)
    a = m(ids, lens)
    torch.manual_seed(2)
    b = m(ids, lens)
    assert not torch.allclose(a, b)
    m0 = _model(dropout=0.0)
    m0.train()
    assert torch.equal(m0(ids, lens), m0(ids, lens))


def test_stacked_lstm_and_gradients():
    m = _model(num_layers=2)
    assert m.lstm.num_layers == 2 and m.lstm.dropout == 0.5
    assert _model(num_layers=1).lstm.dropout == 0.0     # inter-layer dropout only when stacked
    ids, lens = torch.randint(2, 30, (4, 6)), torch.full((4,), 6)
    torch.nn.functional.cross_entropy(m(ids, lens), torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.count_nonzero(m.embedding.weight.grad[L.PAD_IDX]) == 0   # padding row never learns


# ============================================================ config
@pytest.mark.parametrize("bad", [
    dict(embedding_dim=0), dict(hidden_dim=0), dict(num_layers=0), dict(max_len=0), dict(min_freq=0),
    dict(batch_size=0), dict(epochs=0), dict(max_vocab=2), dict(dropout=1.0), dict(dropout=-0.1),
    dict(lr=0), dict(weight_decay=-1), dict(grad_clip=-1), dict(patience=-1),
    dict(init_embeddings="glove"), dict(w2v_variant="x"), dict(device="tpu"),
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        L.LSTMConfig(**bad)


def test_default_config_is_cpu_friendly_and_roundtrips():
    cfg = L.LSTMConfig()
    assert cfg.hidden_dim <= 128 and cfg.num_layers == 1 and cfg.device == "auto"
    assert L.LSTMConfig(**cfg.to_dict()) == cfg


# ============================================================ not trained
def test_no_data_means_not_trained_and_nothing_written(env):
    with pytest.raises(D.DataNotFoundError):
        L.train_all(cfg=tiny())
    assert not _out().exists() or not any(_out().iterdir())
    with pytest.raises(L.LSTMNotTrainedError, match="not trained yet"):
        L.LSTMService.load()


def test_cli_reports_not_trained_yet(env, capsys):
    from training.train_lstm import main

    assert main(["--epochs", "1"]) == 2
    assert "NOT TRAINED YET" in capsys.readouterr().err
    assert not _out().exists() or not any(_out().iterdir())


def test_cli_without_pytorch_exits_with_clear_message(env, capsys, monkeypatch):
    from training.train_lstm import main

    monkeypatch.setitem(sys.modules, "app.lstm", None)       # makes `import app.lstm` raise ImportError
    assert main([]) == 3
    assert "PyTorch is not installed" in capsys.readouterr().err


def test_cli_rejects_invalid_setting(env, capsys):
    from training.train_lstm import main

    assert main(["--dropout", "1.5"]) == 2
    assert "invalid setting" in capsys.readouterr().err


def test_nothing_trainable_writes_nothing(env, toy_csv):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    with pytest.raises(L.LSTMTrainingError, match="No target could be trained"):
        L.train_all(cfg=tiny(epochs=2), targets=("topic",))
    assert not _out().exists() or not any(_out().iterdir())


def test_invalid_targets_rejected(prepared):
    with pytest.raises(ValueError):
        L.train_all(cfg=tiny(), targets=("subject", "colour"))
    with pytest.raises(ValueError):
        L.train_all(cfg=tiny(), targets=())


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present on this machine")
def test_cuda_requested_but_unavailable_is_a_clear_error(prepared):
    with pytest.raises(L.LSTMTrainingError, match="CUDA is not available"):
        L.train_all(cfg=tiny(device="cuda"))
    assert L.resolve_device("auto").type == "cpu"


# ============================================================ artifacts
def test_artifacts_and_metrics_structure(lstm_trained):
    m = lstm_trained
    for t in ("subject", "topic", "difficulty"):
        assert (_out() / f"{t}.pt").is_file()
    assert (_out() / "metrics.json").is_file()
    assert m["model"] == "lstm" and m["seed"] == 42
    assert m["framework"]["torch"] == torch.__version__ and m["framework"]["device"] == "cpu"
    assert m["hyperparameters"] == tiny().to_dict()
    assert m["vocabulary"]["built_from"] == "public_train"
    e = m["targets"]["subject"]
    assert e["status"] == "trained" and e["labels"] == sorted(e["labels"])
    assert set(e["splits"]) == {"val", "test"}                       # no held-out file present
    assert m["data"]["pdf_heldout_evaluated"] is False
    assert e["training"]["model_selection"] == "best validation macro-F1"
    assert e["training"]["epochs_run"] == len(e["training"]["history"])
    assert e["model"]["trainable_parameters"] > 0
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                "precision_weighted", "recall_weighted", "confusion_matrix", "per_class"):
        assert key in e["splits"]["test"]


def test_split_files_and_seed_recorded_for_comparison(lstm_trained):
    m = lstm_trained
    for name in ("public_train", "public_val", "public_test"):
        assert m["data"]["split_files_sha256"][name] == D.file_sha256(D.processed_path(name))
    assert m["data"]["split_manifest"]["split"]["seed"] == 42
    assert m["data"]["rows"]["train"] == len(D.load_split("public_train"))


def test_checkpoint_is_safe_plain_data(lstm_trained):
    ck = _ckpt()
    assert set(ck) >= {"format_version", "arch", "target", "model_config", "labels", "vocab",
                       "max_len", "preprocess_config", "seed", "state_dict"}
    assert ck["arch"] == "lstm" and ck["target"] == "subject" and ck["vocab"][:2] == ["<pad>", "<unk>"]
    assert all(isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in ck["state_dict"].values())
    assert ck["model_config"]["vocab_size"] == len(ck["vocab"])


def test_vocab_and_encoding_stats_recomputed_independently(lstm_trained):
    m = lstm_trained
    ck = _ckpt()
    pre = TextPreprocessor(L.default_preprocess_config())
    train_tokens = [pre.tokens(t) for t in D.load_split("public_train")["text"]]
    assert ck["vocab"] == L.build_vocab(train_tokens, 2, 20_000)
    vocab = set(ck["vocab"])
    toks = [t for x in D.load_split("public_val")["text"] for t in pre.tokens(x)]
    st = m["data"]["encoding"]["val"]
    assert st["tokens"] == len(toks)
    assert st["unk_rate"] == pytest.approx(sum(t not in vocab for t in toks) / len(toks))


def test_class_weights_recorded(lstm_trained):
    train = D.load_split("public_train")
    counts = train["subject"].value_counts().sort_index().tolist()
    recorded = lstm_trained["targets"]["subject"]["training"]["class_weights"]
    assert np.allclose(recorded, L.balanced_class_weights(counts).numpy(), rtol=1e-6)


# ============================================================ metrics are real
@pytest.mark.parametrize("split", ["val", "test"])
def test_reported_metrics_match_independent_recomputation(lstm_trained, split):
    """Reload the SAVED checkpoint, score the SAVED split yourself, and compare every
    reported number. This also proves save -> load reproduces the trained model exactly."""
    svc = L.LSTMService.load()
    df = D.load_split(f"public_{split}")
    preds = svc.predict(df["text"].tolist(), top_k=1)
    y_pred = [p["predictions"]["subject"]["label"] for p in preds]
    assert None not in y_pred
    y_true = df["subject"].tolist()
    rep = lstm_trained["targets"]["subject"]["splits"][split]
    labels = lstm_trained["targets"]["subject"]["labels"]

    assert rep["n"] == len(df)
    assert rep["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        assert rep[f"precision_{avg}"] == pytest.approx(p)
        assert rep[f"recall_{avg}"] == pytest.approx(r)
        assert rep[f"f1_{avg}"] == pytest.approx(f)
    assert rep["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = np.array(rep["confusion_matrix"]["matrix"])
    assert rep["confusion_matrix"]["labels"] == labels
    assert (cm == confusion_matrix(y_true, y_pred, labels=labels)).all() and cm.sum() == len(df)
    maj = D.load_split("public_train")["subject"].value_counts().idxmax()
    assert rep["majority_class_baseline"]["class"] == maj


def test_the_model_learns_the_synthetic_structure(lstm_trained):
    """Sanity check of the whole training chain on a task with known structure
    (chance is 0.25). Not a performance claim."""
    assert lstm_trained["targets"]["subject"]["splits"]["test"]["accuracy"] > 0.9


# ============================================================ training protocol
def test_best_epoch_is_selected_on_validation_and_restored(lstm_trained):
    t = lstm_trained["targets"]["subject"]["training"]
    f1s = [h["val_f1_macro"] for h in t["history"]]
    assert t["best_epoch"] == 1 + int(np.argmax(f1s))             # first epoch with the maximum
    svc = L.LSTMService.load()                                     # the restored best weights
    val = D.load_split("public_val")
    pred = [p["predictions"]["subject"]["label"] for p in svc.predict(val["text"].tolist(), top_k=1)]
    labels = lstm_trained["targets"]["subject"]["labels"]
    got = f1_score(val["subject"], pred, labels=labels, average="macro", zero_division=0)
    assert got == pytest.approx(f1s[t["best_epoch"] - 1])
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in t["history"])
    assert t["history"][-1]["train_loss"] < t["history"][0]["train_loss"]   # it is actually optimising


def test_restored_weights_are_exactly_those_of_the_best_epoch(prepared):
    """Validation F1 can plateau, so equal F1 does not prove the right weights were restored.
    Exact check: training is deterministic, so a run capped at the best epoch ends with
    bit-identical weights to the best-epoch weights restored by the full run."""
    full = L.train_all(cfg=tiny(epochs=40, patience=4), targets=("subject",))
    t = full["targets"]["subject"]["training"]
    assert t["stopped_early"] and t["best_epoch"] < t["epochs_run"]        # precondition: best != last
    restored = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    L.train_all(cfg=tiny(epochs=t["best_epoch"], patience=0), targets=("subject",))
    capped = _ckpt()["state_dict"]
    assert all(torch.equal(restored[k], capped[k]) for k in restored)


def test_early_stopping_and_disabling_it(prepared):
    m = L.train_all(cfg=tiny(epochs=60, patience=2), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is True and t["epochs_run"] == t["best_epoch"] + 2 < 60
    m = L.train_all(cfg=tiny(epochs=6, patience=0), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is False and t["epochs_run"] == 6


def test_training_is_reproducible_and_seed_dependent(prepared):
    cfg = tiny(epochs=6, patience=0)
    a = L.train_all(cfg=cfg, targets=("subject",), seed=42)
    wa = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    b = L.train_all(cfg=cfg, targets=("subject",), seed=42)
    wb = _ckpt()["state_dict"]
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert a["targets"]["subject"]["splits"]["test"]["confusion_matrix"] == \
        b["targets"]["subject"]["splits"]["test"]["confusion_matrix"]
    L.train_all(cfg=cfg, targets=("subject",), seed=7)
    assert not all(torch.equal(wa[k], _ckpt()["state_dict"][k]) for k in wa)


def test_each_target_is_independent_of_the_others(prepared):
    cfg = tiny(epochs=5, patience=0)
    L.train_all(cfg=cfg, targets=("subject",))
    alone = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    L.train_all(cfg=cfg, targets=("subject", "difficulty"))
    assert all(torch.equal(alone[k], _ckpt()["state_dict"][k]) for k in alone)


def test_test_and_heldout_data_never_influence_training(prepared, env):
    """Replace the TEST and PDF held-out data completely: weights, vocabulary, chosen epoch
    and history must be identical; only the reported test numbers may change."""
    cfg = tiny(epochs=8, patience=0)
    a = L.train_all(cfg=cfg, targets=("subject",))
    w_a = {k: v.clone() for k, v in _ckpt()["state_dict"].items()}
    vocab_a = _ckpt()["vocab"]

    test = D.load_split("public_test")
    test.assign(subject=test["subject"].sample(frac=1, random_state=1).values,
                text=test["text"].str[::-1]).to_csv(D.processed_path("public_test"), index=False)
    make_toy_frame(8, seed=99).to_csv(D.processed_path("pdf_heldout"), index=False)
    b = L.train_all(cfg=cfg, targets=("subject",))

    assert _ckpt()["vocab"] == vocab_a
    assert all(torch.equal(w_a[k], _ckpt()["state_dict"][k]) for k in w_a)
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert a["targets"]["subject"]["training"]["best_epoch"] == b["targets"]["subject"]["training"]["best_epoch"]
    assert a["targets"]["subject"]["splits"]["test"]["accuracy"] != b["targets"]["subject"]["splits"]["test"]["accuracy"]
    assert "pdf_heldout" in b["targets"]["subject"]["splits"]


def test_vocabulary_comes_from_train_only(prepared):
    for split in ("public_val", "public_test", "pdf_heldout"):
        _append_rows(split, "zzsentinel zzsentinel zzsentinel zzsentinel history")
    L.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert "zzsentinel" not in _ckpt()["vocab"]


def test_sentinel_control_enters_vocab_when_in_train(prepared):
    _append_rows("public_train", "zzsentinel zzsentinel zzsentinel zzsentinel history")
    L.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert "zzsentinel" in _ckpt()["vocab"]


def test_heldout_evaluated_only_when_present_with_unseen_labels_as_errors(prepared, env):
    m = L.train_all(cfg=tiny(epochs=3), targets=("subject",))
    assert "pdf_heldout" not in m["targets"]["subject"]["splits"]
    held = make_toy_frame(6, seed=12)
    held.loc[:2, "subject"] = "Economics"                  # a class the model never saw
    held.to_csv(D.processed_path("pdf_heldout"), index=False)
    m = L.train_all(cfg=tiny(epochs=3), targets=("subject",))
    r = m["targets"]["subject"]["splits"]["pdf_heldout"]
    assert m["data"]["pdf_heldout_evaluated"] is True and r["n"] == len(held)
    assert r["unseen_label_rows"] == 3 and "Economics" in r["confusion_matrix"]["labels"]
    assert r["per_class"]["Economics"]["recall"] == 0.0
    assert m["data"]["split_files_sha256"]["pdf_heldout"] == D.file_sha256(D.processed_path("pdf_heldout"))


def test_target_without_labels_is_skipped_with_reason(env):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    m = L.train_all(cfg=tiny(epochs=2))
    assert m["targets"]["topic"]["status"] == "skipped"
    assert "needs >= 2 classes" in m["targets"]["topic"]["reason"]
    assert m["targets"]["subject"]["status"] == "trained"
    assert not (_out() / "topic.pt").exists()


def test_no_validation_rows_falls_back_to_last_epoch(prepared):
    val = D.load_split("public_val")
    val.assign(difficulty="").to_csv(D.processed_path("public_val"), index=False)
    m = L.train_all(cfg=tiny(epochs=4, patience=2), targets=("difficulty",))
    t = m["targets"]["difficulty"]["training"]
    assert t["model_selection"].startswith("last epoch") and t["epochs_run"] == 4 and t["best_epoch"] == 4
    assert "val_f1_macro" not in t["history"][0] and "val" not in m["targets"]["difficulty"]["splits"]


def test_stale_checkpoints_removed_on_retrain(prepared):
    L.train_all(cfg=tiny(epochs=2))
    assert (_out() / "topic.pt").exists()
    L.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "topic.pt").exists()


def test_marker_is_written_last(prepared, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(L, "_write_json_atomic", boom)
    with pytest.raises(OSError):
        L.train_all(cfg=tiny(epochs=2), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "metrics.json").exists()
    st = next(m for m in model_status() if m["name"] == "lstm")
    assert st["trained"] is False and st["targets_trained"] == []


def test_progress_callback_reports_real_epochs(prepared):
    seen = []
    L.train_all(cfg=tiny(epochs=3, patience=0), targets=("subject",), on_epoch=lambda t, r: seen.append((t, r)))
    assert [r["epoch"] for _, r in seen] == [1, 2, 3] and {t for t, _ in seen} == {"subject"}
    assert all("val_f1_macro" in r for _, r in seen)


# ============================================================ preprocessing config
def test_preprocessing_defaults_keep_stopwords_and_are_saved(lstm_trained):
    cfg = lstm_trained["preprocessing"]
    assert cfg["remove_stopwords"] is False and cfg["normalizer"] == "none"
    assert _ckpt()["preprocess_config"] == cfg
    assert "the" in _ckpt()["vocab"] and "following" in _ckpt()["vocab"]    # stop-words stay in the vocabulary


def test_stopword_removal_option_changes_vocabulary(prepared):
    from app.preprocessing import PreprocessConfig

    L.train_all(cfg=tiny(epochs=2), targets=("subject",),
                preprocess_cfg=PreprocessConfig(normalizer="none", remove_stopwords=True))
    assert "the" not in _ckpt()["vocab"] and "river" in _ckpt()["vocab"]


# ============================================================ Word2Vec init
def test_word2vec_embeddings_initialise_and_freeze(w2v_trained):
    m = L.train_all(cfg=tiny(epochs=3, patience=0, init_embeddings="word2vec", freeze_embeddings=True),
                    targets=("subject",))
    emb = m["embeddings"]
    assert emb["init"] == "word2vec" and emb["variant"] == "skipgram" and emb["dimension"] == 50
    assert 0 < emb["coverage"] < 1                                   # stop-words are not in Word2Vec
    assert emb["word2vec_corpus_sha256"] == w2v_trained["corpus"]["sha256"]
    ck = _ckpt()
    assert ck["model_config"]["embedding_dim"] == 50                 # follows the Word2Vec dimension
    weights = ck["state_dict"]["embedding.weight"]
    wv = W.EmbeddingService.load().vectors["skipgram"]
    covered = 0
    for i, word in enumerate(ck["vocab"]):
        if i > 1 and word in wv.key_to_index:
            assert np.allclose(weights[i].numpy(), wv[word])         # frozen: still the Word2Vec vector
            covered += 1
    assert covered == emb["words_covered"] > 0
    assert torch.count_nonzero(weights[L.PAD_IDX]) == 0


def test_frozen_random_embeddings_do_not_change(prepared):
    L.train_all(cfg=tiny(epochs=4, patience=0, freeze_embeddings=True), targets=("subject",))
    ck = _ckpt()
    L._seed_everything(42)
    fresh = L.LSTMTextClassifier(**ck["model_config"])
    assert torch.equal(fresh.embedding.weight.detach(), ck["state_dict"]["embedding.weight"])


def test_unfrozen_embeddings_do_change(prepared):
    L.train_all(cfg=tiny(epochs=4, patience=0), targets=("subject",))
    ck = _ckpt()
    L._seed_everything(42)
    fresh = L.LSTMTextClassifier(**ck["model_config"])
    assert not torch.equal(fresh.embedding.weight.detach(), ck["state_dict"]["embedding.weight"])


def test_word2vec_init_refused_without_a_trained_model(prepared):
    with pytest.raises(L.LSTMTrainingError, match="Word2Vec"):
        L.train_all(cfg=tiny(init_embeddings="word2vec"), targets=("subject",))
    assert not _out().exists() or not any(_out().iterdir())


def test_word2vec_init_refused_if_trained_on_a_different_split(w2v_trained):
    """Leak guard: embeddings from a different public_train.csv could carry val/test text."""
    train = D.load_split("public_train")
    pd.concat([train, train.head(1).assign(text="a brand new training sentence")]).to_csv(
        D.processed_path("public_train"), index=False)
    with pytest.raises(L.LSTMTrainingError, match="different public_train"):
        L.train_all(cfg=tiny(init_embeddings="word2vec"), targets=("subject",))


# ============================================================ inference
def test_inference_shapes_probabilities_and_order(lstm_trained):
    svc = L.LSTMService.load()
    assert svc.targets == ["difficulty", "subject", "topic"]
    out = svc.predict(["the river delta and monsoon climate", "an ancient empire battle"], top_k=10)
    s = out[0]["predictions"]["subject"]
    assert s["label"] == "Geography" and out[1]["predictions"]["subject"]["label"] == "History"
    assert sum(t["probability"] for t in s["top"]) == pytest.approx(1.0, abs=1e-5)
    probs = [t["probability"] for t in s["top"]]
    assert probs == sorted(probs, reverse=True) and s["top"][0]["label"] == s["label"]
    assert s["confidence"] == pytest.approx(probs[0]) and 0 <= s["confidence"] <= 1
    assert len(svc.predict(["river delta"], top_k=2)[0]["predictions"]["subject"]["top"]) == 2


def test_batched_and_single_predictions_agree(lstm_trained):
    svc = L.LSTMService.load()
    texts = ["river delta", "an ancient empire battle dynasty", "atom", "the constitution article of the federal parliament"]
    batched = svc.predict(texts, top_k=4)
    for text, b in zip(texts, batched):
        single = svc.predict([text], top_k=4)[0]
        for target in svc.targets:
            assert single["predictions"][target]["label"] == b["predictions"][target]["label"]
            assert single["predictions"][target]["confidence"] == pytest.approx(
                b["predictions"][target]["confidence"], abs=1e-5)


def test_unreadable_text_gets_none_and_a_reason(lstm_trained):
    svc = L.LSTMService.load()
    out = svc.predict(["zzz qqq xxx", "   ", "!!!", "river delta"])
    for i in (0, 1, 2):
        p = out[i]["predictions"]["subject"]
        assert p["label"] is None and p["confidence"] is None and p["top"] == []
    assert "out of the model's vocabulary" in out[0]["predictions"]["subject"]["warning"]
    assert "no usable tokens" in out[1]["predictions"]["subject"]["warning"]
    assert out[3]["predictions"]["subject"]["label"] == "Geography"


def test_long_text_is_truncated_not_rejected(lstm_trained):
    svc = L.LSTMService.load()
    long_text = "river delta monsoon " * 200                       # far beyond max_len
    a = svc.predict([long_text])[0]["predictions"]["subject"]
    b = svc.predict([long_text])[0]["predictions"]["subject"]
    assert a["label"] == "Geography" and a == b


def test_inference_leaves_the_model_in_eval_mode(lstm_trained):
    svc = L.LSTMService.load()
    assert all(not m["model"].training for m in svc.models.values())
    x = svc.predict(["river delta"], top_k=3)
    assert x == svc.predict(["river delta"], top_k=3)               # deterministic (no dropout at inference)


# ============================================================ loading errors
def test_missing_checkpoint_is_a_clear_error(lstm_trained):
    (_out() / "subject.pt").unlink()
    with pytest.raises(L.LSTMNotTrainedError, match="missing"):
        L.LSTMService.load()


def test_corrupt_checkpoint_is_a_clear_error(lstm_trained):
    (_out() / "subject.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(L.LSTMNotTrainedError, match="could not load checkpoint"):
        L.LSTMService.load()


class _Evil:
    def __reduce__(self):                                           # would run code if unpickled
        return (print, ("PWNED",))


def test_loading_never_executes_pickled_code(lstm_trained, capsys):
    ck = _ckpt()
    ck["evil"] = _Evil()
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(L.LSTMNotTrainedError, match="could not load checkpoint"):
        L.LSTMService.load()
    assert "PWNED" not in capsys.readouterr().out


def test_checkpoint_for_the_wrong_target_is_rejected(lstm_trained):
    ck = _ckpt("difficulty")
    torch.save(ck, _out() / "subject.pt")                           # difficulty model saved as subject
    with pytest.raises(L.LSTMNotTrainedError, match="could not load checkpoint"):
        L.LSTMService.load()


def test_metrics_without_checkpoints_is_not_loadable(lstm_trained):
    for f in _out().glob("*.pt"):
        f.unlink()
    with pytest.raises(L.LSTMNotTrainedError):
        L.LSTMService.load()
