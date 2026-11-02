"""Phase 6 tests: BERT fine-tuning (Hugging Face Transformers + PyTorch).

IMPORTANT: Hugging Face is not reachable from the development sandbox, so NO real pretrained BERT
is downloaded here. Every test fine-tunes a TINY stand-in (random weights, 2 layers, hidden size 32)
that conftest builds locally and saves as a directory; the code under test loads it through the
very same `from_pretrained` path a real checkpoint uses. These tests therefore prove the machinery
(tokenizer, attention masks, fine-tuning loop, checkpoints, no leakage, metrics, contract) and say
NOTHING about real BERT quality. Skipped automatically if torch or transformers is missing.
"""
import math
import shutil
import socket
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel  # noqa: E402

from app import bert as BT  # noqa: E402
from app import data as D  # noqa: E402
from app import lstm as L  # noqa: E402
from app import model_metrics as MM  # noqa: E402
from app.registry import model_status  # noqa: E402
from tests.conftest import TINY_BERT, make_toy_frame, tiny_bert_vocab  # noqa: E402

VOCAB = {t: i for i, t in enumerate(tiny_bert_vocab())}
CLS, SEP, PAD, UNK = VOCAB["[CLS]"], VOCAB["[SEP]"], VOCAB["[PAD]"], VOCAB["[UNK]"]


@pytest.fixture
def mk(tiny_bert_dir):
    def make(**kw):
        return BT.BertFineTuneConfig(**{"pretrained": str(tiny_bert_dir), **TINY_BERT, **kw})
    return make


@pytest.fixture(scope="module")
def tok(tiny_bert_dir):
    return AutoTokenizer.from_pretrained(str(tiny_bert_dir))


@pytest.fixture
def prepared(env, toy_csv):
    D.prepare_splits(toy_csv, seed=42)
    return toy_csv


def _out():
    return BT.artifacts_dir()


def _ckpt(target="subject"):
    return torch.load(_out() / f"{target}.pt", map_location="cpu", weights_only=True)


def _state(target="subject"):
    return {k: v.clone() for k, v in _ckpt(target)["state_dict"].items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _append_rows(split, text, n=10):
    p = D.processed_path(split)
    df = pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame(columns=D.DATA_COLUMNS)
    extra = pd.DataFrame([{"text": text, "subject": "History", "topic": "", "difficulty": ""}] * n)
    pd.concat([df, extra], ignore_index=True).to_csv(p, index=False)


# ============================================================ tokenizer + encoding
def test_encoding_adds_special_tokens_lowercases_and_marks_unknown_words(tok):
    ids, stats, reasons = BT.encode_texts(tok, ["River DELTA zzz"], max_len=32)
    assert ids == [[CLS, VOCAB["river"], VOCAB["delta"], UNK, SEP]]          # lower-cased by the tokenizer
    assert reasons == [None]
    assert stats["tokens"] == 3 and stats["unk_rate"] == pytest.approx(1 / 3) and stats["rows"] == 1


def test_truncation_keeps_the_special_tokens_and_is_counted(tok):
    ids, stats, _ = BT.encode_texts(tok, [" ".join(["river"] * 20), "river"], max_len=8)
    assert len(ids[0]) == 8 and ids[0][0] == CLS and ids[0][-1] == SEP
    assert ids[0][1:-1] == [VOCAB["river"]] * 6
    assert ids[1] == [CLS, VOCAB["river"], SEP]
    assert stats["truncated_rows"] == 1


def test_unreadable_texts_are_flagged_but_still_encoded(tok):
    ids, stats, reasons = BT.encode_texts(tok, ["", "   ", "https://x.com/a", "zzz qqq", "river zzz"], max_len=16)
    assert reasons[0] == reasons[1] == reasons[2] == "no usable tokens after preprocessing"
    assert reasons[3] == "no known words: every token is unknown to the tokenizer"
    assert reasons[4] is None
    assert ids[0] == ids[1] == ids[2] == [CLS, SEP]                          # every row can still be scored
    assert stats["empty_after_cleaning"] == 3 and stats["all_unknown_rows"] == 1
    assert BT.encode_texts(tok, [], 16)[0] == []


def test_cleaning_removes_noise_but_keeps_what_a_pretrained_model_needs():
    c = BT.clean_for_bert
    assert c("Visit https://x.com/a me@x.org river Page 3 of 70 ") == "Visit river"
    assert c("Ashoka's  edicts\tdidn't\x00 mention") == "Ashoka's edicts didn't mention"   # case + apostrophes kept
    assert c("ＲＩＶＥＲ ﬁne") == "RIVER fine"                                            # Unicode NFKC
    assert c(None) == "" and c(3.5) == "" and c(float("nan")) == ""


def test_clean_text_can_be_disabled(tok):
    raw, _, _ = BT.encode_texts(tok, ["https://x.com river"], 32, clean=False)
    cleaned, _, _ = BT.encode_texts(tok, ["https://x.com river"], 32, clean=True)
    assert len(raw[0]) > len(cleaned[0]) == 3                                # the URL pieces stay when not cleaned


def test_collate_pads_and_builds_the_mask_from_true_lengths():
    ids, mask = BT.collate([[2, 5, 3], [2, 3]], pad_id=5)                     # pad id equals a real token id
    assert ids.tolist() == [[2, 5, 3], [2, 3, 5]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]                           # the mask does not depend on the pad id
    assert ids.dtype == mask.dtype == torch.long


def test_attention_mask_makes_padding_invisible(tiny_bert_dir):
    """A sequence scored inside a padded batch must give the same logits as scored alone; without
    the mask it would not."""
    m = AutoModelForSequenceClassification.from_pretrained(str(tiny_bert_dir), num_labels=3).eval()
    short, long_ = [CLS, VOCAB["river"], SEP], [CLS] + [VOCAB["delta"]] * 9 + [SEP]
    ids, mask = BT.collate([short, long_], PAD)
    with torch.no_grad():
        alone = m(input_ids=torch.tensor([short])).logits[0]
        masked = m(input_ids=ids, attention_mask=mask).logits[0]
        unmasked = m(input_ids=ids).logits[0]
    assert torch.allclose(alone, masked, atol=1e-5)
    assert not torch.allclose(alone, unmasked, atol=1e-6)


def test_learning_rate_schedule_closed_form():
    f = BT.lr_scale
    assert f(0, 100, 10) == 0.0 and f(5, 100, 10) == 0.5 and f(10, 100, 10) == 1.0
    assert f(55, 100, 10) == pytest.approx(0.5) and f(100, 100, 10) == 0.0 and f(150, 100, 10) == 0.0
    assert f(0, 100, 0) == 1.0 and f(50, 100, 0) == pytest.approx(0.5)
    vals = [f(s, 100, 10) for s in range(0, 101)]
    assert max(vals) == 1.0 and min(vals) == 0.0 and all(0.0 <= v <= 1.0 for v in vals)
    assert vals[:11] == sorted(vals[:11]) and vals[10:] == sorted(vals[10:], reverse=True)


# ============================================================ config
@pytest.mark.parametrize("bad", [
    dict(pretrained=""), dict(pretrained="  "), dict(max_len=3), dict(max_len=513), dict(batch_size=0),
    dict(epochs=0), dict(lr=0), dict(weight_decay=-1), dict(grad_clip=-1), dict(patience=-1),
    dict(warmup_ratio=1.0), dict(warmup_ratio=-0.1), dict(device="tpu"),
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        BT.BertFineTuneConfig(**bad)


def test_default_config_and_roundtrip():
    cfg = BT.BertFineTuneConfig()
    assert cfg.pretrained == "bert-base-uncased" and cfg.lr == 2e-5 and cfg.max_len == 128
    assert cfg.freeze_encoder is False and cfg.class_weights is True and cfg.device == "auto"
    assert BT.BertFineTuneConfig(**cfg.to_dict()) == cfg


# ============================================================ not trained / bad setup
def test_no_data_means_not_trained_and_nothing_written(env, mk):
    with pytest.raises(D.DataNotFoundError):
        BT.train_all(cfg=mk())
    assert not _out().exists() or not any(_out().iterdir())
    with pytest.raises(BT.BertNotTrainedError, match="BERT not trained yet"):
        BT.BertService.load()


def test_cli_reports_not_trained_yet(env, capsys):
    from training.train_bert import main

    assert main(["--epochs", "1"]) == 2
    assert "BERT: NOT TRAINED YET" in capsys.readouterr().err
    assert not _out().exists() or not any(_out().iterdir())


def test_cli_rejects_invalid_setting_and_missing_dependencies(env, capsys, monkeypatch):
    from training.train_bert import main

    assert main(["--max-len", "2"]) == 2 and "BERT: invalid setting" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "app.bert", None)                       # `import app.bert` raises ImportError
    assert main([]) == 3
    assert "PyTorch or Transformers is not installed" in capsys.readouterr().err


def test_cli_fine_tunes_into_its_own_folder(prepared, tiny_bert_dir, capsys):
    from training.train_bert import main

    assert main(["--pretrained", str(tiny_bert_dir), "--targets", "subject", "--epochs", "2", "--max-len", "32",
                 "--batch-size", "16", "--lr", "1e-3", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "Fine-tuning" in out and "BERT fine-tuning finished" in out and "macro-F1" in out and "transformers" in out
    assert (_out() / "subject.pt").is_file() and (_out() / "metrics.json").is_file() and (_out() / "tokenizer").is_dir()
    for other in ("lstm", "bilstm", "tfidf"):
        assert not (_out().parent / other).exists()


def test_cli_freeze_encoder_is_announced(prepared, tiny_bert_dir, capsys):
    from training.train_bert import main

    assert main(["--pretrained", str(tiny_bert_dir), "--targets", "subject", "--epochs", "1", "--max-len", "32",
                 "--freeze-encoder", "--device", "cpu"]) == 0
    assert "not full fine-tuning" in capsys.readouterr().out


def test_unavailable_pretrained_model_writes_nothing(prepared, mk, tmp_path, capsys):
    with pytest.raises(BT.BertTrainingError, match="could not load the pretrained tokenizer"):
        BT.train_all(cfg=mk(pretrained=str(tmp_path / "does-not-exist")))
    assert not _out().exists() or not any(_out().iterdir())
    from training.train_bert import main
    assert main(["--pretrained", str(tmp_path / "does-not-exist")]) == 2
    assert "BERT: NOT TRAINED YET" in capsys.readouterr().err


def test_pretrained_directory_without_a_tokenizer_is_rejected(prepared, mk, tiny_bert_dir, tmp_path):
    bare = tmp_path / "only_weights"
    bare.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy(tiny_bert_dir / name, bare / name)
    with pytest.raises(BT.BertTrainingError, match="tokenizer"):
        BT.train_all(cfg=mk(pretrained=str(bare)))
    assert not _out().exists() or not any(_out().iterdir())


def test_a_tokenizer_that_does_not_fit_the_model_is_rejected(prepared, mk, tiny_bert_dir, tmp_path):
    """A model whose embedding table is smaller than the tokenizer would index out of range."""
    import json

    small = tmp_path / "small_embeddings"
    shutil.copytree(tiny_bert_dir, small)
    cfg = json.loads((small / "config.json").read_text(encoding="utf-8"))
    cfg["vocab_size"] = 10                                                   # tokenizer has 53 tokens
    (small / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(BT.BertTrainingError, match="only has 10 embeddings"):
        BT.train_all(cfg=mk(pretrained=str(small)))
    assert not _out().exists() or not any(_out().iterdir())


def test_tokenizer_plausibility_check_directly(tok):
    BT.check_tokenizer_fits_model(tok, len(tok))                              # same size: fine
    BT.check_tokenizer_fits_model(tok, len(tok) + 5)                          # small padding is normal
    BT.check_tokenizer_fits_model(tok, None)
    with pytest.raises(ValueError, match="only has"):
        BT.check_tokenizer_fits_model(tok, len(tok) - 1)
    with pytest.raises(ValueError, match="missing or belong to another model"):
        BT.check_tokenizer_fits_model(tok, len(tok) * 3)


def test_max_len_beyond_the_model_positions_is_rejected(prepared, mk):
    with pytest.raises(BT.BertTrainingError, match="exceeds the 64 positions"):
        BT.train_all(cfg=mk(max_len=100))
    assert not _out().exists() or not any(_out().iterdir())


def test_invalid_targets_rejected(prepared, mk):
    with pytest.raises(ValueError):
        BT.train_all(cfg=mk(), targets=("subject", "colour"))
    with pytest.raises(ValueError):
        BT.train_all(cfg=mk(), targets=())


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present on this machine")
def test_cuda_requested_but_unavailable_is_a_clear_error(prepared, mk):
    with pytest.raises(BT.BertTrainingError, match="CUDA is not available"):
        BT.train_all(cfg=mk(device="cuda"))


def test_nothing_trainable_writes_nothing(env, mk):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    with pytest.raises(BT.BertTrainingError, match="No target could be trained"):
        BT.train_all(cfg=mk(epochs=1), targets=("topic",))
    assert not _out().exists() or not any(_out().iterdir())


# ============================================================ pretrained, never from scratch
def test_never_builds_the_model_from_a_config(prepared, mk, monkeypatch):
    """Training must use from_pretrained only: a from_config / for_model call would mean random weights."""
    def boom(*a, **k):
        raise AssertionError("the model must be loaded with from_pretrained, never built from a config")

    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_config", boom)
    monkeypatch.setattr(transformers.AutoModel, "from_config", boom)
    monkeypatch.setattr(transformers.AutoConfig, "for_model", boom)
    BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert (_out() / "subject.pt").is_file()


def test_pretrained_encoder_weights_are_really_loaded(prepared, mk, tiny_bert_dir):
    """With the encoder frozen, the saved encoder must equal the pretrained checkpoint exactly,
    while the new classification head must have been trained."""
    BT.train_all(cfg=mk(epochs=2, patience=0, freeze_encoder=True), targets=("subject",))
    saved = _ckpt()["state_dict"]
    pretrained = BertModel.from_pretrained(str(tiny_bert_dir)).state_dict()
    checked = 0
    for k, v in pretrained.items():
        if "position_ids" in k:
            continue
        assert torch.equal(saved["bert." + k], v), k
        checked += 1
    assert checked > 20
    L._seed_everything(42)                                                   # the head's initial values
    init = BT._load_pretrained_model(mk(), _ckpt()["labels"])
    assert not torch.equal(saved["classifier.weight"], init.classifier.weight.detach())


def test_full_fine_tuning_updates_the_encoder(prepared, mk, tiny_bert_dir):
    BT.train_all(cfg=mk(epochs=2, patience=0), targets=("subject",))
    saved = _ckpt()["state_dict"]
    pretrained = BertModel.from_pretrained(str(tiny_bert_dir)).state_dict()
    changed = [k for k, v in pretrained.items() if "position_ids" not in k and not torch.equal(saved["bert." + k], v)]
    assert len(changed) > 10


def test_parameter_counts_for_frozen_and_full_fine_tuning(prepared, mk):
    full = BT.train_all(cfg=mk(epochs=1), targets=("subject",))["targets"]["subject"]["model"]
    frozen = BT.train_all(cfg=mk(epochs=1, freeze_encoder=True), targets=("subject",))
    fm = frozen["targets"]["subject"]["model"]
    H, C = 32, 4
    assert fm["trainable_parameters"] == H * C + C                           # only the classification head
    assert full["trainable_parameters"] == full["total_parameters"] == fm["total_parameters"]
    assert fm["freeze_encoder"] is True and frozen["pretrained"]["freeze_encoder"] is True
    assert full["pretrained"].endswith("pretrained") and full["model_type"] == "bert"


# ============================================================ artifacts and metrics
def test_artifacts_and_metrics_structure(bert_trained, tiny_bert_dir):
    m = bert_trained
    for t in ("subject", "topic", "difficulty"):
        assert (_out() / f"{t}.pt").is_file()
    assert (_out() / "metrics.json").is_file() and (_out() / "tokenizer").is_dir()
    assert not [p for p in _out().parent.iterdir() if p.name != "bert"]      # nothing leaked into other folders
    assert m["model"] == "bert" and m["seed"] == 42 and "BERT" in m["description"]
    assert m["framework"]["transformers"] == transformers.__version__ and m["framework"]["device"] == "cpu"
    assert m["hyperparameters"] == BT.BertFineTuneConfig(pretrained=str(tiny_bert_dir), **TINY_BERT).to_dict()
    assert m["pretrained"]["name"] == str(tiny_bert_dir) and m["pretrained"]["model_type"] == "bert"
    assert m["preprocessing"]["stop_word_removal"] is False and m["preprocessing"]["stemming"] is False
    e = m["targets"]["subject"]
    assert e["status"] == "trained" and e["labels"] == sorted(e["labels"])
    assert set(e["splits"]) == {"val", "test"} and m["data"]["pdf_heldout_evaluated"] is False
    t = e["training"]
    assert t["model_selection"] == "best validation macro-F1" and t["epochs_run"] == len(t["history"])
    steps = math.ceil(e["n_train"] / TINY_BERT["batch_size"]) * TINY_BERT["epochs"]
    assert t["lr_schedule"] == {"total_steps": steps, "warmup_steps": math.ceil(0.1 * steps)}
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                "precision_weighted", "recall_weighted", "confusion_matrix", "per_class"):
        assert key in e["splits"]["test"]


def test_checkpoint_is_safe_plain_data(bert_trained):
    ck = _ckpt()
    assert set(ck) >= {"format_version", "arch", "target", "pretrained", "model_config", "labels", "max_len",
                       "clean_text", "seed", "state_dict"}
    assert ck["arch"] == "bert" and ck["target"] == "subject" and ck["model_config"]["model_type"] == "bert"
    assert all(isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in ck["state_dict"].values())
    assert tuple(ck["state_dict"]["classifier.weight"].shape) == (len(ck["labels"]), 32)


def test_the_tokenizer_comes_from_the_pretrained_model_not_from_our_data(bert_trained, tiny_bert_dir):
    saved = (_out() / "tokenizer" / "tokenizer.json").read_bytes()
    assert saved == (tiny_bert_dir / "tokenizer.json").read_bytes()          # byte-identical: nothing was fitted on data


def test_metrics_follow_the_shared_contract(bert_trained):
    assert MM.validate_metrics(bert_trained) == []
    rows = MM.rows_from_metrics(bert_trained, checkpoint_available=True)
    assert len(rows) == 6 and all(tuple(r) == MM.ROW_FIELDS for r in rows)
    assert all(r["model"] == "bert" and isinstance(r["trainable_parameters"], int) for r in rows)


@pytest.mark.parametrize("split", ["val", "test"])
def test_reported_metrics_match_independent_recomputation(bert_trained, split):
    """Reload the SAVED checkpoint, score the SAVED split yourself, compare every reported number.
    Also proves save -> load reproduces the fine-tuned model exactly."""
    svc = BT.BertService.load()
    df = D.load_split(f"public_{split}")
    preds = svc.predict(df["text"].tolist(), top_k=1)
    y_pred = [p["predictions"]["subject"]["label"] for p in preds]
    assert None not in y_pred
    y_true = df["subject"].tolist()
    rep = bert_trained["targets"]["subject"]["splits"][split]
    labels = bert_trained["targets"]["subject"]["labels"]
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


def test_encoding_stats_are_recomputed_independently(bert_trained, tok):
    df = D.load_split("public_val")
    body = tok([BT.clean_for_bert(t) for t in df["text"]], add_special_tokens=False, verbose=False)["input_ids"]
    n = sum(len(b) for b in body)
    st = bert_trained["data"]["encoding"]["val"]
    assert st["tokens"] == n and st["unk_rate"] == pytest.approx(sum(t == UNK for b in body for t in b) / n)
    assert st["rows"] == len(df)


def test_class_weights_recorded(bert_trained):
    counts = D.load_split("public_train")["subject"].value_counts().sort_index().tolist()
    recorded = bert_trained["targets"]["subject"]["training"]["class_weights"]
    assert np.allclose(recorded, L.balanced_class_weights(counts).numpy(), rtol=1e-6)


def test_the_fine_tuning_chain_learns_the_synthetic_structure(bert_trained):
    """Sanity check of the whole chain on a task with known structure (chance is 0.25) with a tiny
    random-weight encoder. Not a performance claim about BERT."""
    t = bert_trained["targets"]["subject"]
    assert t["splits"]["test"]["accuracy"] > 0.9
    h = t["training"]["history"]
    assert h[-1]["train_loss"] < h[0]["train_loss"]


# ============================================================ training protocol
def test_restored_checkpoint_has_the_best_epochs_validation_loss(bert_trained):
    """Recompute the saved model's validation loss independently: it must equal the logged loss of
    the BEST epoch (and differ from the last epoch's), proving the best weights were restored."""
    t = bert_trained["targets"]["subject"]["training"]
    assert t["stopped_early"] and t["best_epoch"] < t["epochs_run"]          # precondition: best != last
    svc = BT.BertService.load()
    val = D.load_split("public_val")
    labels = bert_trained["targets"]["subject"]["labels"]
    res = svc.predict(val["text"].tolist(), top_k=len(labels))
    p = [{x["label"]: x["probability"] for x in r["predictions"]["subject"]["top"]}[y]
         for r, y in zip(res, val["subject"])]
    loss = float(-np.mean(np.log(p)))
    best, last = t["history"][t["best_epoch"] - 1], t["history"][-1]
    assert loss == pytest.approx(best["val_loss"], abs=1e-4)
    assert abs(loss - last["val_loss"]) > 1e-4
    f1s = [h["val_f1_macro"] for h in t["history"]]
    assert t["best_epoch"] == 1 + int(np.argmax(f1s))


def test_early_stopping_and_disabling_it(prepared, mk):
    m = BT.train_all(cfg=mk(epochs=20, patience=2, lr=3e-3), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is True and t["epochs_run"] == t["best_epoch"] + 2 < 20
    m = BT.train_all(cfg=mk(epochs=4, patience=0), targets=("subject",))
    t = m["targets"]["subject"]["training"]
    assert t["stopped_early"] is False and t["epochs_run"] == 4


def test_training_is_reproducible_and_seed_dependent(prepared, mk):
    cfg = mk(epochs=3, patience=0)
    a = BT.train_all(cfg=cfg, targets=("subject",), seed=42)
    wa = _state()
    b = BT.train_all(cfg=cfg, targets=("subject",), seed=42)
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert _same(wa, _state())
    assert a["targets"]["subject"]["splits"]["test"]["confusion_matrix"] == \
        b["targets"]["subject"]["splits"]["test"]["confusion_matrix"]
    BT.train_all(cfg=cfg, targets=("subject",), seed=7)
    assert not _same(wa, _state())


def test_each_target_is_independent_of_the_others(prepared, mk):
    cfg = mk(epochs=2, patience=0)
    BT.train_all(cfg=cfg, targets=("subject",))
    alone = _state()
    BT.train_all(cfg=cfg, targets=("subject", "difficulty"))
    assert _same(alone, _state())


def test_test_and_heldout_data_never_influence_training(prepared, mk):
    """Replace the TEST and PDF held-out data completely: weights, chosen epoch and history must be
    identical; only the reported test numbers may change."""
    cfg = mk(epochs=3, patience=0)
    a = BT.train_all(cfg=cfg, targets=("subject",))
    w_a = _state()
    test = D.load_split("public_test")
    test.assign(subject=test["subject"].sample(frac=1, random_state=1).values,
                text=test["text"].str[::-1]).to_csv(D.processed_path("public_test"), index=False)
    make_toy_frame(8, seed=99).to_csv(D.processed_path("pdf_heldout"), index=False)
    b = BT.train_all(cfg=cfg, targets=("subject",))
    assert _same(w_a, _state())
    assert a["targets"]["subject"]["training"]["history"] == b["targets"]["subject"]["training"]["history"]
    assert a["targets"]["subject"]["training"]["best_epoch"] == b["targets"]["subject"]["training"]["best_epoch"]
    assert a["targets"]["subject"]["splits"]["test"]["accuracy"] != b["targets"]["subject"]["splits"]["test"]["accuracy"]
    assert "pdf_heldout" in b["targets"]["subject"]["splits"]


def test_validation_does_drive_selection_control(prepared, mk):
    """Control for the leakage test above: changing the VALIDATION data does change the training
    history, so 'test/held-out changes nothing' is a real finding, not an insensitive test."""
    cfg = mk(epochs=3, patience=0)
    a = BT.train_all(cfg=cfg, targets=("subject",))
    val = D.load_split("public_val")
    val.assign(subject=val["subject"].sample(frac=1, random_state=3).values).to_csv(D.processed_path("public_val"), index=False)
    b = BT.train_all(cfg=cfg, targets=("subject",))
    assert a["targets"]["subject"]["training"]["history"] != b["targets"]["subject"]["training"]["history"]


def test_no_vocabulary_is_ever_built_from_our_data(prepared, mk, tiny_bert_dir):
    for split in ("public_train", "public_val", "public_test", "pdf_heldout"):
        _append_rows(split, "zzsentinel zzsentinel zzsentinel zzsentinel history")
    BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert (_out() / "tokenizer" / "tokenizer.json").read_bytes() == (tiny_bert_dir / "tokenizer.json").read_bytes()
    assert "zzsentinel" not in (_out() / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8")


def test_heldout_evaluated_only_when_present_with_unseen_labels_as_errors(prepared, mk):
    m = BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert "pdf_heldout" not in m["targets"]["subject"]["splits"]
    held = make_toy_frame(6, seed=12)
    held.loc[:2, "subject"] = "Economics"                                    # a class the model never saw
    held.to_csv(D.processed_path("pdf_heldout"), index=False)
    m = BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    r = m["targets"]["subject"]["splits"]["pdf_heldout"]
    assert m["data"]["pdf_heldout_evaluated"] is True and r["n"] == len(held)
    assert r["unseen_label_rows"] == 3 and "Economics" in r["confusion_matrix"]["labels"]
    assert r["per_class"]["Economics"]["recall"] == 0.0
    assert MM.validate_metrics(m) == []
    assert m["data"]["split_files_sha256"]["pdf_heldout"] == D.file_sha256(D.processed_path("pdf_heldout"))


def test_target_without_enough_classes_is_skipped_with_reason(env, mk):
    path = env / "data" / "public" / "no_topic.csv"
    make_toy_frame(40, with_topic=False).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    m = BT.train_all(cfg=mk(epochs=1))
    assert m["targets"]["topic"]["status"] == "skipped" and "needs >= 2 classes" in m["targets"]["topic"]["reason"]
    assert m["targets"]["subject"]["status"] == "trained" and not (_out() / "topic.pt").exists()
    assert MM.validate_metrics(m) == []


def test_no_validation_rows_falls_back_to_last_epoch(prepared, mk):
    D.load_split("public_val").assign(difficulty="").to_csv(D.processed_path("public_val"), index=False)
    m = BT.train_all(cfg=mk(epochs=3, patience=1), targets=("difficulty",))
    t = m["targets"]["difficulty"]["training"]
    assert t["model_selection"].startswith("last epoch") and t["epochs_run"] == 3 and t["best_epoch"] == 3
    assert "val" not in m["targets"]["difficulty"]["splits"]


def test_split_files_and_seed_recorded_for_comparison(bert_trained):
    for name in ("public_train", "public_val", "public_test"):
        assert bert_trained["data"]["split_files_sha256"][name] == D.file_sha256(D.processed_path(name))
    assert bert_trained["data"]["split_manifest"]["split"]["seed"] == 42
    assert bert_trained["data"]["rows"]["train"] == len(D.load_split("public_train"))


def test_stale_checkpoints_removed_on_retrain(prepared, mk):
    BT.train_all(cfg=mk(epochs=1))
    assert (_out() / "topic.pt").exists()
    BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "topic.pt").exists()
    assert not list(_out().glob("*.tmp"))


def test_marker_is_written_last_and_health_stays_honest(prepared, mk, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(BT, "_write_json_atomic", boom)
    with pytest.raises(OSError):
        BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert (_out() / "subject.pt").exists() and not (_out() / "metrics.json").exists()
    assert not list(_out().glob("*.tmp"))
    st = next(m for m in model_status() if m["name"] == "bert")
    assert st["trained"] is False and st["targets_trained"] == []


def test_a_crash_midway_leaves_previous_artifacts_intact_and_no_staging_files(prepared, mk, monkeypatch):
    BT.train_all(cfg=mk(epochs=1), targets=("subject", "difficulty"))
    before = {p.name: p.read_bytes() for p in _out().glob("*.pt")} | {"metrics.json": (_out() / "metrics.json").read_bytes()}
    real = BT.train_target

    def crash_on_second(target, *a, **k):
        if target == "difficulty":
            raise RuntimeError("simulated crash")
        return real(target, *a, **k)

    monkeypatch.setattr(BT, "train_target", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        BT.train_all(cfg=mk(epochs=1), targets=("subject", "difficulty"))
    assert not list(_out().glob("*.tmp"))                                    # staged subject.pt.tmp was removed
    after = {p.name: p.read_bytes() for p in _out().glob("*.pt")} | {"metrics.json": (_out() / "metrics.json").read_bytes()}
    assert after == before                                                    # the previous model is untouched


def test_progress_callback_reports_real_epochs(prepared, mk):
    seen = []
    BT.train_all(cfg=mk(epochs=2, patience=0), targets=("subject",), on_epoch=lambda t, r: seen.append((t, r)))
    assert [r["epoch"] for _, r in seen] == [1, 2] and all("val_f1_macro" in r for _, r in seen)


def test_training_bert_never_touches_the_other_models_artifacts(lstm_trained, mk):
    import hashlib

    lstm_dir = L.artifacts_dir()
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in lstm_dir.iterdir()}
    BT.train_all(cfg=mk(epochs=1), targets=("subject",))
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in lstm_dir.iterdir()} == before
    status = lambda n: next(m for m in model_status() if m["name"] == n)  # noqa: E731
    assert status("lstm")["trained"] is True and status("bert")["trained"] is True
    assert status("bilstm")["trained"] is False


# ============================================================ inference
def test_inference_shapes_probabilities_and_order(bert_trained):
    svc = BT.BertService.load()
    assert svc.targets == ["difficulty", "subject", "topic"]
    out = svc.predict(["the river delta and monsoon climate", "an ancient empire battle"], top_k=10)
    s = out[0]["predictions"]["subject"]
    assert s["label"] == "Geography" and out[1]["predictions"]["subject"]["label"] == "History"
    probs = [t["probability"] for t in s["top"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-5) and probs == sorted(probs, reverse=True)
    assert s["top"][0]["label"] == s["label"] and s["confidence"] == pytest.approx(probs[0])
    assert len(svc.predict(["river delta"], top_k=2)[0]["predictions"]["subject"]["top"]) == 2


def test_batched_and_single_predictions_agree(bert_trained):
    """The attention mask must make a text's prediction independent of what it is batched with."""
    svc = BT.BertService.load()
    texts = ["river delta", "an ancient empire battle dynasty", "atom", "the constitution article of the federal parliament"]
    batched = svc.predict(texts, top_k=4)
    for text, b in zip(texts, batched):
        single = svc.predict([text], top_k=4)[0]
        for target in svc.targets:
            assert single["predictions"][target]["label"] == b["predictions"][target]["label"]
            assert single["predictions"][target]["confidence"] == pytest.approx(
                b["predictions"][target]["confidence"], abs=1e-4)


def test_unreadable_text_gets_none_and_a_reason(bert_trained):
    out = BT.BertService.load().predict(["zzz qqq xxx", "   ", "https://x.com/a", "river delta"])
    for i in (0, 1, 2):
        p = out[i]["predictions"]["subject"]
        assert p["label"] is None and p["confidence"] is None and p["top"] == []
    assert "unknown to the tokenizer" in out[0]["predictions"]["subject"]["warning"]
    assert "no usable tokens" in out[1]["predictions"]["subject"]["warning"]
    assert out[3]["predictions"]["subject"]["label"] == "Geography"


def test_long_text_is_truncated_and_inference_is_deterministic(bert_trained):
    svc = BT.BertService.load()
    long_text = "river delta monsoon " * 200                                  # far beyond max_len
    a = svc.predict([long_text])[0]["predictions"]["subject"]
    assert a["label"] == "Geography" and a == svc.predict([long_text])[0]["predictions"]["subject"]
    assert all(not m["model"].training for m in svc.models.values())          # eval mode: no dropout
    assert all(p.grad is None for m in svc.models.values() for p in m["model"].parameters())


def test_inference_needs_no_network(bert_trained, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("inference tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    svc = BT.BertService.load()                                              # tokenizer + models come from artifacts/
    assert svc.predict(["river delta monsoon"])[0]["predictions"]["subject"]["label"] == "Geography"


def test_artifacts_are_self_contained(prepared, mk, tiny_bert_dir, tmp_path):
    """After training, the original pretrained directory is no longer needed for inference."""
    local = tmp_path / "pretrained_copy"
    shutil.copytree(tiny_bert_dir, local)
    BT.train_all(cfg=mk(pretrained=str(local), epochs=1), targets=("subject",))
    shutil.rmtree(local)
    svc = BT.BertService.load()
    assert svc.predict(["river delta monsoon"])[0]["predictions"]["subject"]["label"] is not None


# ============================================================ loading errors and wrong architecture
def test_missing_and_corrupt_checkpoints_are_clear_errors(bert_trained):
    (_out() / "subject.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(BT.BertNotTrainedError, match="could not load checkpoint"):
        BT.BertService.load()
    (_out() / "subject.pt").unlink()
    with pytest.raises(BT.BertNotTrainedError, match="missing"):
        BT.BertService.load()


class _Evil:
    def __reduce__(self):                                                    # would run code if unpickled
        return (print, ("PWNED",))


def test_loading_never_executes_pickled_code(bert_trained, capsys):
    ck = _ckpt()
    ck["evil"] = _Evil()
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(BT.BertNotTrainedError, match="could not load checkpoint"):
        BT.BertService.load()
    assert "PWNED" not in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["lstm", "bilstm"])
def test_an_lstm_or_bilstm_checkpoint_cannot_be_loaded_as_bert(bert_trained, _lstm_baseline, _bilstm_baseline, arch):
    root = (_lstm_baseline if arch == "lstm" else _bilstm_baseline)[0]
    foreign = torch.load(root / "artifacts" / arch / "subject.pt", map_location="cpu", weights_only=True)
    assert foreign["arch"] == arch
    torch.save(foreign, _out() / "subject.pt")                               # filed under bert/
    with pytest.raises(BT.BertNotTrainedError, match=f"'{arch}' checkpoint, expected 'bert'"):
        BT.BertService.load()


@pytest.mark.parametrize("name,spec", [("lstm", L.LSTM_SPEC), ("bilstm", L.BILSTM_SPEC)])
def test_a_bert_checkpoint_cannot_be_loaded_as_lstm_or_bilstm(bert_trained, name, spec):
    foreign = _out().parent / name
    foreign.mkdir()
    for f in _out().glob("*.pt"):
        shutil.copy(f, foreign / f.name)
    shutil.copy(_out() / "metrics.json", foreign / "metrics.json")
    with pytest.raises(L.LSTMNotTrainedError, match=f"'bert' checkpoint, expected '{name}'"):
        L.LSTMService.load(spec=spec)


def test_a_checkpoint_for_the_wrong_target_is_rejected(bert_trained):
    torch.save(_ckpt("difficulty"), _out() / "subject.pt")                   # difficulty model saved as subject
    with pytest.raises(BT.BertNotTrainedError, match="could not load checkpoint"):
        BT.BertService.load()


@pytest.mark.parametrize("tamper,expected", [
    (lambda c: c.update(hidden_size=64), "size mismatch|shape"),             # encoder width differs from the weights
    (lambda c: c.update(num_hidden_layers=3), "Missing key|missing"),        # more layers than the weights contain
    (lambda c: c.update(num_hidden_layers=1), "Unexpected key|unexpected"),  # fewer layers than the weights contain
    (lambda c: c.update(id2label={"0": "a", "1": "b"}), "size mismatch|shape"),   # a different head size
    (lambda c: c.update(model_type="not-a-real-model"), ".*"),
])
def test_an_architecture_mismatch_inside_the_checkpoint_is_a_clear_error(bert_trained, tamper, expected):
    ck = _ckpt()
    tamper(ck["model_config"])
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(BT.BertNotTrainedError, match="could not load checkpoint") as e:
        BT.BertService.load()
    import re
    assert re.search(expected, str(e.value), re.IGNORECASE)


def test_the_saved_tokenizer_must_fit_the_checkpoint(bert_trained):
    ck = _ckpt()
    ck["model_config"]["vocab_size"] = 5000                                   # the tokenizer has far fewer tokens
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(BT.BertNotTrainedError, match="tokenizer"):
        BT.BertService.load()


def test_unsupported_format_version_is_rejected(bert_trained):
    ck = _ckpt()
    ck["format_version"] = 99
    torch.save(ck, _out() / "subject.pt")
    with pytest.raises(BT.BertNotTrainedError, match="could not load checkpoint"):
        BT.BertService.load()


def test_missing_or_corrupt_tokenizer_is_a_clear_error(bert_trained):
    (_out() / "tokenizer" / "tokenizer.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(BT.BertNotTrainedError, match="could not load the saved tokenizer"):
        BT.BertService.load()
    shutil.rmtree(_out() / "tokenizer")
    with pytest.raises(BT.BertNotTrainedError, match="tokenizer files are missing"):
        BT.BertService.load()


def test_metrics_without_checkpoints_is_not_loadable(bert_trained):
    for f in _out().glob("*.pt"):
        f.unlink()
    with pytest.raises(BT.BertNotTrainedError):
        BT.BertService.load()
