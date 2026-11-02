"""Shared test fixtures.

IMPORTANT: the data below is SYNTHETIC and exists only to exercise the code
(splitting, training, saving, loading, API). Any score computed on it says nothing
about the real models and must never be reported as a result.

Every test runs against temporary data/artifacts directories. A session-level
guard fails the run if anything was written into the real nlp_service/data or
nlp_service/artifacts folders.
"""
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest

from app import api_bert, api_bilstm, api_lstm, api_nlp, api_word2vec
from app import data as D
from app import word2vec as W
from app.config import settings

POOLS = {
    "History": "empire dynasty emperor battle ancient kingdom medieval treaty colony revolt".split(),
    "Geography": "river mountain plateau monsoon delta latitude climate soil desert coast".split(),
    "Science": "atom molecule enzyme voltage velocity photon cell reaction gravity orbit".split(),
    "Polity": "constitution article parliament amendment judiciary federal preamble election bill statute".split(),
}
FILLER = "which of the following is correct about the".split()


def make_toy_frame(n_per_class: int = 60, seed: int = 0, with_topic: bool = True) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for subject, pool in POOLS.items():
        for i in range(n_per_class):
            words = rng.sample(pool, 4) + rng.sample(FILLER, 3)
            rng.shuffle(words)
            rows.append(
                {
                    "text": " ".join(words) + f" case{subject[:2].lower()}{i}",
                    "subject": subject,
                    "topic": f"{subject.lower()}-t{i % 2}" if with_topic else "",
                    "difficulty": rng.choice(["easy", "medium", "hard"]),
                }
            )
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "subject", "topic", "difficulty"])


@pytest.fixture(scope="session", autouse=True)
def _real_dirs_untouched():
    real = [Path(settings.data_dir), Path(settings.artifacts_dir)]

    def snapshot():
        return sorted(
            str(p.relative_to(r)) + f":{p.stat().st_size}"
            for r in real for p in r.rglob("*") if p.is_file()
        )

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote into the real data/ or artifacts/ folder"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path / "artifacts")
    (tmp_path / "data" / "public").mkdir(parents=True)
    api_nlp._cache.clear()
    api_word2vec._cache.clear()
    api_lstm._cache.clear()
    api_bilstm._cache.clear()
    api_bert._cache.clear()
    yield tmp_path
    api_nlp._cache.clear()
    api_word2vec._cache.clear()
    api_lstm._cache.clear()
    api_bilstm._cache.clear()
    api_bert._cache.clear()


@pytest.fixture
def toy_csv(env):
    path = env / "data" / "public" / "toy.csv"
    make_toy_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def w2v_trained(env):
    """Synthetic 4-subject corpus (100 docs/class) with BOTH Word2Vec variants trained.
    Test-only: never a source of reported results."""
    path = env / "data" / "public" / "toy100.csv"
    make_toy_frame(100).to_csv(path, index=False)
    D.prepare_splits(path, seed=42)
    # 100 epochs: a corpus this small (about 280 documents) is under-trained at 30
    return W.train_all(cfg=W.Word2VecConfig(vector_size=50, epochs=100), seed=42)


# ------------------------------------------------------------- LSTM / BiLSTM
# A deliberately tiny CPU configuration so the tests stay fast. Synthetic data only.
TINY_LSTM = dict(embedding_dim=16, hidden_dim=16, epochs=30, batch_size=16, patience=4)


def _train_baseline(tmp_path_factory, module_name: str):
    """Train a recurrent model ONCE per test session (all three targets, synthetic data, temp dirs).

    Both architectures get byte-identical data and splits, so they can be compared."""
    pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module(f"app.{module_name}")
    root = tmp_path_factory.mktemp(f"{module_name}_baseline")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "data_dir", root / "data")
    mp.setattr(settings, "artifacts_dir", root / "artifacts")
    try:
        (root / "data" / "public").mkdir(parents=True)
        csv = root / "data" / "public" / "toy.csv"
        make_toy_frame(60).to_csv(csv, index=False)
        D.prepare_splits(csv, seed=42)
        metrics = mod.train_all(cfg=mod.LSTMConfig(**TINY_LSTM) if module_name == "lstm"
                                else mod.BiLSTMConfig(**TINY_LSTM), seed=42)
    finally:
        mp.undo()
    return root, metrics


def _private_copy(baseline, tmp_path, monkeypatch):
    root, metrics = baseline
    shutil.copytree(root / "data", tmp_path / "data")
    shutil.copytree(root / "artifacts", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path / "artifacts")
    api_nlp._cache.clear()
    api_word2vec._cache.clear()
    api_lstm._cache.clear()
    api_bilstm._cache.clear()
    api_bert._cache.clear()
    return metrics


@pytest.fixture(scope="session")
def _lstm_baseline(tmp_path_factory):
    return _train_baseline(tmp_path_factory, "lstm")


@pytest.fixture(scope="session")
def _bilstm_baseline(tmp_path_factory):
    return _train_baseline(tmp_path_factory, "bilstm")


@pytest.fixture
def lstm_trained(_lstm_baseline, tmp_path, monkeypatch):
    """A private, mutable copy of the trained LSTM artifacts and splits for one test."""
    yield _private_copy(_lstm_baseline, tmp_path, monkeypatch)
    api_lstm._cache.clear()


@pytest.fixture
def bilstm_trained(_bilstm_baseline, tmp_path, monkeypatch):
    """A private, mutable copy of the trained BiLSTM artifacts and splits for one test."""
    yield _private_copy(_bilstm_baseline, tmp_path, monkeypatch)
    api_bilstm._cache.clear()


@pytest.fixture
def all_models_trained(lstm_trained, _bilstm_baseline):
    """TF-IDF, LSTM and BiLSTM all trained (synthetic data) on the SAME processed splits."""
    from app import tfidf as T

    root, bilstm_metrics = _bilstm_baseline
    shutil.copytree(root / "artifacts" / "bilstm", Path(settings.artifacts_dir) / "bilstm")
    T.train_all(seed=42)
    return {"lstm": lstm_trained, "bilstm": bilstm_metrics}


# --------------------------------------------------------------------- BERT
# IMPORTANT: Hugging Face is not reachable from the development sandbox, so these tests never
# download a real pretrained BERT. They build a TINY stand-in locally (random weights, 2 layers,
# hidden size 32, a vocabulary of just the synthetic words) and save it as a directory. The code
# under test loads it with the very same `from_pretrained` path a real checkpoint uses, which
# verifies the machinery (tokenizer, attention masks, fine-tuning loop, checkpoints, metrics).
# It says NOTHING about real BERT quality, and nothing computed on it is a result.
# 30 epochs / patience 8: a tiny random-weight transformer sits at chance-level loss for its first few
# epochs, so a short schedule would stop before it has learned anything.
TINY_BERT = dict(max_len=32, batch_size=16, epochs=30, lr=1e-3, patience=8)


def tiny_bert_vocab() -> list[str]:
    words = sorted({w for ws in POOLS.values() for w in ws} | set(FILLER) | {"history"})
    return ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words


def build_tiny_bert(directory: Path) -> Path:
    """Save a tiny random-weight BERT encoder + WordPiece tokenizer into ``directory``."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(tiny_bert_vocab())}
    tk = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    directory.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(directory))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(str(directory))       # encoder only, like a real pretrained checkpoint
    return directory


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return build_tiny_bert(tmp_path_factory.mktemp("tiny_bert") / "pretrained")


@pytest.fixture(scope="session")
def _bert_baseline(tmp_path_factory, tiny_bert_dir):
    """Fine-tune the tiny BERT ONCE per session (all three targets, synthetic data, temp dirs)."""
    from app import bert as BERT

    root = tmp_path_factory.mktemp("bert_baseline")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "data_dir", root / "data")
    mp.setattr(settings, "artifacts_dir", root / "artifacts")
    try:
        (root / "data" / "public").mkdir(parents=True)
        csv = root / "data" / "public" / "toy.csv"
        make_toy_frame(60).to_csv(csv, index=False)
        D.prepare_splits(csv, seed=42)
        metrics = BERT.train_all(cfg=BERT.BertFineTuneConfig(pretrained=str(tiny_bert_dir), **TINY_BERT), seed=42)
    finally:
        mp.undo()
    return root, metrics


@pytest.fixture
def bert_trained(_bert_baseline, tmp_path, monkeypatch):
    """A private, mutable copy of the fine-tuned tiny-BERT artifacts and splits for one test."""
    yield _private_copy(_bert_baseline, tmp_path, monkeypatch)
    api_bert._cache.clear()


@pytest.fixture
def four_models_trained(all_models_trained, _bert_baseline):
    """TF-IDF, LSTM, BiLSTM and BERT all trained (synthetic data) on the SAME processed splits."""
    root, bert_metrics = _bert_baseline
    shutil.copytree(root / "artifacts" / "bert", Path(settings.artifacts_dir) / "bert")
    return {**all_models_trained, "bert": bert_metrics}
