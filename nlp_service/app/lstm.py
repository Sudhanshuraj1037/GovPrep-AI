"""LSTM and BiLSTM text classifiers (PyTorch) for subject / topic / difficulty.

    text -> TextPreprocessor -> vocabulary ids -> pad / truncate -> Embedding
         -> Dropout -> LSTM (packed sequences) -> final hidden state -> Dropout -> Linear

This module is the single shared implementation. An ``ArchSpec`` selects the
architecture: ``LSTM_SPEC`` (unidirectional) or ``BILSTM_SPEC`` (bidirectional: the
classifier reads the concatenation of the forward LSTM's final state and the backward
LSTM's final state). Vocabulary, encoding, training loop, evaluation, checkpointing and
inference are identical for both, so the two models differ ONLY in the architecture and
are directly comparable. ``app/bilstm.py`` is a thin façade over this module.

One independent model is trained per target, exactly like the TF-IDF baseline, so
the two are directly comparable in the Phase 8 model comparison.

Data discipline (identical to Phase 2)
--------------------------------------
* Reads the SAME ``data/processed/public_{train,val,test}.csv`` produced by
  ``training.prepare_data`` (stratified, seed 42). It never re-splits.
* The vocabulary is built from the TRAIN split only.
* The VALIDATION split drives early stopping / best-epoch selection.
* The public TEST split and the PDF held-out set are only scored, after training,
  and never influence weights, vocabulary or any choice.
* Optional Word2Vec initialisation (``init_embeddings="word2vec"``) is refused
  unless that Word2Vec model was trained on exactly the current ``public_train.csv``.

Honest reporting
----------------
Every number in ``metrics.json`` comes from real predictions via ``app.metrics``.
Training is reproducible on one machine (seeded; deterministic algorithms
requested); bit-identical results across different hardware (CPU vs GPU) are not
guaranteed and are not claimed.

Checkpoints are saved as plain tensors + primitives and loaded with
``torch.load(..., weights_only=True)``, so loading never executes pickled code.

PyTorch is imported here only. ``app.main`` does not import this module, so the
service (and /health) keeps working when PyTorch is not installed.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from . import data as data_mod
from .config import settings
from .metrics import classification_metrics
from .preprocessing import PreprocessConfig, TextPreprocessor

MODEL_NAME = "lstm"  # kept for backwards compatibility; see ArchSpec below
TARGETS = ("subject", "topic", "difficulty")
EVAL_SPLITS = ("val", "test", "pdf_heldout")
PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1
CHECKPOINT_FORMAT = 1
DEVICES = ("auto", "cpu", "cuda")
INIT_EMBEDDINGS = ("none", "word2vec")


# ------------------------------------------------------------ architectures
@dataclass(frozen=True)
class ArchSpec:
    """Everything that differs between the unidirectional and bidirectional model."""

    name: str            # artifacts folder, metrics "model" field, checkpoint "arch"
    label: str           # human-readable name used in messages
    bidirectional: bool
    description: str


LSTM_SPEC = ArchSpec("lstm", "LSTM", False, "LSTM text classifier (PyTorch), one model per target")
BILSTM_SPEC = ArchSpec(
    "bilstm", "BiLSTM", True, "Bidirectional LSTM text classifier (PyTorch), one model per target"
)
SPECS = {s.name: s for s in (LSTM_SPEC, BILSTM_SPEC)}


# ------------------------------------------------------------------- errors
class LSTMNotTrainedError(RuntimeError):
    """No usable trained LSTM / BiLSTM exists (or its checkpoint cannot be loaded)."""


class LSTMTrainingError(ValueError):
    """Training cannot proceed (bad config, unusable data, unavailable device...)."""


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class LSTMConfig:
    """CPU-friendly defaults; every value can be raised for a real run on a GPU / Colab.

    Shared by LSTM and BiLSTM. ``hidden_dim`` is the size PER DIRECTION, so a BiLSTM has
    about twice the recurrent parameters of an LSTM with the same settings; the parameter
    counts are recorded in metrics.json so a comparison can account for it.
    """

    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    max_len: int = 64
    min_freq: int = 2
    max_vocab: int = 20_000
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 4              # early stopping on val macro-F1; 0 disables it
    class_weights: bool = True     # "balanced" weights, like TF-IDF's class_weight
    init_embeddings: str = "none"  # "none" | "word2vec"
    w2v_variant: str = "skipgram"
    freeze_embeddings: bool = False
    device: str = "auto"           # "auto" | "cpu" | "cuda"

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "hidden_dim", "num_layers", "max_len", "min_freq",
                     "batch_size", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_vocab < 3:
            raise ValueError("max_vocab must be >= 3 (two special tokens + at least one word)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0 or self.patience < 0:
            raise ValueError("lr must be > 0; weight_decay, grad_clip and patience must be >= 0")
        if self.init_embeddings not in INIT_EMBEDDINGS:
            raise ValueError(f"init_embeddings must be one of {INIT_EMBEDDINGS}")
        if self.w2v_variant not in ("cbow", "skipgram"):
            raise ValueError("w2v_variant must be 'cbow' or 'skipgram'")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_preprocess_config() -> PreprocessConfig:
    """Keep stop-words and negations (word order is what an LSTM models); no stemming."""
    return PreprocessConfig(normalizer="none", remove_stopwords=False)


def artifacts_dir(spec: ArchSpec = LSTM_SPEC) -> Path:
    return Path(settings.artifacts_dir) / spec.name


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------- vocabulary and encoding
def build_vocab(token_docs: Sequence[Sequence[str]], min_freq: int, max_vocab: int) -> list[str]:
    """Index -> token list. 0 = <pad>, 1 = <unk>, then words by descending frequency
    (ties alphabetical, so the vocabulary is fully deterministic)."""
    counts = Counter(t for doc in token_docs for t in doc)
    items = sorted(((w, c) for w, c in counts.items() if c >= min_freq), key=lambda wc: (-wc[1], wc[0]))
    return [PAD, UNK] + [w for w, _ in items[: max_vocab - 2]]


def encode_docs(
    token_docs: Sequence[Sequence[str]], index: dict[str, int], max_len: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Integer-encode, truncate to ``max_len`` (keeping the FIRST tokens) and right-pad
    with ``<pad>``. Returns ``(ids [N, max_len], lengths [N], stats)``.

    An empty token list becomes a single ``<unk>`` so every row still gets a
    prediction; those rows are counted in ``stats['empty_after_preprocessing']``.
    """
    n = len(token_docs)
    ids = np.zeros((n, max_len), dtype=np.int64)
    lengths = np.ones(n, dtype=np.int64)
    tokens = unk = empty = truncated = 0
    for i, toks in enumerate(token_docs):
        row = [index.get(t, UNK_IDX) for t in toks]
        tokens += len(row)
        unk += row.count(UNK_IDX)
        if not row:
            row, empty = [UNK_IDX], empty + 1
        if len(row) > max_len:
            truncated += 1
            row = row[:max_len]
        ids[i, : len(row)] = row
        lengths[i] = len(row)
    stats = {
        "rows": n,
        "tokens": tokens,
        "unk_rate": (unk / tokens) if tokens else None,
        "empty_after_preprocessing": empty,
        "truncated_rows": truncated,
    }
    return torch.from_numpy(ids), torch.from_numpy(lengths), stats


def balanced_class_weights(counts: Sequence[int]) -> torch.Tensor:
    """n / (k * count_c): the same formula as scikit-learn's class_weight='balanced'."""
    c = np.asarray(counts, dtype=np.float64)
    return torch.tensor(c.sum() / (len(c) * c), dtype=torch.float32)


# --------------------------------------------------------------------- model
class LSTMTextClassifier(nn.Module):
    """Embedding -> Dropout -> LSTM -> final hidden state(s) -> Dropout -> Linear.

    Sequences are packed, so padding never reaches the LSTM and the final hidden
    state is the one at each sequence's true last token.

    ``bidirectional=True`` gives the BiLSTM: the representation is the concatenation of
    the top layer's forward final state (after the LAST real token) and backward final
    state (after the FIRST real token), so the classifier sees ``2 * hidden_dim`` features.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = PAD_IDX,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # between stacked layers only
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n is (num_layers * num_directions, batch, hidden): the last two rows are the top
        # layer's forward and backward final states.
        last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(self.dropout(last))


def _model_config(
    cfg: LSTMConfig, vocab_size: int, embedding_dim: int, num_classes: int, bidirectional: bool = False
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "num_classes": num_classes,
        "dropout": cfg.dropout,
        "pad_idx": PAD_IDX,
        "bidirectional": bidirectional,
    }


# ---------------------------------------------------------------- utilities
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise LSTMTrainingError("device 'cuda' was requested but CUDA is not available; use --device cpu or auto")
    return torch.device(name)


def _batch_indices(n: int, batch_size: int, generator: torch.Generator | None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _slice(ids: torch.Tensor, lengths: torch.Tensor, idx: torch.Tensor, device: torch.device):
    lens = lengths[idx]
    width = int(lens.max())  # trim padding that no sequence in this batch uses
    return ids[idx, :width].to(device), lens


@torch.no_grad()
def _forward_all(model, ids, lengths, batch_size: int, device) -> torch.Tensor:
    """Logits for every row, in order."""
    model.eval()
    out = []
    for idx in _batch_indices(len(ids), batch_size, None):
        x, lens = _slice(ids, lengths, idx, device)
        out.append(model(x, lens).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 0)


def _word2vec_matrix(vocab: list[str], variant: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Initial embedding matrix from the trained Word2Vec model (Phase 3).

    Refuses to run unless that model was trained on exactly the current
    public_train.csv, so no validation/test text can leak in through the embeddings.
    """
    from gensim.models import Word2Vec  # lazy: only needed for this option

    from . import word2vec as W

    wdir = W.artifacts_dir()
    meta_file, model_file = wdir / "metadata.json", wdir / f"{variant}.model"
    if not meta_file.is_file() or not model_file.is_file():
        raise LSTMTrainingError(
            f"init_embeddings='word2vec' needs a trained {variant} Word2Vec model "
            f"(missing under {wdir}). Run: python -m training.train_word2vec"
        )
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    current = data_mod.file_sha256(data_mod.processed_path("public_train"))
    if meta.get("corpus", {}).get("sha256") != current:
        raise LSTMTrainingError(
            "the Word2Vec model was trained on a different public_train.csv than the current one. "
            "Re-run training.train_word2vec so embeddings and LSTM use the same training split."
        )
    wv = Word2Vec.load(str(model_file)).wv
    rng = np.random.default_rng(seed)
    std = float(wv.vectors.std()) or 0.1
    matrix = rng.normal(0.0, std, size=(len(vocab), wv.vector_size)).astype(np.float32)
    matrix[PAD_IDX] = 0.0
    covered = 0
    for i, word in enumerate(vocab):
        if i > UNK_IDX and word in wv.key_to_index:
            matrix[i] = wv[word]
            covered += 1
    real_words = max(len(vocab) - 2, 1)
    return matrix, {
        "init": "word2vec",
        "variant": variant,
        "dimension": int(wv.vector_size),
        "words_covered": covered,
        "vocabulary_words": len(vocab) - 2,
        "coverage": covered / real_words,
        "word2vec_trained_at": meta.get("trained_at"),
        "word2vec_corpus_sha256": meta["corpus"]["sha256"],
    }


# ------------------------------------------------------------------ training
def train_target(
    target: str,
    frames: dict[str, pd.DataFrame],
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]],
    vocab: list[str],
    cfg: LSTMConfig,
    preprocess_config: dict[str, Any],
    seed: int,
    device: torch.device,
    embedding_matrix: np.ndarray | None = None,
    on_epoch: Callable[[str, dict[str, Any]], None] | None = None,
    spec: ArchSpec = LSTM_SPEC,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Train one target. Returns ``(checkpoint_or_None, metrics_entry)``."""

    def rows(split: str) -> np.ndarray:
        return np.flatnonzero((frames[split][target] != "").to_numpy())

    tr = rows("train")
    y_train_str = frames["train"][target].to_numpy()[tr]
    classes = sorted(set(y_train_str))
    if len(tr) == 0 or len(classes) < 2:
        return None, {
            "status": "skipped",
            "reason": f"needs >= 2 classes with labelled training rows; "
            f"found {len(classes)} class(es) in {len(tr)} rows",
        }
    to_idx = {c: i for i, c in enumerate(classes)}
    tr_t = torch.from_numpy(tr)
    X_tr, L_tr = encoded["train"][0][tr_t], encoded["train"][1][tr_t]
    y_tr = torch.tensor([to_idx[c] for c in y_train_str], dtype=torch.long)

    # validation rows used for model selection (labels must exist in the train classes)
    va = rows("val")
    va_labels = frames["val"][target].to_numpy()[va]
    keep = np.array([lab in to_idx for lab in va_labels], dtype=bool)
    va_sel, va_ignored = va[keep], int((~keep).sum())
    have_val = len(va_sel) > 0
    if have_val:
        va_t = torch.from_numpy(va_sel)
        X_va, L_va = encoded["val"][0][va_t], encoded["val"][1][va_t]
        y_va = torch.tensor([to_idx[c] for c in frames["val"][target].to_numpy()[va_sel]], dtype=torch.long)

    # ---- model (seeded per target so each target is independent of the others)
    _seed_everything(seed)
    emb_dim = embedding_matrix.shape[1] if embedding_matrix is not None else cfg.embedding_dim
    model_cfg = _model_config(cfg, len(vocab), emb_dim, len(classes), spec.bidirectional)
    model = LSTMTextClassifier(**model_cfg)
    if embedding_matrix is not None:
        with torch.no_grad():
            model.embedding.weight.copy_(torch.from_numpy(embedding_matrix))
    if cfg.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_w = (
        balanced_class_weights(np.bincount(y_tr.numpy(), minlength=len(classes))).to(device)
        if cfg.class_weights else None
    )
    gen = torch.Generator().manual_seed(seed)

    history: list[dict[str, Any]] = []
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    stopped_early = False
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        for idx in _batch_indices(len(y_tr), cfg.batch_size, gen):
            x, lens = _slice(X_tr, L_tr, idx, device)
            y = y_tr[idx].to(device)
            ce = F.cross_entropy(model(x, lens), y, reduction="none")
            if class_w is not None:
                w = class_w[y]
                loss = (ce * w).sum() / w.sum()      # weighted mean, as torch does
            else:
                loss = ce.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            loss_sum += float(ce.sum().detach())     # plain (unweighted) cross-entropy, for reporting
        record: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / len(y_tr)}

        score = float(epoch)  # no validation rows: later epochs simply win ("last epoch")
        if have_val:
            logits = _forward_all(model, X_va, L_va, cfg.batch_size, device)
            pred = logits.argmax(1)
            record["val_loss"] = float(F.cross_entropy(logits, y_va, reduction="mean"))
            record["val_accuracy"] = float((pred == y_va).float().mean())
            score = float(f1_score(y_va.numpy(), pred.numpy(), labels=list(range(len(classes))),
                                   average="macro", zero_division=0))
            record["val_f1_macro"] = score
        history.append(record)
        if on_epoch is not None:
            on_epoch(target, record)

        if score > best_f1:
            best_f1, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if have_val and cfg.patience > 0 and stale >= cfg.patience:
                stopped_early = True
                break
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)       # restore the best epoch
    model.to("cpu")

    entry: dict[str, Any] = {
        "status": "trained",
        "labels": classes,
        "n_train": int(len(tr)),
        "model": {**model_cfg, "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                  "total_parameters": int(sum(p.numel() for p in model.parameters()))},
        "training": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "model_selection": "best validation macro-F1" if have_val else "last epoch (no validation rows)",
            "validation_rows_used": int(len(va_sel)),
            "validation_rows_ignored_unseen_label": va_ignored,
            "class_weights": [float(w) for w in class_w.cpu()] if class_w is not None else None,
            "train_seconds": round(train_seconds, 3),
            "history": history,
        },
        "splits": {},
    }
    for name in EVAL_SPLITS:
        if name not in frames:
            continue
        idx = rows(name)
        if len(idx) == 0:
            continue
        t = torch.from_numpy(idx)
        logits = _forward_all(model, encoded[name][0][t], encoded[name][1][t], cfg.batch_size, torch.device("cpu"))
        pred = [classes[i] for i in logits.argmax(1).tolist()]
        entry["splits"][name] = classification_metrics(
            frames[name][target].to_numpy()[idx].tolist(), pred, classes, train_labels=y_train_str.tolist()
        )
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT,
        "arch": spec.name,
        "target": target,
        "model_config": model_cfg,
        "labels": classes,
        "vocab": vocab,
        "max_len": cfg.max_len,
        "preprocess_config": preprocess_config,
        "seed": seed,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    return checkpoint, entry


def train_all(
    *,
    targets: Sequence[str] = TARGETS,
    cfg: LSTMConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
    seed: int = data_mod.DEFAULT_SEED,
    on_epoch: Callable[[str, dict[str, Any]], None] | None = None,
    spec: ArchSpec = LSTM_SPEC,
) -> dict[str, Any]:
    """Train every requested target from ``data/processed/*.csv`` and save the artifacts
    under ``artifacts/<spec.name>/`` (``lstm`` or ``bilstm``).

    Raises ``DataNotFoundError`` if the processed splits are missing,
    ``LSTMTrainingError`` for bad config / unusable data / unavailable device.
    If no target can be trained nothing is written, so /health keeps reporting
    "not trained". ``metrics.json`` is written last and is the marker /health uses.
    """
    cfg = cfg or LSTMConfig()
    pre = TextPreprocessor(preprocess_cfg or default_preprocess_config())
    unknown = [t for t in targets if t not in TARGETS]
    if unknown or not targets:
        raise ValueError(f"targets must be a non-empty subset of {TARGETS}; got {list(targets)}")
    device = resolve_device(cfg.device)

    frames: dict[str, pd.DataFrame] = {
        "train": data_mod.load_split("public_train"),
        "val": data_mod.load_split("public_val"),
        "test": data_mod.load_split("public_test"),
    }
    has_heldout = data_mod.processed_path("pdf_heldout").is_file()
    if has_heldout:
        frames["pdf_heldout"] = data_mod.load_split("pdf_heldout")

    tokens = {name: [pre.tokens(t) for t in df["text"]] for name, df in frames.items()}
    vocab = build_vocab(tokens["train"], cfg.min_freq, cfg.max_vocab)   # TRAIN split only
    if len(vocab) <= 2:
        raise LSTMTrainingError(
            f"no word reaches min_freq={cfg.min_freq} in the training split; lower --min-freq or add data"
        )
    index = {w: i for i, w in enumerate(vocab)}
    encoded, enc_stats = {}, {}
    for name, docs in tokens.items():
        ids, lens, st = encode_docs(docs, index, cfg.max_len)
        encoded[name], enc_stats[name] = (ids, lens), st

    matrix, embedding_info = None, {"init": "random"}
    if cfg.init_embeddings == "word2vec":
        matrix, embedding_info = _word2vec_matrix(vocab, cfg.w2v_variant, seed)

    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        checkpoints: dict[str, dict[str, Any]] = {}
        entries: dict[str, Any] = {}
        for target in targets:
            ckpt, entry = train_target(
                target, frames, encoded, vocab, cfg, pre.config.to_dict(), seed, device,
                embedding_matrix=matrix, on_epoch=on_epoch, spec=spec,
            )
            entries[target] = entry
            if ckpt is not None:
                checkpoints[target] = ckpt
    finally:
        torch.use_deterministic_algorithms(prev_det)
    if not checkpoints:
        raise LSTMTrainingError(
            f"No target could be trained: { {t: e.get('reason') for t, e in entries.items()} }"
        )

    manifest = None
    if data_mod.manifest_path().is_file():
        manifest = json.loads(data_mod.manifest_path().read_text(encoding="utf-8"))
    names = ["public_train", "public_val", "public_test"] + (["pdf_heldout"] if has_heldout else [])
    metrics = {
        "model": spec.name,
        "description": spec.description,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "framework": {
            "torch": torch.__version__,
            "device": str(device),
            "deterministic_algorithms_requested": True,
        },
        "data": {
            "rows": {name: len(df) for name, df in frames.items()},
            "split_files_sha256": {n: data_mod.file_sha256(data_mod.processed_path(n)) for n in names},
            "pdf_heldout_evaluated": has_heldout,
            "split_manifest": manifest,
            "encoding": enc_stats,
        },
        "preprocessing": pre.config.to_dict(),
        "hyperparameters": cfg.to_dict(),
        "vocabulary": {
            "size": len(vocab),
            "min_freq": cfg.min_freq,
            "max_vocab": cfg.max_vocab,
            "built_from": "public_train",
        },
        "embeddings": embedding_info,
        "targets": entries,
    }

    out = artifacts_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").unlink(missing_ok=True)   # drop the old marker first
    for old in out.glob("*.pt"):
        old.unlink()
    for target, ckpt in checkpoints.items():
        tmp = out / f"{target}.pt.tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, out / f"{target}.pt")
    _write_json_atomic(out / "metrics.json", metrics)   # marker: written last
    return metrics


# ----------------------------------------------------------------- inference
class LSTMService:
    """Loads the saved LSTM (or BiLSTM) checkpoints and classifies text."""

    def __init__(self, models: dict[str, dict[str, Any]], metrics: dict[str, Any],
                 spec: ArchSpec = LSTM_SPEC):
        if not models:
            raise LSTMNotTrainedError(f"no trained {spec.label} target available")
        self.models = models
        self.metrics = metrics
        self.spec = spec

    @property
    def targets(self) -> list[str]:
        return sorted(self.models)

    @property
    def trained_at(self) -> str | None:
        return self.metrics.get("trained_at")

    @classmethod
    def load(cls, directory: str | Path | None = None, spec: ArchSpec = LSTM_SPEC) -> "LSTMService":
        d = Path(directory) if directory else artifacts_dir(spec)
        metrics_file = d / "metrics.json"
        if not metrics_file.is_file():
            raise LSTMNotTrainedError(
                f"{spec.label} not trained yet (artifacts/{spec.name}/metrics.json is missing). "
                f"Run: python -m training.prepare_data ... then python -m training.train_{spec.name}"
            )
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        models: dict[str, dict[str, Any]] = {}
        for target, entry in metrics.get("targets", {}).items():
            if entry.get("status") != "trained":
                continue
            f = d / f"{target}.pt"
            if not f.is_file():
                raise LSTMNotTrainedError(
                    f"metrics.json lists '{target}' as trained but {f.name} is missing. "
                    "Re-run training (checkpoints are not committed to git)."
                )
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=True)
                if ckpt["format_version"] != CHECKPOINT_FORMAT or ckpt["target"] != target:
                    raise ValueError("unexpected checkpoint format or target")
                if ckpt["arch"] != spec.name:
                    raise ValueError(f"this is a '{ckpt['arch']}' checkpoint, expected '{spec.name}'")
                if bool(ckpt["model_config"].get("bidirectional", False)) != spec.bidirectional:
                    raise ValueError(f"checkpoint directionality does not match {spec.label}")
                model = LSTMTextClassifier(**ckpt["model_config"])
                model.load_state_dict(ckpt["state_dict"])
            except Exception as exc:
                raise LSTMNotTrainedError(f"could not load checkpoint {f.name}: {exc}") from exc
            model.eval()
            models[target] = {
                "model": model,
                "labels": list(ckpt["labels"]),
                "index": {w: i for i, w in enumerate(ckpt["vocab"])},
                "max_len": int(ckpt["max_len"]),
                "pre": TextPreprocessor(PreprocessConfig.from_dict(ckpt["preprocess_config"])),
            }
        return cls(models, metrics, spec)

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[dict[str, Any]]:
        """Per text and per target: label, confidence (softmax probability) and the
        top-k alternatives. A text the model cannot read, because it has no tokens or
        every token is out of vocabulary, gets ``None`` and a warning instead of a guess."""
        results: list[dict[str, Any]] = [{"predictions": {}} for _ in texts]
        for target, m in self.models.items():
            docs = [m["pre"].tokens(t) for t in texts]
            reasons: dict[int, str] = {}
            usable: list[int] = []
            for i, d in enumerate(docs):
                if not d:
                    reasons[i] = "no usable tokens after preprocessing"
                elif all(t not in m["index"] for t in d):
                    reasons[i] = "no known words: every token is out of the model's vocabulary"
                else:
                    usable.append(i)
            probs = None
            if usable:
                ids, lens, _ = encode_docs([docs[i] for i in usable], m["index"], m["max_len"])
                logits = _forward_all(m["model"], ids, lens, 64, torch.device("cpu"))
                probs = F.softmax(logits, dim=1).numpy()
            labels = m["labels"]
            pos = {i: k for k, i in enumerate(usable)}
            for i in range(len(texts)):
                if i not in pos:
                    results[i]["predictions"][target] = {
                        "label": None, "confidence": None, "top": [],
                        "warning": reasons[i],
                    }
                    continue
                row = probs[pos[i]]
                order = np.argsort(row)[::-1][:top_k]
                results[i]["predictions"][target] = {
                    "label": labels[order[0]],
                    "confidence": float(row[order[0]]),
                    "top": [{"label": labels[j], "probability": float(row[j])} for j in order],
                }
        return results
