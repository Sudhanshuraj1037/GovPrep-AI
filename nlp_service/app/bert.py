"""BERT fine-tuning for subject / topic / difficulty (Hugging Face Transformers + PyTorch).

    text -> light cleaning -> WordPiece tokenizer ([CLS] ... [SEP], truncation)
         -> dynamic padding + attention mask -> pretrained BERT encoder
         -> pooled [CLS] -> classification head -> logits

A *pretrained* encoder is loaded with ``from_pretrained`` (never ``from_config``, so it is
never trained from scratch) and fine-tuned together with a newly initialised classification
head. The checkpoint is configurable: any Hugging Face id or local directory that
``AutoModelForSequenceClassification`` accepts, e.g. ``bert-base-uncased`` (default),
``distilbert-base-uncased`` (lighter, CPU-friendly) or ``bert-base-multilingual-cased`` (Hindi).

Same data protocol as TF-IDF, LSTM and BiLSTM
---------------------------------------------
* Reads the SAME ``data/processed/public_{train,val,test}.csv`` (stratified, seed 42) and the
  optional ``pdf_heldout.csv``. It never re-splits.
* One independent model per target (subject, topic, difficulty); class labels sorted exactly as
  in the other models, so confusion matrices are directly comparable.
* The VALIDATION split drives best-epoch selection and early stopping (macro-F1). The public TEST
  split and the PDF held-out set are only scored after training and never influence any choice.
* Class-weighted cross-entropy (``n / (k * count)``), like the other models.
* Metrics come from the shared ``app.metrics.classification_metrics``, and ``metrics.json``
  follows the ``app.model_metrics`` contract, so BERT drops into the same comparison table.

Differences that are inherent to BERT (and documented, not hidden)
------------------------------------------------------------------
* Its vocabulary and tokenizer come from the pretrained checkpoint, not from our data. No
  stop-word removal or stemming (that would hurt a pretrained model); only the same noise removal
  as the other models (URLs, e-mails, "Page 3 of 70", control characters) and Unicode NFKC.
* It has far more parameters; counts are recorded in ``metrics.json``.

Honest reporting
----------------
Every number comes from real predictions. Training is seeded and deterministic algorithms are
requested, so a run is reproducible on one machine; bit-identical results across different
hardware (CPU vs GPU) are not guaranteed and not claimed.

Artifacts under ``artifacts/bert/``: ``<target>.pt`` (plain tensors + primitives, loaded with
``torch.load(..., weights_only=True)`` so loading never executes pickled code), ``tokenizer/``
(the pretrained tokenizer, so inference needs no network) and, written last, ``metrics.json``.
A bert-base checkpoint is about 440 MB per target.

PyTorch and Transformers are imported here only. ``app.main`` never imports this module, so the
service (and /health) keeps working when either is not installed.
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from . import data as data_mod
from .config import settings
from .lstm import (
    EVAL_SPLITS,
    TARGETS,
    LSTMTrainingError,
    _batch_indices,
    _seed_everything,
    _write_json_atomic,
    balanced_class_weights,
    resolve_device,
)
from .metrics import classification_metrics
from .preprocessing import _CTRL_RE, _EMAIL_RE, _PAGE_RE, _URL_RE, _WS_RE, _ZW_RE

MODEL_NAME = "bert"
ARCH = "bert"
CHECKPOINT_FORMAT = 1          # same value as the LSTM/BiLSTM so a wrong-architecture file is reported as such
TOKENIZER_DIR = "tokenizer"
MAX_POSITIONS_HARD_LIMIT = 512
DEVICES = ("auto", "cpu", "cuda")


# ------------------------------------------------------------------- errors
class BertNotTrainedError(RuntimeError):
    """No usable trained BERT exists (or a checkpoint / tokenizer cannot be loaded)."""


class BertTrainingError(ValueError):
    """Training cannot proceed (bad config, unavailable pretrained model, unusable data...)."""


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class BertFineTuneConfig:
    """Defaults suit a real run (bert-base, GPU). For a quick CPU development run use
    ``--pretrained distilbert-base-uncased --max-len 64 --epochs 2 --batch-size 8``."""

    pretrained: str = "bert-base-uncased"   # Hugging Face id or local directory
    max_len: int = 128                      # tokens incl. [CLS] and [SEP]; BERT's limit is 512
    batch_size: int = 16
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1               # linear warm-up then linear decay to 0
    grad_clip: float = 1.0
    patience: int = 2                       # early stopping on val macro-F1; 0 disables it
    class_weights: bool = True              # "balanced" weights, like the other models
    freeze_encoder: bool = False            # True = train only the classification head (not full fine-tuning)
    clean_text: bool = True                 # URL / e-mail / page-marker / control-character removal
    device: str = "auto"                    # "auto" | "cpu" | "cuda"

    def __post_init__(self) -> None:
        if not isinstance(self.pretrained, str) or not self.pretrained.strip():
            raise ValueError("pretrained must be a non-empty Hugging Face id or local directory")
        if not 4 <= self.max_len <= MAX_POSITIONS_HARD_LIMIT:
            raise ValueError(f"max_len must be between 4 and {MAX_POSITIONS_HARD_LIMIT}")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip < 0 or self.patience < 0:
            raise ValueError("lr must be > 0; weight_decay, grad_clip and patience must be >= 0")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifacts_dir() -> Path:
    return Path(settings.artifacts_dir) / MODEL_NAME


@contextlib.contextmanager
def _quiet_transformers():
    """Silence the expected 'classifier weights newly initialised' notice and progress bars."""
    log = transformers.logging
    prev_level, prev_bar = log.get_verbosity(), log.is_progress_bar_enabled()
    log.set_verbosity_error()
    log.disable_progress_bar()
    try:
        yield
    finally:
        log.set_verbosity(prev_level)
        if prev_bar:
            log.enable_progress_bar()


# ------------------------------------------------------- text and encoding
def clean_for_bert(text: Any) -> str:
    """The noise removal shared with the other models, minus everything that would hurt a
    pretrained model: no lower-casing (the tokenizer decides), no stop-word removal, no
    stemming, apostrophes kept."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZW_RE.sub("", text)
    text = _CTRL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PAGE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def encode_texts(
    tokenizer, texts: Sequence[Any], max_len: int, clean: bool = True
) -> tuple[list[list[int]], dict[str, Any], list[str | None]]:
    """Tokenise with the pretrained tokenizer.

    Returns ``(ids, stats, reasons)``: ``ids[i]`` is ``[CLS] ... [SEP]`` truncated to ``max_len``
    (the special tokens are always kept); ``reasons[i]`` is ``None`` if text ``i`` is readable, or
    why it is not (empty after cleaning / every word is unknown to the tokenizer).
    Unreadable rows still get ids (just the special tokens) so every row can be scored.
    """
    cleaned = [clean_for_bert(t) if clean else (t if isinstance(t, str) else "") for t in texts]
    n = len(cleaned)
    if n == 0:
        return [], {"rows": 0, "tokens": 0, "unk_rate": None, "empty_after_cleaning": 0,
                    "truncated_rows": 0, "all_unknown_rows": 0}, []
    body = tokenizer(cleaned, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    ids = tokenizer(cleaned, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    n_special = len(tokenizer("", add_special_tokens=True)["input_ids"])
    unk = tokenizer.unk_token_id
    tokens = unknown = empty = truncated = all_unknown = 0
    reasons: list[str | None] = []
    for text, b in zip(cleaned, body):
        tokens += len(b)
        n_unk = sum(1 for t in b if t == unk) if unk is not None else 0
        unknown += n_unk
        if len(b) > max_len - n_special:
            truncated += 1
        if not text or not b:
            empty += 1
            reasons.append("no usable tokens after preprocessing")
        elif unk is not None and n_unk == len(b):
            all_unknown += 1
            reasons.append("no known words: every token is unknown to the tokenizer")
        else:
            reasons.append(None)
    stats = {
        "rows": n,
        "tokens": tokens,
        "unk_rate": (unknown / tokens) if tokens else None,
        "empty_after_cleaning": empty,
        "truncated_rows": truncated,
        "all_unknown_rows": all_unknown,
    }
    return ids, stats, reasons


def collate(id_lists: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest sequence in the batch. The attention mask is built from the true
    lengths (1 = real token, 0 = padding), never inferred from the pad id."""
    width = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, : len(x)] = torch.tensor(list(x), dtype=torch.long)
        mask[i, : len(x)] = 1
    return ids, mask


def lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up to 1.0, then linear decay to 0.0 (the standard BERT fine-tuning schedule)."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


# ---------------------------------------------------------------- model I/O
def _load_tokenizer(source: str):
    try:
        with _quiet_transformers():
            tok = AutoTokenizer.from_pretrained(source)
    except Exception as exc:
        raise BertTrainingError(
            f"could not load the pretrained tokenizer '{source}': {exc}. "
            "Check the name (or the local path) and, for a Hugging Face id, your internet connection."
        ) from exc
    if tok.pad_token_id is None:
        raise BertTrainingError(f"the tokenizer of '{source}' has no padding token; choose a BERT-style checkpoint")
    return tok


def check_tokenizer_fits_model(tokenizer, model_vocab_size: int | None) -> None:
    """Guard against a tokenizer that does not belong to the model.

    Some transformers versions silently build a default tokenizer holding only the special tokens
    when a directory has no tokenizer files; every word would then become [UNK] and training would
    'succeed' on garbage. A tokenizer larger than the embedding table would index out of range."""
    if model_vocab_size is None:
        return
    n = len(tokenizer)
    if n > model_vocab_size:
        raise ValueError(f"the tokenizer has {n} tokens but the model only has {model_vocab_size} embeddings")
    if n < 0.5 * model_vocab_size:
        raise ValueError(
            f"the tokenizer has only {n} tokens but the model expects about {model_vocab_size}: "
            "the tokenizer files are missing or belong to another model"
        )


def _load_pretrained_model(cfg: BertFineTuneConfig, classes: list[str]):
    """Pretrained encoder weights + a freshly initialised classification head. Never from_config."""
    label2id = {c: i for i, c in enumerate(classes)}
    try:
        with _quiet_transformers():
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.pretrained, num_labels=len(classes),
                id2label={i: c for c, i in label2id.items()}, label2id=label2id,
            )
    except Exception as exc:
        raise BertTrainingError(
            f"could not load the pretrained model '{cfg.pretrained}': {exc}. "
            "Check the name (or the local path) and, for a Hugging Face id, your internet connection."
        ) from exc
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and cfg.max_len > limit:
        raise BertTrainingError(
            f"max_len={cfg.max_len} exceeds the {limit} positions of '{cfg.pretrained}'; lower --max-len"
        )
    return model.float()  # fine-tune in fp32 whatever dtype the checkpoint was stored in


def _config_to_primitives(config) -> dict[str, Any]:
    return json.loads(config.to_json_string())


@torch.inference_mode()
def _forward_logits(model, id_lists: Sequence[Sequence[int]], pad_id: int, batch_size: int, device) -> torch.Tensor:
    """Logits for every row, in order (model in eval mode, no gradients)."""
    model.eval()
    out = []
    for start in range(0, len(id_lists), batch_size):
        ids, mask = collate(id_lists[start : start + batch_size], pad_id)
        out.append(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 0)


# ------------------------------------------------------------------ training
def train_target(
    target: str,
    frames: dict[str, pd.DataFrame],
    encoded: dict[str, list[list[int]]],
    pad_id: int,
    cfg: BertFineTuneConfig,
    seed: int,
    device: torch.device,
    out_dir: Path,
    on_epoch: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Fine-tune one target. The best-epoch checkpoint is written to ``<target>.pt.tmp`` in
    ``out_dir`` and renamed by ``train_all`` once every target has finished.
    Returns ``(staged_checkpoint_path_or_None, metrics_entry)``."""

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
    X_tr = [encoded["train"][i] for i in tr]
    y_tr = torch.tensor([to_idx[c] for c in y_train_str], dtype=torch.long)

    va = rows("val")
    va_labels = frames["val"][target].to_numpy()[va]
    keep = np.array([lab in to_idx for lab in va_labels], dtype=bool)
    va_sel, va_ignored = va[keep], int((~keep).sum())
    have_val = len(va_sel) > 0
    if have_val:
        X_va = [encoded["val"][i] for i in va_sel]
        y_va = torch.tensor([to_idx[c] for c in frames["val"][target].to_numpy()[va_sel]], dtype=torch.long)

    # ---- model (seeded per target, so a target never depends on the others)
    _seed_everything(seed)
    model = _load_pretrained_model(cfg, classes)
    if cfg.freeze_encoder:
        for p in model.base_model.parameters():
            p.requires_grad_(False)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},   # biases, LayerNorm
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.lr)
    steps_per_epoch = math.ceil(len(y_tr) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = math.ceil(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_scale(s, total_steps, warmup_steps))
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
            ids, mask = collate([X_tr[i] for i in idx.tolist()], pad_id)
            y = y_tr[idx].to(device)
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            ce = F.cross_entropy(logits, y, reduction="none")
            if class_w is not None:
                w = class_w[y]
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(ce.sum().detach())
        record: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / len(y_tr)}

        score = float(epoch)  # no validation rows: later epochs simply win ("last epoch")
        if have_val:
            logits = _forward_logits(model, X_va, pad_id, cfg.batch_size, device)
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
    entry: dict[str, Any] = {
        "status": "trained",
        "labels": classes,
        "n_train": int(len(tr)),
        "model": {
            "pretrained": cfg.pretrained,
            "model_type": model.config.model_type,
            "num_labels": len(classes),
            "freeze_encoder": cfg.freeze_encoder,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "model_selection": "best validation macro-F1" if have_val else "last epoch (no validation rows)",
            "validation_rows_used": int(len(va_sel)),
            "validation_rows_ignored_unseen_label": va_ignored,
            "class_weights": [float(w) for w in class_w.cpu()] if class_w is not None else None,
            "lr_schedule": {"total_steps": total_steps, "warmup_steps": warmup_steps},
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
        logits = _forward_logits(model, [encoded[name][i] for i in idx], pad_id, cfg.batch_size, device)
        pred = [classes[i] for i in logits.argmax(1).tolist()]
        entry["splits"][name] = classification_metrics(
            frames[name][target].to_numpy()[idx].tolist(), pred, classes, train_labels=y_train_str.tolist()
        )

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT,
        "arch": ARCH,
        "target": target,
        "pretrained": cfg.pretrained,
        "model_config": _config_to_primitives(model.config),
        "labels": classes,
        "max_len": cfg.max_len,
        "clean_text": cfg.clean_text,
        "seed": seed,
        "state_dict": best_state,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / f"{target}.pt.tmp"
    torch.save(checkpoint, staged)
    del model, optimizer, checkpoint, best_state     # free memory before the next target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return staged, entry


def train_all(
    *,
    targets: Sequence[str] = TARGETS,
    cfg: BertFineTuneConfig | None = None,
    seed: int = data_mod.DEFAULT_SEED,
    on_epoch: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune BERT for every requested target and save ``artifacts/bert/``.

    Raises ``DataNotFoundError`` if the processed splits are missing and ``BertTrainingError``
    for a bad config, an unavailable pretrained model or when no target can be trained. In those
    cases nothing is written, so /health keeps reporting "not trained". ``metrics.json`` is
    written last and is the marker /health looks for.
    """
    cfg = cfg or BertFineTuneConfig()
    unknown = [t for t in targets if t not in TARGETS]
    if unknown or not targets:
        raise ValueError(f"targets must be a non-empty subset of {TARGETS}; got {list(targets)}")
    try:
        device = resolve_device(cfg.device)
    except LSTMTrainingError as exc:
        raise BertTrainingError(str(exc)) from exc

    frames: dict[str, pd.DataFrame] = {
        "train": data_mod.load_split("public_train"),
        "val": data_mod.load_split("public_val"),
        "test": data_mod.load_split("public_test"),
    }
    has_heldout = data_mod.processed_path("pdf_heldout").is_file()
    if has_heldout:
        frames["pdf_heldout"] = data_mod.load_split("pdf_heldout")

    tokenizer = _load_tokenizer(cfg.pretrained)
    try:
        with _quiet_transformers():
            pretrained_vocab = getattr(AutoConfig.from_pretrained(cfg.pretrained), "vocab_size", None)
        check_tokenizer_fits_model(tokenizer, pretrained_vocab)
    except Exception as exc:
        raise BertTrainingError(f"unusable pretrained tokenizer for '{cfg.pretrained}': {exc}") from exc
    encoded: dict[str, list[list[int]]] = {}
    enc_stats: dict[str, Any] = {}
    for name, df in frames.items():
        encoded[name], enc_stats[name], _ = encode_texts(tokenizer, df["text"].tolist(), cfg.max_len, cfg.clean_text)

    out = artifacts_dir()
    staged: dict[str, Path] = {}
    entries: dict[str, Any] = {}
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    finished = False
    try:
        for target in targets:
            path, entry = train_target(
                target, frames, encoded, tokenizer.pad_token_id, cfg, seed, device, out, on_epoch=on_epoch
            )
            entries[target] = entry
            if path is not None:
                staged[target] = path
        if not staged:
            raise BertTrainingError(
                f"No target could be trained: { {t: e.get('reason') for t, e in entries.items()} }"
            )

        manifest = None
        if data_mod.manifest_path().is_file():
            manifest = json.loads(data_mod.manifest_path().read_text(encoding="utf-8"))
        names = ["public_train", "public_val", "public_test"] + (["pdf_heldout"] if has_heldout else [])
        first = next(e for e in entries.values() if e["status"] == "trained")
        metrics = {
            "model": MODEL_NAME,
            "description": "Pretrained BERT fine-tuned for text classification (Hugging Face Transformers + PyTorch), "
                           "one model per target",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "framework": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
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
            "preprocessing": {
                "type": "pretrained WordPiece tokenizer",
                "tokenizer": cfg.pretrained,
                "clean_text": cfg.clean_text,
                "max_len": cfg.max_len,
                "stop_word_removal": False,
                "stemming": False,
            },
            "hyperparameters": cfg.to_dict(),
            "pretrained": {
                "name": cfg.pretrained,
                "model_type": first["model"]["model_type"],
                "vocab_size": int(tokenizer.vocab_size) if tokenizer.vocab_size is not None else None,
                "freeze_encoder": cfg.freeze_encoder,
            },
            "targets": entries,
        }

        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").unlink(missing_ok=True)   # drop the old marker first
        for old in out.glob("*.pt"):
            old.unlink()
        shutil.rmtree(out / TOKENIZER_DIR, ignore_errors=True)
        for target, tmp in staged.items():
            os.replace(tmp, out / f"{target}.pt")
        tokenizer.save_pretrained(str(out / TOKENIZER_DIR))
        _write_json_atomic(out / "metrics.json", metrics)   # marker: written last
        finished = True
        return metrics
    finally:
        torch.use_deterministic_algorithms(prev_det)
        if not finished:                                    # never leave half-written staging files behind
            for tmp in out.glob("*.pt.tmp") if out.exists() else []:
                tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------- inference
class BertService:
    """Loads the saved BERT checkpoints and classifies text (CPU, no network)."""

    def __init__(self, tokenizer, models: dict[str, dict[str, Any]], metrics: dict[str, Any]):
        if not models:
            raise BertNotTrainedError("no trained BERT target available")
        self.tokenizer = tokenizer
        self.models = models
        self.metrics = metrics

    @property
    def targets(self) -> list[str]:
        return sorted(self.models)

    @property
    def trained_at(self) -> str | None:
        return self.metrics.get("trained_at")

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "BertService":
        d = Path(directory) if directory else artifacts_dir()
        metrics_file = d / "metrics.json"
        if not metrics_file.is_file():
            raise BertNotTrainedError(
                "BERT not trained yet (artifacts/bert/metrics.json is missing). "
                "Run: python -m training.prepare_data ... then python -m training.train_bert"
            )
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        tok_dir = d / TOKENIZER_DIR
        if not tok_dir.is_dir():
            raise BertNotTrainedError(
                f"the tokenizer files are missing ({TOKENIZER_DIR}/). Re-run training (artifacts are not committed to git)."
            )
        try:
            with _quiet_transformers():
                tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
        except Exception as exc:
            raise BertNotTrainedError(f"could not load the saved tokenizer: {exc}") from exc

        models: dict[str, dict[str, Any]] = {}
        for target, entry in metrics.get("targets", {}).items():
            if entry.get("status") != "trained":
                continue
            f = d / f"{target}.pt"
            if not f.is_file():
                raise BertNotTrainedError(
                    f"metrics.json lists '{target}' as trained but {f.name} is missing. "
                    "Re-run training (checkpoints are not committed to git)."
                )
            try:
                # Memory-map the checkpoint so its tensors are not eagerly copied
                # into a second full-size allocation.
                ckpt = torch.load(
                    f,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                if ckpt["format_version"] != CHECKPOINT_FORMAT or ckpt["target"] != target:
                    raise ValueError("unexpected checkpoint format or target")
                if ckpt["arch"] != ARCH:
                    raise ValueError(f"this is a '{ckpt['arch']}' checkpoint, expected '{ARCH}'")

                spec = dict(ckpt["model_config"])
                check_tokenizer_fits_model(tokenizer, spec.get("vocab_size"))
                config = AutoConfig.for_model(spec.pop("model_type"), **spec)

                # Construct parameters on the meta device to avoid allocating
                # random initial weights before attaching the checkpoint tensors.
                with torch.device("meta"):
                    with _quiet_transformers():
                        model = AutoModelForSequenceClassification.from_config(config)

                # assign=True attaches the checkpoint tensors instead of copying
                # them into separately allocated parameter storage.
                model.load_state_dict(
                    ckpt["state_dict"],
                    strict=True,
                    assign=True,
                )
                model.eval()
                models[target] = {
                    "model": model,
                    "labels": list(ckpt["labels"]),
                    "max_len": int(ckpt["max_len"]),
                    "clean_text": bool(ckpt["clean_text"]),
                }
                del ckpt
                gc.collect()
            except Exception as exc:
                raise BertNotTrainedError(f"could not load checkpoint {f.name}: {exc}") from exc
        return cls(tokenizer, models, metrics)

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[dict[str, Any]]:
        """Per text and per target: label, confidence (softmax probability) and the top-k
        alternatives. A text the model cannot read, because it is empty after cleaning or every
        word is unknown to the tokenizer, gets ``None`` and a warning instead of a guess."""
        results: list[dict[str, Any]] = [{"predictions": {}} for _ in texts]
        pad_id = self.tokenizer.pad_token_id
        for target, m in self.models.items():
            ids, _, reasons = encode_texts(self.tokenizer, list(texts), m["max_len"], m["clean_text"])
            usable = [i for i, r in enumerate(reasons) if r is None]
            probs = None
            if usable:
                logits = _forward_logits(m["model"], [ids[i] for i in usable], pad_id, 32, torch.device("cpu"))
                probs = F.softmax(logits, dim=1).numpy()
            labels = m["labels"]
            pos = {i: k for k, i in enumerate(usable)}
            for i in range(len(texts)):
                if i not in pos:
                    results[i]["predictions"][target] = {
                        "label": None, "confidence": None, "top": [], "warning": reasons[i],
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
