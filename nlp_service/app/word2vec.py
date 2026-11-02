"""Word2Vec (CBOW and Skip-Gram) trained on the public TRAIN split only.

Training corpus
---------------
Only ``data/processed/public_train.csv`` is read for training. The validation,
test and PDF held-out files are never part of the training corpus or of the
vocabulary. After training, token coverage is *measured* on them (how many of
their tokens the vocabulary knows); that measurement is never fed back into
training or hyper-parameter choice.

Text is tokenised with the shared ``TextPreprocessor``. For embeddings the default
config keeps the surface word forms (no stemming) so that nearest-neighbour
lists stay readable; stop-words are removed and negations kept.

Reproducibility
---------------
Fixed seed (42), ``workers=1`` and a stable hash function. Two runs on the same
``public_train.csv`` give identical vectors; a test verifies this across separate
processes with different ``PYTHONHASHSEED`` values.

What is (not) reported
----------------------
Word2Vec is unsupervised, so it has no accuracy or F1. ``metadata.json`` holds
descriptive statistics only (corpus size, vocabulary size, token coverage,
CBOW-vs-Skip-Gram neighbour overlap). No analogy or word-similarity benchmark
score is computed, because no suitable benchmark exists for this domain corpus.

Security note: gensim models are pickles. Only load artifacts you trained yourself.
"""
from __future__ import annotations

import json
import os
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import gensim

from . import data as data_mod
from .config import settings
from .preprocessing import PreprocessConfig, TextPreprocessor

MODEL_NAME = "word2vec"
VARIANTS = ("cbow", "skipgram")
DEFAULT_VARIANT = "skipgram"  # copes better with small corpora / rarer words
_SG = {"cbow": 0, "skipgram": 1}
MIN_VOCAB = 5
MAX_VIS_POINTS = 500

ANALOGY_CAVEAT = (
    "Embeddings were learned from a small domain corpus; analogy results are "
    "illustrative only and no analogy accuracy is claimed."
)


# ------------------------------------------------------------------- errors
class Word2VecNotTrainedError(RuntimeError):
    """No trained Word2Vec model exists."""


class Word2VecTrainingError(ValueError):
    """Training data exists but a model cannot be trained from it."""


class UnknownVariantError(LookupError):
    def __init__(self, variant: str, available: Sequence[str]):
        self.variant, self.available = variant, list(available)
        super().__init__(f"variant {variant!r} is not trained; available: {self.available}")


class WordNotInVocabularyError(LookupError):
    def __init__(self, words: Sequence[str], variant: str):
        self.words, self.variant = list(words), variant
        super().__init__(f"not in the {variant} vocabulary: {self.words}")


class VisualizationError(ValueError):
    """Invalid visualisation request (too few words, bad method, ...)."""


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class Word2VecConfig:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    negative: int = 5
    sample: float = 1e-3
    epochs: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_preprocess_config() -> PreprocessConfig:
    """Surface forms (no stemming), stop-words removed, negations kept."""
    return PreprocessConfig(normalizer="none")


def artifacts_dir() -> Path:
    return Path(settings.artifacts_dir) / MODEL_NAME


def _stable_hash(s: str) -> int:
    # Process-independent replacement for the built-in str hash (which is salted).
    return zlib.crc32(s.encode("utf-8"))


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- training
def _vocab_stats(
    docs: list[tuple[list[str], str]], vocab: set[str]
) -> dict[str, Any]:
    """Which subject each word belongs to, from TRAIN questions only.

    rate_s = (#train docs of subject s containing the word) / (#train docs of s)
    dominant_subject = argmax_s rate_s;  dominant_share = rate / sum of rates
    (normalised by class size so frequent subjects do not dominate).
    1.0 means the word only ever occurs in that subject.
    """
    n_docs: Counter[str] = Counter()
    df: dict[str, Counter[str]] = defaultdict(Counter)
    for tokens, subject in docs:
        n_docs[subject] += 1
        for w in set(tokens) & vocab:
            df[w][subject] += 1
    words: dict[str, Any] = {}
    for w in sorted(vocab):
        rates = {s: c / n_docs[s] for s, c in df[w].items()}
        total = sum(rates.values())
        dominant, best = max(rates.items(), key=lambda kv: (kv[1], kv[0]))
        words[w] = {
            "doc_freq": int(sum(df[w].values())),
            "dominant_subject": dominant,
            "dominant_share": best / total,
        }
    return {"subjects": sorted(n_docs), "words": words}


def _coverage(texts: Sequence[str], pre: TextPreprocessor, vocab: set[str]) -> dict[str, Any]:
    total = covered = 0
    for t in texts:
        toks = pre.tokens(t)
        total += len(toks)
        covered += sum(1 for w in toks if w in vocab)
    return {
        "tokens": total,
        "in_vocabulary": covered,
        "coverage": (covered / total) if total else None,
    }


def _neighbour_overlap(models: dict[str, Word2Vec], top_words: int = 200, k: int = 10):
    if not {"cbow", "skipgram"} <= set(models):
        return None
    a, b = models["cbow"].wv, models["skipgram"].wv
    words = [w for w in a.index_to_key[:top_words] if w in b.key_to_index]
    k = min(k, len(a) - 1)
    if not words or k < 1:
        return None
    scores = []
    for w in words:
        na = {x for x, _ in a.most_similar(w, topn=k)}
        nb = {x for x, _ in b.most_similar(w, topn=k)}
        scores.append(len(na & nb) / len(na | nb))
    return {"words_compared": len(words), "k": k, "mean_jaccard": float(np.mean(scores))}


def train_all(
    *,
    variants: Sequence[str] = VARIANTS,
    cfg: Word2VecConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
    seed: int = data_mod.DEFAULT_SEED,
) -> dict[str, Any]:
    """Train the requested variants on ``public_train`` and save the artifacts.

    Raises ``DataNotFoundError`` if the processed split is missing and
    ``Word2VecTrainingError`` if no usable model can be trained. In both cases
    nothing is written, so /health keeps reporting "not trained".
    ``metadata.json`` is written last and is the marker /health looks for.
    """
    cfg = cfg or Word2VecConfig()
    pre = TextPreprocessor(preprocess_cfg or default_preprocess_config())
    bad = [v for v in variants if v not in _SG]
    if bad or not variants:
        raise ValueError(f"variants must be a non-empty subset of {VARIANTS}; got {list(variants)}")

    # The ONLY data read for training: the public TRAIN split.
    train_df = data_mod.load_split("public_train")
    docs: list[tuple[list[str], str]] = []
    for text, subject in zip(train_df["text"], train_df["subject"]):
        toks = pre.tokens(text)
        if len(toks) >= 2:  # a single token has no context to learn from
            docs.append((toks, subject))
    if not docs:
        raise Word2VecTrainingError("public_train has no document with at least 2 usable tokens")
    corpus = [toks for toks, _ in docs]
    n_tokens = sum(len(c) for c in corpus)

    models: dict[str, Word2Vec] = {}
    per_variant: dict[str, Any] = {}
    for v in variants:
        t0 = time.perf_counter()
        try:
            m = Word2Vec(
                sentences=corpus,
                vector_size=cfg.vector_size,
                window=cfg.window,
                min_count=cfg.min_count,
                sg=_SG[v],
                negative=cfg.negative,
                sample=cfg.sample,
                epochs=cfg.epochs,
                seed=seed,
                workers=1,
                hashfxn=_stable_hash,
            )
        except (RuntimeError, ValueError) as exc:
            raise Word2VecTrainingError(
                f"gensim could not build a vocabulary with min_count={cfg.min_count}: {exc}"
            ) from exc
        if len(m.wv) < MIN_VOCAB:
            raise Word2VecTrainingError(
                f"vocabulary has only {len(m.wv)} word(s) (need >= {MIN_VOCAB}); "
                "use more data or lower --min-count"
            )
        models[v] = m
        per_variant[v] = {
            "architecture": "skip-gram" if _SG[v] else "CBOW",
            "sg": _SG[v],
            "vocab_size": len(m.wv),
            "train_seconds": round(time.perf_counter() - t0, 3),
        }

    vocab = set(next(iter(models.values())).wv.index_to_key)
    stats = _vocab_stats(docs, vocab)

    coverage: dict[str, Any] = {}
    for name, split in (("val", "public_val"), ("test", "public_test"), ("pdf_heldout", "pdf_heldout")):
        try:
            df = data_mod.load_split(split)
        except data_mod.DataNotFoundError:
            continue
        coverage[name] = _coverage(df["text"].tolist(), pre, vocab)

    metadata = {
        "model": MODEL_NAME,
        "description": "Word2Vec embeddings (CBOW and/or Skip-Gram) trained on the public train split",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gensim_version": gensim.__version__,
        "preprocessing": pre.config.to_dict(),
        "word2vec": cfg.to_dict(),
        "corpus": {
            "source": "public_train",
            "sha256": data_mod.file_sha256(data_mod.processed_path("public_train")),
            "documents": len(docs),
            "tokens": n_tokens,
            "documents_per_subject": dict(Counter(s for _, s in docs)),
        },
        "variants": per_variant,
        "vocabulary_coverage": coverage,
        "variant_comparison": _neighbour_overlap(models),
        "notes": [
            "Word2Vec is unsupervised: no accuracy/F1 exists for it. Values here are descriptive statistics.",
            "Trained on public_train only. Coverage on val/test/pdf_heldout is measured after training and never used for training or tuning.",
        ],
    }

    out = artifacts_dir()
    out.mkdir(parents=True, exist_ok=True)
    # clear the previous run, marker first, so a failed save can never look "trained"
    (out / "metadata.json").unlink(missing_ok=True)
    (out / "vocab_stats.json").unlink(missing_ok=True)
    for old in out.glob("*.model*"):
        old.unlink()
    for v, m in models.items():
        m.hashfxn = hash  # keep the pickle free of references to this module
        m.save(str(out / f"{v}.model"))
    _write_json_atomic(out / "vocab_stats.json", stats)
    _write_json_atomic(out / "metadata.json", metadata)  # marker: written last
    return metadata


# ------------------------------------------------------------ cosine helper
def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """cos(u, v) = u.v / (|u||v|), in [-1, 1]. Zero vectors give 0.0."""
    a = np.asarray(u, dtype=np.float64)
    b = np.asarray(v, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _count(wv: KeyedVectors, key: str) -> int:
    try:
        return int(wv.get_vecattr(key, "count"))
    except (KeyError, AttributeError):
        return 0


# ---------------------------------------------------------------- inference
class EmbeddingService:
    """Loads the saved Word2Vec models and answers similarity / vector / viz queries."""

    def __init__(
        self,
        vectors: dict[str, KeyedVectors],
        metadata: dict[str, Any],
        vocab_stats: dict[str, Any] | None = None,
    ):
        if not vectors:
            raise Word2VecNotTrainedError("no Word2Vec variant available")
        self.vectors = vectors
        self.metadata = metadata
        self.vocab_stats = vocab_stats or {"subjects": [], "words": {}}
        self._pre = TextPreprocessor(PreprocessConfig.from_dict(metadata["preprocessing"]))

    # -- loading --------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path | None = None) -> "EmbeddingService":
        d = Path(directory) if directory else artifacts_dir()
        meta_file = d / "metadata.json"
        if not meta_file.is_file():
            raise Word2VecNotTrainedError(
                "Word2Vec model not trained yet (artifacts/word2vec/metadata.json is missing). "
                "Run: python -m training.prepare_data ... then python -m training.train_word2vec"
            )
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        vectors: dict[str, KeyedVectors] = {}
        for v in metadata.get("variants", {}):
            f = d / f"{v}.model"
            if not f.is_file():
                raise Word2VecNotTrainedError(
                    f"metadata.json lists '{v}' but {f.name} is missing. "
                    "Re-run training (model files are not committed to git)."
                )
            vectors[v] = Word2Vec.load(str(f)).wv
        stats_file = d / "vocab_stats.json"
        stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.is_file() else None
        return cls(vectors, metadata, stats)

    @property
    def variants(self) -> list[str]:
        return [v for v in VARIANTS if v in self.vectors]

    @property
    def trained_at(self) -> str | None:
        return self.metadata.get("trained_at")

    # -- helpers --------------------------------------------------------
    def _wv(self, variant: str | None) -> tuple[str, KeyedVectors]:
        if variant is None:
            variant = DEFAULT_VARIANT if DEFAULT_VARIANT in self.vectors else self.variants[0]
        if variant not in self.vectors:
            raise UnknownVariantError(variant, self.variants)
        return variant, self.vectors[variant]

    def resolve(self, word: str) -> str | None:
        """Map user input to the token form used in training (same preprocessing,
        but without stop-word removal). Returns None unless it is exactly one token."""
        pre = self._pre
        toks = pre.normalize(pre.tokenize(pre.clean(word)))
        return toks[0] if len(toks) == 1 else None

    def _require(self, words: Sequence[str], variant: str, wv: KeyedVectors) -> list[str]:
        keys, missing = [], []
        for w in words:
            k = self.resolve(w)
            if k is None or k not in wv.key_to_index:
                missing.append(w)
            else:
                keys.append(k)
        if missing:
            raise WordNotInVocabularyError(missing, variant)
        return keys

    # -- queries --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        out = dict(self.metadata)
        out.update(
            trained=True,
            status="trained",
            variants_available=self.variants,
            default_variant=DEFAULT_VARIANT if DEFAULT_VARIANT in self.vectors else self.variants[0],
        )
        return out

    def vector(self, word: str, variant: str | None = None) -> dict[str, Any]:
        variant, wv = self._wv(variant)
        (key,) = self._require([word], variant, wv)
        vec = wv[key]
        return {
            "word": key,
            "variant": variant,
            "dimension": int(wv.vector_size),
            "count": _count(wv, key),
            "norm": float(np.linalg.norm(vec)),
            "vector": [float(x) for x in vec],
        }

    def similarity(self, word_a: str, word_b: str, variant: str | None = None) -> dict[str, Any]:
        variant, wv = self._wv(variant)
        ka, kb = self._require([word_a, word_b], variant, wv)
        return {
            "variant": variant,
            "word_a": ka,
            "word_b": kb,
            "cosine_similarity": cosine_similarity(wv[ka], wv[kb]),
        }

    def most_similar(self, word: str, topn: int = 10, variant: str | None = None) -> dict[str, Any]:
        variant, wv = self._wv(variant)
        (key,) = self._require([word], variant, wv)
        neighbours = wv.most_similar(key, topn=topn)
        return {
            "query": key,
            "variant": variant,
            "neighbors": [
                {"word": w, "similarity": float(s), "count": _count(wv, w)} for w, s in neighbours
            ],
        }

    def analogy(self, a: str, b: str, c: str, topn: int = 5, variant: str | None = None) -> dict[str, Any]:
        """a is to b as c is to ?   (vector arithmetic: b - a + c, cosine nearest)."""
        variant, wv = self._wv(variant)
        ka, kb, kc = self._require([a, b, c], variant, wv)
        results = wv.most_similar(positive=[kb, kc], negative=[ka], topn=topn)
        return {
            "variant": variant,
            "question": f"{ka} : {kb} :: {kc} : ?",
            "a": ka,
            "b": kb,
            "c": kc,
            "results": [{"word": w, "similarity": float(s)} for w, s in results],
            "caveat": ANALOGY_CAVEAT,
        }

    def visualize(
        self,
        *,
        method: str = "pca",
        variant: str | None = None,
        words: Sequence[str] | None = None,
        top_n: int = 100,
        neighbors_per_word: int = 0,
        perplexity: float | None = None,
        seed: int = data_mod.DEFAULT_SEED,
        max_points: int = MAX_VIS_POINTS,
    ) -> dict[str, Any]:
        """2-D coordinates (PCA or t-SNE) of selected embeddings, as plain JSON."""
        if method not in ("pca", "tsne"):
            raise VisualizationError(f"method must be 'pca' or 'tsne', got {method!r}")
        variant, wv = self._wv(variant)

        source: dict[str, str] = {}
        chosen: list[str] = []
        missing: list[str] = []
        if words:
            if len(words) > max_points:
                raise VisualizationError(f"at most {max_points} words per request")
            for w in words:
                k = self.resolve(w)
                if k is None or k not in wv.key_to_index:
                    missing.append(w)
                elif k not in source:
                    source[k] = "requested"
                    chosen.append(k)
            if not chosen:
                raise WordNotInVocabularyError(missing, variant)
            if neighbors_per_word > 0:
                for k in list(chosen):
                    for nb, _ in wv.most_similar(k, topn=neighbors_per_word):
                        if nb not in source and len(chosen) < max_points:
                            source[nb] = "neighbor"
                            chosen.append(nb)
        else:
            if not 1 <= top_n <= max_points:
                raise VisualizationError(f"top_n must be between 1 and {max_points}")
            for k in wv.index_to_key[:top_n]:
                source[k] = "frequent"
                chosen.append(k)

        n = len(chosen)
        need = 3 if method == "pca" else 5
        if n < need:
            raise VisualizationError(
                f"{method} needs at least {need} words in the vocabulary; got {n}"
            )
        X = np.asarray(wv[chosen], dtype=np.float64)

        extra: dict[str, Any]
        if method == "pca":
            pca = PCA(n_components=2, svd_solver="full", random_state=seed)
            coords = pca.fit_transform(X)
            extra = {"explained_variance_ratio": [float(r) for r in pca.explained_variance_ratio_]}
            params: dict[str, Any] = {"seed": seed}
            notes = ["Linear projection. explained_variance_ratio is the share of variance each axis keeps."]
        else:
            perp = float(min(perplexity if perplexity is not None else 30.0, (n - 1) / 3.0))
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                metric="cosine",
                random_state=seed,
                n_jobs=1,
            )
            coords = tsne.fit_transform(X)
            extra = {"kl_divergence": float(tsne.kl_divergence_)}
            params = {"seed": seed, "perplexity": perp, "metric": "cosine"}
            notes = [
                "t-SNE keeps local neighbourhoods only: distances between clusters and cluster sizes are not meaningful.",
                "The layout depends on perplexity and seed.",
            ]
        notes.append(
            "dominant_subject: subject whose TRAIN questions contain the word most often relative to class size; "
            "dominant_share 1.0 = the word occurs only in that subject; a share near 1/(number of subjects) "
            "means a general word that appears in every subject (the named subject is then just a tie-break)."
        )

        stats = self.vocab_stats.get("words", {})
        points = []
        for w, (x, y) in zip(chosen, coords):
            s = stats.get(w, {})
            points.append(
                {
                    "word": w,
                    "x": float(x),
                    "y": float(y),
                    "count": _count(wv, w),
                    "source": source[w],
                    "dominant_subject": s.get("dominant_subject"),
                    "dominant_share": s.get("dominant_share"),
                }
            )
        return {
            "method": method,
            "variant": variant,
            "n_points": n,
            "parameters": params,
            **extra,
            "missing_words": missing,
            "subjects": self.vocab_stats.get("subjects", []),
            "points": points,
            "notes": notes,
        }
