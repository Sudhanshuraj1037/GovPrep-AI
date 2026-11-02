"""Reusable NLP preprocessing pipeline (shared by TF-IDF now, and by Word2Vec,
LSTM and BiLSTM in later phases).

Stages, in order:

1. clean       Unicode NFKC, strip control chars, URLs, e-mails, "Page 3 of 70"
               markers, possessives ('s) and contractions
2. lowercase
3. tokenize    regex tokenizer; keeps digits (years such as 1526 matter) and
               Devanagari so Hindi text is not destroyed
4. stop-words  scikit-learn's English list (no download needed). Negations
               ("not", "no", "never"...) are kept by default because exam stems
               such as "which is NOT correct" change meaning without them.
5. normalise   "stem"  -> Porter stemmer (default, needs no corpus download)
               "lemma" -> WordNet lemmatiser (needs: python -m nltk.downloader wordnet)
               "none"

Scope note: stop-word removal and stemming are English-only. Devanagari tokens
are preserved as-is. Proper Hindi support is left to the multilingual BERT
phase.

The same ``PreprocessConfig`` is saved inside every trained artifact so that
inference preprocesses text exactly like training did.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATIONS = frozenset(
    {"no", "not", "nor", "never", "none", "nothing", "neither", "cannot",
     "without", "nobody", "nowhere"}
)
NORMALIZERS = ("none", "stem", "lemma")

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PAGE_RE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
# control characters -> space; zero-width / BOM characters -> removed.
# (U+200C / U+200D are deliberately NOT removed: Devanagari conjuncts need them.)
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f]")
_ZW_RE = re.compile(r"[\u200b\u200e\u200f\u2060\ufeff]")
_APOS_RE = re.compile(r"[\u2019\u2018`\u00b4]")
_NT_RE = re.compile(r"\b([A-Za-z]+)n't\b", re.IGNORECASE)
_POSS_RE = re.compile(r"'s\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# ASCII letters/digits, Latin-1/Extended-A/B letters, Devanagari (minus the
# danda punctuation U+0964/0965), ZWJ/ZWNJ.
_TOKEN_RE = re.compile(
    r"[0-9A-Za-z\u00c0-\u024f\u0900-\u0963\u0966-\u097f\u200c\u200d]+"
)


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    strip_urls_emails: bool = True
    strip_page_markers: bool = True
    remove_stopwords: bool = True
    keep_negations: bool = True
    normalizer: str = "stem"          # "none" | "stem" | "lemma"
    min_token_len: int = 2

    def __post_init__(self) -> None:
        if self.normalizer not in NORMALIZERS:
            raise ValueError(
                f"normalizer must be one of {NORMALIZERS}, got {self.normalizer!r}"
            )
        if self.min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@lru_cache(maxsize=1)
def _stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "nltk is required for normalizer='stem'. Run: pip install -r requirements.txt"
        ) from exc
    return PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    return _stemmer().stem(token)


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer

    lem = WordNetLemmatizer()
    try:
        lem.lemmatize("tests")  # forces the WordNet corpus to load
    except LookupError as exc:
        raise LookupError(
            "WordNet data is missing for normalizer='lemma'. "
            "Run: python -m nltk.downloader wordnet"
        ) from exc
    return lem


def _lemma(token: str) -> str:
    lem = _lemmatizer()
    out = lem.lemmatize(token, "n")
    return out if out != token else lem.lemmatize(token, "v")


def _is_latin_word(token: str) -> bool:
    return token.isascii() and token.isalpha()


class TextPreprocessor:
    """Callable preprocessing pipeline: ``TextPreprocessor()(text) -> list[str]``."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        stop = set(ENGLISH_STOP_WORDS)
        if self.config.keep_negations:
            stop -= NEGATIONS
        self._stopwords = frozenset(stop)

    # -- stage 1 ---------------------------------------------------------
    def clean(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        c = self.config
        text = unicodedata.normalize("NFKC", text)
        text = _ZW_RE.sub("", text)
        text = _CTRL_RE.sub(" ", text)
        if c.strip_urls_emails:
            text = _URL_RE.sub(" ", text)
            text = _EMAIL_RE.sub(" ", text)
        if c.strip_page_markers:
            text = _PAGE_RE.sub(" ", text)
        text = _APOS_RE.sub("'", text)
        text = _NT_RE.sub(r"\1 not", text)
        text = _POSS_RE.sub("", text)
        text = text.replace("'", "")
        if c.lowercase:
            text = text.lower()
        return _WS_RE.sub(" ", text).strip()

    # -- stage 2 ---------------------------------------------------------
    def tokenize(self, cleaned: str) -> list[str]:
        n = self.config.min_token_len
        return [t for t in _TOKEN_RE.findall(cleaned) if len(t) >= n]

    # -- stage 3 ---------------------------------------------------------
    def drop_stopwords(self, tokens: list[str]) -> list[str]:
        if not self.config.remove_stopwords:
            return tokens
        return [t for t in tokens if t.lower() not in self._stopwords]

    # -- stage 4 ---------------------------------------------------------
    def normalize(self, tokens: list[str]) -> list[str]:
        mode = self.config.normalizer
        if mode == "none":
            return tokens
        fn = _stem if mode == "stem" else _lemma
        return [fn(t) if _is_latin_word(t) else t for t in tokens]

    # -- full pipeline ---------------------------------------------------
    def tokens(self, text: Any) -> list[str]:
        return self.normalize(self.drop_stopwords(self.tokenize(self.clean(text))))

    __call__ = tokens

    def to_string(self, text: Any) -> str:
        """Space-joined tokens; this is what the TF-IDF vectoriser consumes."""
        return " ".join(self.tokens(text))

    def explain(self, text: Any) -> dict[str, Any]:
        """Every stage's output, for the API / viva demo."""
        cleaned = self.clean(text)
        tokenized = self.tokenize(cleaned)
        no_stop = self.drop_stopwords(tokenized)
        final = self.normalize(no_stop)
        kept = set(no_stop)
        return {
            "config": self.config.to_dict(),
            "cleaned": cleaned,
            "tokens": tokenized,
            "after_stopword_removal": no_stop,
            "final_tokens": final,
            "removed_stopwords": [t for t in tokenized if t not in kept],
        }
