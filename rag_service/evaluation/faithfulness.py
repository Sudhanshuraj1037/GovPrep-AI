"""
evaluation/faithfulness.py

Stage 5: Explainable AI — Faithfulness Checker & Citation Tracker.

This is the "flagship touch" that separates a production system
from a toy project. It answers: "Is the LLM's answer actually
supported by the retrieved documents, or did it hallucinate?"

Two complementary methods:
  1. Lexical Faithfulness  — word overlap (fast, no model needed)
  2. Semantic Faithfulness — cosine similarity of embeddings (more accurate)

Interview talking point:
  "Faithfulness is one of the four RAGAS metrics. I implemented a
   lightweight version that runs locally: I check what fraction of
   key phrases in the generated answer can be traced back to the
   retrieved context. A score below 0.25 flags the answer for
   human review — this is my automated hallucination detector."
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ml_core.vector_store import RetrievedChunk


# ── Result model ───────────────────────────────────────────────
@dataclass
class FaithfulnessResult:
    """
    Output of the faithfulness check for one RAG response.
    Stored alongside the answer for offline evaluation.
    """
    score: float              # 0.0 (hallucinated) → 1.0 (fully grounded)
    method: str               # "lexical" or "semantic"
    is_faithful: bool         # score >= threshold
    supporting_chunks: List[str]   # chunk IDs that support the answer
    unsupported_claims: List[str]  # sentences with low support
    verdict: str              # human-readable verdict


class FaithfulnessChecker:
    """
    Checks whether the LLM's generated answer is grounded in
    the retrieved context chunks.

    Methodology:
      1. Split the answer into individual sentences (claims)
      2. For each claim, compute overlap with each context chunk
      3. A claim is "supported" if any chunk has overlap >= threshold
      4. Faithfulness score = fraction of supported claims
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def check(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> FaithfulnessResult:
        """
        Run lexical faithfulness check.
        Fast (~1ms), no model loading required.
        """
        if not retrieved_chunks or not answer.strip():
            return FaithfulnessResult(
                score=0.0, method="lexical", is_faithful=False,
                supporting_chunks=[], unsupported_claims=[answer],
                verdict="No context available to verify answer.",
            )

        # Combine all retrieved context into one string
        full_context = " ".join(c.text.lower() for c in retrieved_chunks)
        context_words = set(self._tokenize(full_context))

        # Split answer into sentences (claims)
        claims = self._split_sentences(answer)
        if not claims:
            return FaithfulnessResult(
                score=1.0, method="lexical", is_faithful=True,
                supporting_chunks=[], unsupported_claims=[],
                verdict="Answer is empty — nothing to verify.",
            )

        supported_claims = []
        unsupported_claims = []
        supporting_chunk_ids = set()

        for claim in claims:
            claim_words = set(self._tokenize(claim))
            if len(claim_words) < 3:
                # Skip very short sentences (e.g. "Yes.", "Okay.")
                continue

            # Find which chunk best supports this claim
            best_overlap = 0.0
            best_chunk_id = None

            for chunk in retrieved_chunks:
                chunk_words = set(self._tokenize(chunk.text))
                overlap = self._jaccard_overlap(claim_words, chunk_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk_id = chunk.chunk_id

            if best_overlap >= self.threshold:
                supported_claims.append(claim)
                if best_chunk_id:
                    supporting_chunk_ids.add(best_chunk_id)
            else:
                unsupported_claims.append(claim)

        total = len(supported_claims) + len(unsupported_claims)
        score = len(supported_claims) / total if total > 0 else 0.0
        is_faithful = score >= self.threshold

        verdict = self._generate_verdict(score, unsupported_claims)

        logger.info(
            f"Faithfulness: {score:.2f} | "
            f"Supported: {len(supported_claims)}/{total} claims | "
            f"{'✅ FAITHFUL' if is_faithful else '⚠️ FLAGGED'}"
        )

        return FaithfulnessResult(
            score=round(score, 3),
            method="lexical",
            is_faithful=is_faithful,
            supporting_chunks=list(supporting_chunk_ids),
            unsupported_claims=unsupported_claims[:3],  # top 3 for display
            verdict=verdict,
        )

    def check_semantic(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> FaithfulnessResult:
        """
        Semantic faithfulness check using embedding cosine similarity.
        More accurate than lexical but requires the embedding model.

        Interview talking point:
          "Semantic faithfulness catches paraphrasing that lexical
           overlap misses. If the LLM rephrases a fact from the context
           in different words, lexical check flags it — but semantic
           check correctly marks it as supported."
        """
        from ml_core.embeddings import EmbeddingGenerator
        embedder = EmbeddingGenerator()

        claims = self._split_sentences(answer)
        substantive = [c for c in claims if len(c.split()) >= 5]

        if not substantive or not retrieved_chunks:
            return self.check(answer, retrieved_chunks)

        context_texts = [c.text for c in retrieved_chunks]
        context_embeddings = embedder.embed_texts(context_texts, show_progress=False)

        supported = []
        unsupported = []
        supporting_ids = set()

        for claim in substantive:
            claim_emb = embedder.embed_query(claim)
            similarities = embedder.compute_similarity(claim_emb, context_embeddings)
            best_score = max(similarities)
            best_idx = int(np.argmax(similarities))

            if best_score >= self.threshold:
                supported.append(claim)
                supporting_ids.add(retrieved_chunks[best_idx].chunk_id)
            else:
                unsupported.append(claim)

        total = len(supported) + len(unsupported)
        score = len(supported) / total if total > 0 else 0.0

        return FaithfulnessResult(
            score=round(score, 3),
            method="semantic",
            is_faithful=score >= self.threshold,
            supporting_chunks=list(supporting_ids),
            unsupported_claims=unsupported[:3],
            verdict=self._generate_verdict(score, unsupported),
        )

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase + remove punctuation + split on whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 2]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on '.', '!', '?'"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _jaccard_overlap(set_a: set, set_b: set) -> float:
        """
        Jaccard similarity = |intersection| / |union|
        Ranges from 0.0 (no overlap) to 1.0 (identical).
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    @staticmethod
    def _generate_verdict(score: float, unsupported: List[str]) -> str:
        if score >= 0.8:
            return "✅ Highly faithful — answer is well-grounded in the retrieved context."
        elif score >= 0.5:
            return f"⚠️ Partially faithful — {len(unsupported)} claim(s) could not be verified."
        elif score >= 0.25:
            return f"🔶 Low faithfulness — significant portions may not be from the source material."
        else:
            return "❌ Unfaithful — answer appears to contain information not in retrieved context."


# ── Citation Tracker ───────────────────────────────────────────

class CitationTracker:
    """
    Tracks which specific parts of the retrieved context
    were used in the final answer.

    Produces an annotated version of the answer where
    each sentence is tagged with its source chunk.
    """

    def annotate(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of {sentence, source_citation, confidence} dicts.
        Used to render inline citations in the frontend.
        """
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        annotated = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            best_chunk = None
            best_overlap = 0.0
            sentence_words = set(sentence.lower().split())

            for chunk in retrieved_chunks:
                chunk_words = set(chunk.text.lower().split())
                overlap = len(sentence_words & chunk_words) / max(len(sentence_words), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk = chunk

            annotated.append({
                "sentence": sentence,
                "source_citation": best_chunk.citation() if best_chunk and best_overlap > 0.1 else None,
                "confidence": round(best_overlap, 3),
                "grounded": best_overlap >= 0.1,
            })

        return annotated
