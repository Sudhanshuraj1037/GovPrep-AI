"""
ml_core/rag_pipeline.py

Stage 4 of the ML Pipeline: Agentic-RAG "Search-and-Synthesize".

Flow for every student query:
  1. RETRIEVE  → Search ChromaDB for top-k relevant chunks
  2. RERANK    → Score chunks by relevance + recency (lightweight agent step)
  3. GROUND    → Build a context-stuffed prompt (no hallucination)
  4. GENERATE  → Call Groq Llama-3 with the grounded prompt
  5. CITE      → Attach source citations to the answer

Interview talking point:
  "This is 'grounded generation' — the LLM only answers from the
   retrieved context. If the context doesn't contain the answer,
   the system says so instead of hallucinating. This is the key
   difference between a toy chatbot and a production RAG system."
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from groq import Groq
from loguru import logger

from config.settings import settings
from ml_core.vector_store import VectorStore, RetrievedChunk


# ── Response model ─────────────────────────────────────────────
@dataclass
class RAGResponse:
    """
    Everything the API returns for one student query.
    Structured so the frontend can display answer + citations separately.
    """
    answer: str                              # LLM's final answer
    sources: List[Dict[str, Any]]            # cited chunks
    retrieved_chunks: List[RetrievedChunk]   # all retrieved (for eval)
    query: str
    model_used: str
    latency_ms: float
    faithfulness_score: float = 0.0          # filled by evaluator
    context_used: bool = True                # False if no chunks found
    grounded_prompt: str = ""                # stored for RAGAS evaluation


# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are GovPrep AI — an expert tutor for Indian government competitive exams (Railway RRB, SSC CGL, UPSC, Banking IBPS).

STRICT RULES (follow without exception):
1. Answer ONLY using information present in the CONTEXT provided below.
2. If the context does not contain enough information, say:
   "I could not find a reliable answer in the available study material. Please check a textbook or official source."
3. Never fabricate facts, dates, names, or statistics not present in the context.
4. Keep answers concise, accurate, and exam-focused.
5. Always end with: "📚 Source: [cite the source file and page from context]"
6. If the question is in Hindi, answer in Hindi. Otherwise answer in English.

CONTEXT:
{context}

Remember: Your knowledge is LIMITED to the context above."""

USER_PROMPT_TEMPLATE = """Student Question: {question}

Provide a clear, accurate answer based strictly on the context provided. Include relevant facts, dates, or formulas if present in the context."""


# ── RAG Pipeline ───────────────────────────────────────────────
class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation cycle.

    This is the "Agent" in Agentic-RAG:
      - It DECIDES whether retrieved context is sufficient
      - It SELECTS and RERANKS the most relevant chunks
      - It CONSTRUCTS the grounded prompt
      - It CALLS the LLM only with verified context
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

    # ── Main entry point ───────────────────────────────────────

    def answer(
        self,
        query: str,
        subject_filter: Optional[str] = None,
        exam_context: Optional[str] = None,
    ) -> RAGResponse:
        """
        Full RAG cycle: retrieve → rerank → ground → generate → cite.

        Args:
            query:          Student's question
            subject_filter: Optionally restrict retrieval to one subject
            exam_context:   e.g. "Railway RRB NTPC" — injected into prompt

        Returns:
            RAGResponse with answer, sources, and evaluation metadata
        """
        start_time = time.time()
        logger.info(f"RAG query: '{query[:80]}...'")

        # ── Step 1: Retrieve ──────────────────────────────────
        chunks = self.vector_store.search(
            query=query,
            top_k=settings.TOP_K_RESULTS,
            subject_filter=subject_filter,
        )

        # ── Step 2: Rerank (lightweight agent step) ───────────
        ranked_chunks = self._rerank(chunks, query)

        # ── Step 3: Build grounded prompt ─────────────────────
        if not ranked_chunks:
            logger.warning("No relevant chunks found — using fallback response")
            return self._no_context_response(query, time.time() - start_time)

        context_str = self._build_context_string(ranked_chunks)
        grounded_prompt = SYSTEM_PROMPT.format(context=context_str)
        user_message = USER_PROMPT_TEMPLATE.format(question=query)

        if exam_context:
            user_message = f"[Exam: {exam_context}]\n{user_message}"

        # ── Step 4: Generate with Groq Llama-3 ───────────────
        answer_text = self._call_groq(grounded_prompt, user_message)

        # ── Step 5: Build citations ────────────────────────────
        sources = self._extract_citations(ranked_chunks)

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"RAG completed in {latency_ms:.0f}ms")

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            retrieved_chunks=ranked_chunks,
            query=query,
            model_used=settings.GROQ_MODEL,
            latency_ms=round(latency_ms, 1),
            context_used=True,
            grounded_prompt=grounded_prompt,
        )

    # ── Step 2: Reranking ──────────────────────────────────────

    def _rerank(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """
        Lightweight reranking using a combined score:
          final_score = 0.7 * semantic_similarity + 0.3 * keyword_overlap

        This is a simple but effective heuristic that improves precision
        without requiring a separate cross-encoder model (saves RAM).

        Interview talking point:
          "In production I would use a cross-encoder for reranking,
           but on constrained hardware I use a hybrid score that
           combines dense (vector) and sparse (keyword) signals."
        """
        if not chunks:
            return []

        query_keywords = set(query.lower().split())

        def hybrid_score(chunk: RetrievedChunk) -> float:
            # Sparse: keyword overlap ratio
            chunk_words = set(chunk.text.lower().split())
            keyword_overlap = len(query_keywords & chunk_words) / max(len(query_keywords), 1)
            # Hybrid
            return 0.7 * chunk.relevance_score + 0.3 * keyword_overlap

        scored = sorted(chunks, key=hybrid_score, reverse=True)

        # Take top 3 for the prompt — keeps token count manageable
        top_chunks = scored[:3]
        logger.debug(f"Reranked: top scores = {[c.relevance_score for c in top_chunks]}")
        return top_chunks

    # ── Step 3: Context building ───────────────────────────────

    def _build_context_string(self, chunks: List[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into a numbered context block.
        Numbers make it easy for the LLM to reference specific chunks.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Chunk {i}] Source: {chunk.source_file} | "
                f"Page {chunk.page_number} | "
                f"Subject: {chunk.subject}"
                + (f" | Chapter: {chunk.chapter}" if chunk.chapter else "")
                + f"\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Step 4: Groq API call ──────────────────────────────────

    def _call_groq(self, system_prompt: str, user_message: str) -> str:
        """
        Calls Groq's Llama-3 with the grounded system prompt.
        Temperature=0.2 for factual, consistent answers.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    # ── Step 5: Citations ──────────────────────────────────────

    def _extract_citations(
        self, chunks: List[RetrievedChunk]
    ) -> List[Dict[str, Any]]:
        """
        Returns a deduplicated list of source citations.
        Frontend displays these as clickable source tags.
        """
        seen = set()
        citations = []
        for chunk in chunks:
            key = (chunk.source_file, chunk.page_number)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "subject": chunk.subject,
                    "chapter": chunk.chapter,
                    "relevance_score": chunk.relevance_score,
                    "citation": chunk.citation(),
                })
        return citations

    # ── Fallback when no context found ────────────────────────

    def _no_context_response(
        self, query: str, elapsed: float
    ) -> RAGResponse:
        return RAGResponse(
            answer=(
                "❌ I could not find relevant information in the uploaded study material "
                "for this question. Please upload relevant PDFs or try rephrasing your question.\n\n"
                "💡 Tip: Make sure you have ingested PDFs covering this topic using the "
                "'/ingest' endpoint first."
            ),
            sources=[],
            retrieved_chunks=[],
            query=query,
            model_used=settings.GROQ_MODEL,
            latency_ms=round(elapsed * 1000, 1),
            faithfulness_score=0.0,
            context_used=False,
        )
