"""
ml_core/vector_store.py
Stage 3: Vector Database using ChromaDB (local, persistent).

ChromaDB stores text + embeddings + metadata together.
At query time it finds k-nearest neighbours using HNSW algorithm
(Hierarchical Navigable Small World) - O(log n) complexity.

Interview talking point:
  "HNSW is an approximate nearest-neighbour algorithm. It builds
   a graph where each node connects to its nearest neighbours at
   multiple hierarchy levels. Search starts at the top level
   (few nodes, wide coverage) and drills down - similar to binary
   search but for high-dimensional vector spaces."
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from data_pipeline.pdf_loader import TextChunk
from ml_core.embeddings import EmbeddingGenerator


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with its relevance score."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    subject: str
    chapter: Optional[str]
    relevance_score: float

    def citation(self) -> str:
        parts = [f"[{self.source_file}"]
        if self.chapter:
            parts.append(f" | {self.chapter}")
        parts.append(f" | Page {self.page_number}]")
        return "".join(parts)


class VectorStore:
    """
    Manages ChromaDB collection for semantic search.

    Lifecycle:
      1. add_chunks()  - during ingestion to populate the DB
      2. search()      - at query time to retrieve relevant context
      3. get_stats()   - for health checks
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info(f"ChromaDB initialised at {settings.CHROMA_PERSIST_DIR}")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Collection: {settings.CHROMA_COLLECTION_NAME} "
                f"({self._collection.count()} docs)"
            )
        return self._collection

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Embed and upsert chunks. Skips already-ingested IDs (idempotent)."""
        if not chunks:
            return
        collection = self._get_collection()
        existing = set(collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            logger.info("All chunks already in vector store.")
            return
        logger.info(f"Adding {len(new_chunks)} new chunks...")
        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "subject": c.subject,
                "chapter": c.chapter or "",
                "word_count": c.word_count,
            }
            for c in new_chunks
        ]
        # Upsert in batches of 100
        for i in range(0, len(new_chunks), 100):
            collection.upsert(
                ids=ids[i:i+100],
                embeddings=embeddings[i:i+100],
                documents=texts[i:i+100],
                metadatas=metadatas[i:i+100],
            )
        logger.info(f"Vector store total: {collection.count()} docs")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        subject_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Semantic similarity search.
        Converts query to vector, finds k-nearest neighbours in ChromaDB.
        """
        k = top_k or settings.TOP_K_RESULTS
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("Vector store is empty. Ingest PDFs first.")
            return []
        query_emb = self.embedder.embed_query(query)
        where = {"subject": {"$eq": subject_filter}} if subject_filter else None
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Convert to similarity 0->1
            similarity = 1.0 - (dist / 2.0)
            if similarity < settings.MIN_RELEVANCE_SCORE:
                continue
            retrieved.append(RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                subject=meta.get("subject", ""),
                chapter=meta.get("chapter") or None,
                relevance_score=round(similarity, 4),
            ))
        logger.info(f"Retrieved {len(retrieved)} chunks for: {query[:60]}")
        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        collection = self._get_collection()
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": collection.count(),
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dim": self.embedder.embedding_dim,
        }

    def reset(self) -> None:
        """Delete all vectors. Dev/testing only."""
        self._get_client().delete_collection(settings.CHROMA_COLLECTION_NAME)
        self._collection = None
        logger.warning("Vector store reset.")
