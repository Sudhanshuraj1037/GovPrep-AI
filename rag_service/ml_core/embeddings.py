"""
ml_core/embeddings.py
Stage 2: Embedding Generation using sentence-transformers/all-MiniLM-L6-v2.
Memory-efficient batch processing for 8GB RAM systems.

Model: all-MiniLM-L6-v2
  - Size:       ~90MB
  - Dimensions: 384
  - Speed:      ~14,000 sentences/sec on CPU
  - RAM usage:  ~400MB loaded
"""
from __future__ import annotations
import gc, time
from typing import List
import numpy as np
from loguru import logger

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from config.settings import settings
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        t = time.time()
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
        logger.info(f"Model loaded in {time.time()-t:.1f}s | dim={_model.get_sentence_embedding_dimension()}")
    return _model


class EmbeddingGenerator:
    """
    Wraps SentenceTransformer with memory-safe batch processing.
    Processes texts in batches of 32 and calls gc.collect() after each
    batch to keep peak RAM under 1.5GB even for large document sets.
    """

    def __init__(self, batch_size: int = 32):
        from config.settings import settings
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    @property
    def model(self):
        return _get_model()

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed a list of texts in memory-safe batches."""
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(f"Embedding {len(texts)} texts in {n_batches} batches")
        for i in range(n_batches):
            batch = texts[i * self.batch_size:(i + 1) * self.batch_size]
            vecs = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.extend(vecs.tolist())
            del vecs
            gc.collect()
        logger.info(f"Embedding complete: ({len(all_embeddings)}, {self.embedding_dim})")
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query — low-latency path (no batching overhead)."""
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].tolist()

    def compute_similarity(self, query_emb: List[float], doc_embs: List[List[float]]) -> List[float]:
        """
        Cosine similarity between one query and many docs.
        Since embeddings are L2-normalised: cosine_sim = dot product.
        """
        q = np.array(query_emb)
        D = np.array(doc_embs)
        return (D @ q).tolist()
