"""
tests/test_rag_pipeline.py

Unit tests for the core ML pipeline components.
Run with: pytest tests/ -v

Interview talking point:
  "I wrote unit tests for each stage of the ML pipeline independently —
   data ingestion, chunking, embeddings, retrieval, and faithfulness.
   This makes it easy to isolate failures and ensures the system is
   production-ready, not just a Jupyter notebook demo."
"""

import sys
import os
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════
#  Test: Data Pipeline
# ══════════════════════════════════════════════════════════════

class TestPDFExtractor:
    def test_clean_text_removes_extra_whitespace(self):
        from data_pipeline.pdf_loader import PDFExtractor
        raw = "Hello   World\n\n\n\nTest"
        cleaned = PDFExtractor._clean_text(raw)
        assert "   " not in cleaned
        assert cleaned.count("\n\n") <= 1

    def test_clean_text_removes_standalone_numbers(self):
        from data_pipeline.pdf_loader import PDFExtractor
        raw = "Some content\n42\nMore content"
        cleaned = PDFExtractor._clean_text(raw)
        assert "\n42\n" not in cleaned


class TestDocumentChunker:
    def test_chunking_creates_correct_structure(self):
        from data_pipeline.pdf_loader import DocumentChunker, TextChunk
        chunker = DocumentChunker()
        pages = [
            {"page_number": 1, "text": "The Mughal Empire was founded by Babur in 1526. " * 20},
            {"page_number": 2, "text": "Akbar was the greatest Mughal emperor. " * 20},
        ]
        chunks = chunker.chunk_pages(pages, "test.pdf", "History", "Mughal Empire")
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.subject == "History" for c in chunks)
        assert all(c.chapter == "Mughal Empire" for c in chunks)
        assert all(len(c.text) >= 50 for c in chunks)

    def test_chunk_ids_are_unique(self):
        from data_pipeline.pdf_loader import DocumentChunker
        chunker = DocumentChunker()
        pages = [{"page_number": i, "text": f"Content for page {i}. " * 30} for i in range(1, 4)]
        chunks = chunker.chunk_pages(pages, "test.pdf", "Science")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_short_chunks_are_filtered(self):
        from data_pipeline.pdf_loader import DocumentChunker
        chunker = DocumentChunker()
        pages = [{"page_number": 1, "text": "Short."}]
        chunks = chunker.chunk_pages(pages, "test.pdf", "Math")
        assert len(chunks) == 0, "Chunks shorter than 50 chars should be filtered"


# ══════════════════════════════════════════════════════════════
#  Test: Embeddings
# ══════════════════════════════════════════════════════════════

class TestEmbeddingGenerator:
    def test_embed_query_returns_correct_dimensions(self):
        from ml_core.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        vector = gen.embed_query("What is the capital of India?")
        assert isinstance(vector, list)
        assert len(vector) == 384, "all-MiniLM-L6-v2 produces 384-dim vectors"

    def test_embed_texts_batch(self):
        from ml_core.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        texts = [
            "Babur founded the Mughal Empire.",
            "SSC CGL has four tiers.",
            "F = ma is Newton's second law.",
        ]
        embeddings = gen.embed_texts(texts, show_progress=False)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_similar_sentences_have_higher_similarity(self):
        from ml_core.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        q = gen.embed_query("Who founded the Mughal Empire?")
        similar = gen.embed_query("Babur was the founder of the Mughal Empire in 1526.")
        unrelated = gen.embed_query("The formula for compound interest is A = P(1+r/n)^nt")
        sims = gen.compute_similarity(q, [similar, unrelated])
        assert sims[0] > sims[1], "Similar texts should have higher cosine similarity"

    def test_embeddings_are_normalised(self):
        import numpy as np
        from ml_core.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        vec = gen.embed_query("Test sentence")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, "Embeddings should be L2-normalised"


# ══════════════════════════════════════════════════════════════
#  Test: Faithfulness Checker
# ══════════════════════════════════════════════════════════════

class TestFaithfulnessChecker:
    def _make_chunk(self, text: str, chunk_id: str = "test_chunk"):
        from ml_core.vector_store import RetrievedChunk
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            source_file="test.pdf",
            page_number=1,
            subject="History",
            chapter="Test",
            relevance_score=0.9,
        )

    def test_high_faithfulness_when_answer_matches_context(self):
        from evaluation.faithfulness import FaithfulnessChecker
        checker = FaithfulnessChecker(threshold=0.1)
        context_text = (
            "Babur founded the Mughal Empire in 1526 after his victory at "
            "the First Battle of Panipat against Ibrahim Lodi."
        )
        answer = (
            "Babur founded the Mughal Empire in 1526. "
            "He defeated Ibrahim Lodi at Panipat."
        )
        chunk = self._make_chunk(context_text)
        result = checker.check(answer, [chunk])
        assert result.score > 0.3, f"Expected high faithfulness, got {result.score}"

    def test_low_faithfulness_when_answer_is_hallucinated(self):
        from evaluation.faithfulness import FaithfulnessChecker
        checker = FaithfulnessChecker(threshold=0.1)
        context_text = "The Battle of Panipat was fought in 1526."
        answer = (
            "The Mughal Empire traded extensively with China and had a complex "
            "monetary policy based on gold reserves. The GDP was 30% of global output."
        )
        chunk = self._make_chunk(context_text)
        result = checker.check(answer, [chunk])
        assert result.score < 0.4, f"Expected low faithfulness, got {result.score}"

    def test_empty_context_returns_zero_score(self):
        from evaluation.faithfulness import FaithfulnessChecker
        checker = FaithfulnessChecker()
        result = checker.check("Some answer.", [])
        assert result.score == 0.0

    def test_tokenizer_removes_punctuation(self):
        from evaluation.faithfulness import FaithfulnessChecker
        tokens = FaithfulnessChecker._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "," not in tokens
        assert "!" not in tokens


# ══════════════════════════════════════════════════════════════
#  Test: Performance Predictor
# ══════════════════════════════════════════════════════════════

class TestPerformancePredictor:
    SAMPLE_STUDENT = {
        "avg_score": 55.0,
        "latest_score": 60.0,
        "score_trend": 5.0,
        "attempts": 8,
        "correct_rate": 0.58,
        "wrong_rate": 0.17,
        "skip_rate": 0.25,
        "avg_time_per_q": 70.0,
        "consistency": 0.75,
        "days_active": 20,
    }

    def test_mock_data_generation(self):
        from analytics.performance_predictor import generate_mock_data
        df = generate_mock_data(n_students=100)
        assert len(df) == 100
        assert "will_pass" in df.columns
        assert df["avg_score"].between(0, 100).all()
        assert df["correct_rate"].between(0, 1).all()

    def test_training_produces_valid_report(self):
        from analytics.performance_predictor import PerformancePredictor
        predictor = PerformancePredictor()
        report = predictor.train(model_type="logistic_regression")
        assert "accuracy" in report
        assert "roc_auc" in report
        assert 0.5 < report["accuracy"] <= 1.0
        assert 0.5 < report["roc_auc"] <= 1.0

    def test_prediction_returns_valid_structure(self):
        from analytics.performance_predictor import PerformancePredictor
        predictor = PerformancePredictor()
        predictor.train(model_type="logistic_regression")
        result = predictor.predict(self.SAMPLE_STUDENT)
        assert result["prediction"] in ["PASS", "AT_RISK"]
        assert 0.0 <= result["pass_probability"] <= 1.0
        assert isinstance(result["top_improvement_areas"], list)

    def test_high_scoring_student_predicts_pass(self):
        from analytics.performance_predictor import PerformancePredictor
        predictor = PerformancePredictor()
        predictor.train()
        strong_student = {
            "avg_score": 85.0, "latest_score": 88.0, "score_trend": 3.0,
            "attempts": 15, "correct_rate": 0.85, "wrong_rate": 0.08,
            "skip_rate": 0.07, "avg_time_per_q": 55.0, "consistency": 0.92,
            "days_active": 50,
        }
        result = predictor.predict(strong_student)
        assert result["pass_probability"] > 0.6, "Strong student should have high pass probability"

    def test_weak_student_predicts_at_risk(self):
        from analytics.performance_predictor import PerformancePredictor
        predictor = PerformancePredictor()
        predictor.train()
        weak_student = {
            "avg_score": 22.0, "latest_score": 25.0, "score_trend": 3.0,
            "attempts": 2, "correct_rate": 0.25, "wrong_rate": 0.35,
            "skip_rate": 0.40, "avg_time_per_q": 110.0, "consistency": 0.45,
            "days_active": 4,
        }
        result = predictor.predict(weak_student)
        assert result["pass_probability"] < 0.5, "Weak student should have low pass probability"


# ══════════════════════════════════════════════════════════════
#  Test: Jaccard Similarity
# ══════════════════════════════════════════════════════════════

class TestJaccardSimilarity:
    def test_identical_sets_return_one(self):
        from evaluation.faithfulness import FaithfulnessChecker
        a = {"the", "quick", "brown", "fox"}
        assert FaithfulnessChecker._jaccard_overlap(a, a) == 1.0

    def test_disjoint_sets_return_zero(self):
        from evaluation.faithfulness import FaithfulnessChecker
        a = {"apple", "banana"}
        b = {"car", "dog"}
        assert FaithfulnessChecker._jaccard_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        from evaluation.faithfulness import FaithfulnessChecker
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        result = FaithfulnessChecker._jaccard_overlap(a, b)
        assert abs(result - 0.5) < 0.01   # |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
