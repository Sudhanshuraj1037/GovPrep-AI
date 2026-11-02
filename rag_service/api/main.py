"""
api/main.py

FastAPI Application — Production REST API for GovPrep-AI RAG System.

Endpoints:
  POST /ingest          — Upload and process a PDF into the vector store
  POST /ask             — Agentic-RAG Q&A with faithfulness check
  POST /predict         — Student performance prediction (ML model)
  POST /evaluate        — Run RAGAS evaluation on test questions
  GET  /health          — System health + vector store stats
  GET  /stats           — Detailed system metrics

Interview talking point:
  "I used FastAPI because it auto-generates OpenAPI docs, has native
   async support, uses Pydantic for request validation, and is the
   industry standard for ML serving — used at Uber, Microsoft, etc."
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings
from data_pipeline.pdf_loader import DataIngestionPipeline
from ml_core.vector_store import VectorStore
from ml_core.rag_pipeline import RAGPipeline
from evaluation.faithfulness import FaithfulnessChecker, CitationTracker
from analytics.performance_predictor import PerformancePredictor, FEATURE_COLUMNS


# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="GovPrep AI — Agentic-RAG API",
    description=(
        "Production-grade RAG system for Indian government exam preparation. "
        "Built with ChromaDB + Sentence Transformers + Groq Llama-3."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (lazy-loaded) ────────────────────────────────────
_vector_store: Optional[VectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None
_predictor: Optional[PerformancePredictor] = None
_faithfulness: Optional[FaithfulnessChecker] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_predictor() -> PerformancePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PerformancePredictor()
    return _predictor


def get_faithfulness() -> FaithfulnessChecker:
    global _faithfulness
    if _faithfulness is None:
        _faithfulness = FaithfulnessChecker()
    return _faithfulness


# ══════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="The student's question")
    subject_filter: Optional[str] = Field(None,
                          description="Filter retrieval to one subject (e.g. 'History')")
    exam_context: Optional[str] = Field("Railway RRB NTPC",
                          description="Exam template context for the answer")
    faithfulness_check: bool = Field(True,
                          description="Run faithfulness check on the answer")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Who founded the Mughal Empire and in which year?",
                "subject_filter": "History",
                "exam_context": "Railway RRB NTPC",
                "faithfulness_check": True,
            }
        }


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    faithfulness_score: float
    faithfulness_verdict: str
    annotated_answer: List[Dict[str, Any]]
    model_used: str
    latency_ms: float
    context_used: bool
    chunks_retrieved: int


class PredictRequest(BaseModel):
    avg_score: float = Field(..., ge=0, le=100, description="Average test score (0-100)")
    latest_score: float = Field(..., ge=0, le=100)
    score_trend: float = Field(0.0, description="Score improvement (latest - first)")
    attempts: int = Field(..., ge=1, description="Number of tests taken")
    correct_rate: float = Field(..., ge=0, le=1)
    wrong_rate: float = Field(..., ge=0, le=1)
    skip_rate: float = Field(..., ge=0, le=1)
    avg_time_per_q: float = Field(60.0, description="Average seconds per question")
    consistency: float = Field(0.7, ge=0, le=1)
    days_active: int = Field(1, ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "avg_score": 58.5,
                "latest_score": 63.0,
                "score_trend": 4.5,
                "attempts": 8,
                "correct_rate": 0.60,
                "wrong_rate": 0.15,
                "skip_rate": 0.25,
                "avg_time_per_q": 72.0,
                "consistency": 0.78,
                "days_active": 25,
            }
        }


class EvalRequest(BaseModel):
    questions: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of {question, ground_truth} dicts. Uses sample data if not provided."
    )


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """
    System health check — returns vector store stats and model status.
    Call this first to verify everything is loaded correctly.
    """
    vs_stats = get_vector_store().get_stats()
    return {
        "status": "ok",
        "groq_key_configured": bool(settings.GROQ_API_KEY),
        "groq_model": settings.GROQ_MODEL,
        "vector_store": vs_stats,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "top_k_retrieval": settings.TOP_K_RESULTS,
    }


@app.post("/ingest", tags=["Data Pipeline"])
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
    subject: str = Form("General", description="Subject name e.g. History, Mathematics"),
    chapter: Optional[str] = Form(None, description="Chapter name"),
):
    """
    **Stage 1 + 2 + 3 of the ML Pipeline**:
    Upload PDF → Extract & Chunk → Embed → Store in ChromaDB.

    - Accepts any text-based PDF
    - Runs chunking and embedding in background (non-blocking)
    - Returns chunk count immediately

    Interview talking point:
      "I made ingestion async using FastAPI BackgroundTasks so the
       HTTP response returns immediately while embedding happens in
       the background. In production I'd use Celery + Redis for this."
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run ingestion in background
    def _ingest():
        try:
            pipeline = DataIngestionPipeline()
            chunks = pipeline.ingest(
                pdf_path=tmp_path,
                subject=subject,
                chapter=chapter,
                save_chunks=True,
            )
            get_vector_store().add_chunks(chunks)
            logger.info(f"Background ingestion complete: {len(chunks)} chunks added")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    background_tasks.add_task(_ingest)

    return {
        "status": "ingestion_started",
        "filename": file.filename,
        "subject": subject,
        "chapter": chapter,
        "message": (
            "PDF is being processed in the background. "
            "Check /health for updated chunk count in ~10-30 seconds."
        ),
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG Pipeline"])
async def ask_question(request: AskRequest):
    """
    **The Core Agentic-RAG Endpoint**:

    Flow:
      1. Embed the question
      2. Retrieve top-k relevant chunks from ChromaDB
      3. Rerank using hybrid scoring
      4. Build grounded prompt (context-stuffed)
      5. Generate answer with Groq Llama-3
      6. Run faithfulness check
      7. Return answer + citations + faithfulness score

    This is the "Search-and-Synthesize" pattern.
    """
    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to your .env file."
        )

    try:
        # Run RAG pipeline
        rag = get_rag_pipeline()
        response = rag.answer(
            query=request.question,
            subject_filter=request.subject_filter,
            exam_context=request.exam_context,
        )

        # Run faithfulness check
        faith_score = 0.0
        faith_verdict = "No context available."
        annotated = []

        if request.faithfulness_check and response.retrieved_chunks:
            checker = get_faithfulness()
            faith_result = checker.check(response.answer, response.retrieved_chunks)
            faith_score = faith_result.score
            faith_verdict = faith_result.verdict

            tracker = CitationTracker()
            annotated = tracker.annotate(response.answer, response.retrieved_chunks)

        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            faithfulness_score=faith_score,
            faithfulness_verdict=faith_verdict,
            annotated_answer=annotated,
            model_used=response.model_used,
            latency_ms=response.latency_ms,
            context_used=response.context_used,
            chunks_retrieved=len(response.retrieved_chunks),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["Analytics ML"])
async def predict_performance(request: PredictRequest):
    """
    **Bonus ML Endpoint**: Student Performance Prediction.

    Uses a trained Random Forest / Logistic Regression model to predict
    whether a student will pass their target exam based on mock test history.

    Returns:
      - Pass/Fail prediction with probability
      - Top 3 improvement areas
      - Actionable study recommendations
    """
    predictor = get_predictor()
    student_data = request.model_dump()

    try:
        result = predictor.predict(student_data)
        return {
            "status": "ok",
            "input_features": student_data,
            **result,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Auto-train if model not found
        if "No trained model" in str(e) or not Path("analytics/models/performance_predictor.joblib").exists():
            logger.info("Training model with mock data...")
            predictor.train()
            result = predictor.predict(student_data)
            return {"status": "ok", "note": "Model was auto-trained with mock data.", **result}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-predictor", tags=["Analytics ML"])
async def train_predictor(background_tasks: BackgroundTasks):
    """
    Trigger (re)training of the performance predictor model.
    Runs in background. Check /health after ~30 seconds.
    """
    def _train():
        predictor = get_predictor()
        report = predictor.train(model_type="random_forest")
        logger.info(f"Training complete: accuracy={report['accuracy']:.4f}")

    background_tasks.add_task(_train)
    return {"status": "training_started", "message": "Model training in background..."}


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_rag(request: EvalRequest):
    """
    **RAGAS Evaluation Endpoint**: Measure RAG system quality.

    Computes:
      - Faithfulness score
      - Answer relevancy
      - Context precision
      - Context recall

    Use this to guide hyperparameter tuning (chunk size, top-k, threshold).
    """
    from evaluation.ragas_eval import RAGASEvaluator, EvalDataset

    questions = request.questions
    if not questions:
        questions = EvalDataset.SAMPLE_QUESTIONS

    rag = get_rag_pipeline()
    evaluator = RAGASEvaluator()

    try:
        report = evaluator.evaluate(questions, rag)
        return report
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-vectorstore", tags=["System"])
async def reset_vector_store():
    """
    ⚠️ DANGER: Delete all vectors from ChromaDB.
    Use only during development / testing.
    """
    get_vector_store().reset()
    return {"status": "ok", "message": "Vector store has been reset."}


# ── Startup event ──────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("GovPrep AI RAG API starting up...")
    logger.info(f"Groq model: {settings.GROQ_MODEL}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"ChromaDB path: {settings.CHROMA_PERSIST_DIR}")
    logger.info("API ready. Visit http://localhost:8000/docs for Swagger UI.")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # auto-reload on code changes during dev
        workers=1,         # single worker to stay within 8GB RAM
    )
