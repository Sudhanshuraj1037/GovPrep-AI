# GovPrep AI — Agentic-RAG System
### Production-Grade ML Pipeline for Indian Government Exam Preparation

> **Built for 2027 Placements** — Demonstrates full ML lifecycle: Data Ingestion → Embeddings → Vector DB → Agentic RAG → Evaluation → Serving

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE (5 Stages)                       │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│ Stage 1  │ Stage 2  │ Stage 3  │ Stage 4  │ Stage 5            │
│ Ingest   │ Embed    │ Store    │ RAG      │ Evaluate           │
│          │          │          │          │                    │
│ PyPDF2   │ MiniLM   │ ChromaDB │ Groq     │ RAGAS +            │
│ LangChain│ L6-v2    │ (local)  │ Llama-3  │ Faithfulness       │
│ Chunking │ 384-dim  │ HNSW     │ Grounded │ Checker            │
│          │ ~90MB    │ cosine   │ Prompts  │                    │
└──────────┴──────────┴──────────┴──────────┴────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────┐
│              BONUS: Analytics ML (scikit-learn)                  │
│  Features → LogisticRegression / RandomForest → Pass Prediction │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
govprep-rag/
│
├── config/
│   └── settings.py              # Centralised config (Pydantic Settings)
│
├── data_pipeline/               # Stage 1: Data Ingestion & Preprocessing
│   └── pdf_loader.py            # PDF → Pages → Chunks (RecursiveCharacterTextSplitter)
│
├── ml_core/                     # Stages 2-4: The ML Core
│   ├── embeddings.py            # all-MiniLM-L6-v2 wrapper (batch, memory-safe)
│   ├── vector_store.py          # ChromaDB manager (upsert + semantic search)
│   └── rag_pipeline.py          # Agentic RAG: Retrieve → Rerank → Ground → Generate
│
├── evaluation/                  # Stage 5: Explainable AI
│   ├── faithfulness.py          # Faithfulness checker + Citation tracker
│   └── ragas_eval.py            # RAGAS framework integration
│
├── analytics/                   # Bonus: Classical ML
│   └── performance_predictor.py # Random Forest pass/fail predictor (sklearn)
│
├── api/
│   └── main.py                  # FastAPI application (all endpoints)
│
├── tests/
│   └── test_rag_pipeline.py     # pytest unit tests for all components
│
├── scripts/
│   └── ingest_pdfs.py           # CLI: bulk PDF ingestion
│
├── data/
│   ├── raw/                     # Drop your PDFs here
│   ├── processed/               # Chunked JSON (auto-generated)
│   └── chroma_db/               # ChromaDB persistent storage (auto-generated)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd govprep-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Ingest Your PDFs

```bash
# Drop NCERT/government PDFs into data/raw/
# Then run ingestion:

python scripts/ingest_pdfs.py \
    --pdf data/raw/ncert_history_9.pdf \
    --subject History \
    --chapter "French Revolution"

# Or ingest an entire folder:
python scripts/ingest_pdfs.py --dir data/raw/ --subject "General Studies"
```

### 4. Start the API

```bash
python api/main.py
# Visit: http://localhost:8000/docs
```

### 5. Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who founded the Mughal Empire and in which year?",
    "subject_filter": "History",
    "exam_context": "Railway RRB NTPC"
  }'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload PDF → chunk → embed → store |
| `POST` | `/ask` | Agentic-RAG Q&A with faithfulness check |
| `POST` | `/predict` | Student pass/fail ML prediction |
| `POST` | `/evaluate` | Run RAGAS evaluation |
| `GET`  | `/health` | System health + vector store stats |
| `DELETE` | `/reset-vectorstore` | Clear all vectors (dev only) |

---

## ML Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Embeddings | `all-MiniLM-L6-v2` | ~90MB | Text → 384-dim vectors |
| Generation | `openai/gpt-oss-20b` (Groq) | API | Grounded answer generation |
| Prediction | `RandomForestClassifier` | ~2MB | Student pass/fail prediction |
| Vector DB | ChromaDB (HNSW) | Local | Semantic similarity search |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

## Key Interview Talking Points

### Why RAG instead of fine-tuning?
"Fine-tuning is expensive, requires GPUs, and the knowledge becomes stale. RAG gives us up-to-date, source-cited answers at inference time using only CPU resources."

### Why all-MiniLM-L6-v2?
"It's 90MB vs 1.3GB for large models, runs on CPU at 14,000 sentences/second, and gives 80% of large model quality. Perfect for 8GB RAM constraint."

### Why ChromaDB over Pinecone?
"During development on a laptop, ChromaDB runs locally with zero cost and zero latency. The VectorStore class abstracts the backend — switching to Pinecone in production is a one-line config change."

### What is the faithfulness checker?
"It's my hallucination detector. It checks what fraction of claims in the LLM's answer can be traced back to the retrieved context using Jaccard overlap. Score below 0.25 flags the answer for review."

### What does the ML model predict?
"Given a student's mock test history (10 features: avg_score, correct_rate, attempts, consistency, etc.), the Random Forest predicts pass probability and identifies the top 3 areas needing improvement."

---

## Memory Usage (8GB RAM)

| Component | RAM Usage |
|-----------|-----------|
| all-MiniLM-L6-v2 model | ~400MB |
| ChromaDB (10K chunks) | ~200MB |
| FastAPI + Python | ~150MB |
| Batch embedding (32 texts) | ~300MB peak |
| **Total** | **~1GB working** |

Leaves 7GB free for OS and other processes.

---

## Groq Free Tier Limits

- 100,000 tokens/day
- Each RAG call uses ~1,500 tokens (context + answer)
- That's ~66 queries/day on free tier
- Use `openai/gpt-oss-20b` (fastest, cheapest)

---

## Tech Stack

- **FastAPI** — REST API + auto OpenAPI docs
- **Groq + Llama-3** — LLM inference (free, fast)
- **ChromaDB** — Local vector database
- **Sentence Transformers** — CPU embeddings
- **LangChain Text Splitters** — Smart chunking
- **scikit-learn** — Performance prediction ML
- **RAGAS** — RAG evaluation framework
- **pytest** — Unit testing
- **Pydantic** — Data validation
- **loguru** — Structured logging
