"""
scripts/ingest_pdfs.py

CLI script to bulk-ingest NCERT/Government PDFs into ChromaDB.
Run this once (or whenever you add new PDFs) to populate the vector store.

Usage:
    # Ingest a single PDF
    python scripts/ingest_pdfs.py --pdf data/raw/ncert_history_9.pdf --subject History

    # Ingest all PDFs in a folder
    python scripts/ingest_pdfs.py --dir data/raw/ --subject "General Studies"

    # Ingest with chapter metadata
    python scripts/ingest_pdfs.py --pdf data/raw/mughal.pdf \
        --subject History --chapter "Mughal Empire"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from data_pipeline.pdf_loader import DataIngestionPipeline
from ml_core.vector_store import VectorStore


def ingest(
    pdf_path: str | None,
    directory: str | None,
    subject: str,
    chapter: str | None,
) -> None:
    pipeline = DataIngestionPipeline()
    vector_store = VectorStore()

    all_chunks = []

    if pdf_path:
        logger.info(f"Ingesting single PDF: {pdf_path}")
        chunks = pipeline.ingest(
            pdf_path=pdf_path,
            subject=subject,
            chapter=chapter,
        )
        all_chunks.extend(chunks)

    elif directory:
        logger.info(f"Ingesting all PDFs in: {directory}")
        chunks = pipeline.ingest_directory(directory=directory, subject=subject)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks were created. Check the PDF files.")
        sys.exit(1)

    logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
    vector_store.add_chunks(all_chunks)

    stats = vector_store.get_stats()
    logger.info(f"Done! Vector store now has {stats['total_chunks']} total chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into GovPrep AI vector store")
    parser.add_argument("--pdf",     type=str, help="Path to a single PDF file")
    parser.add_argument("--dir",     type=str, help="Directory containing PDF files")
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject name (e.g. History, Mathematics, Science)")
    parser.add_argument("--chapter", type=str, default=None,
                        help="Chapter name (optional)")

    args = parser.parse_args()

    if not args.pdf and not args.dir:
        parser.error("Provide either --pdf or --dir")

    ingest(
        pdf_path=args.pdf,
        directory=args.dir,
        subject=args.subject,
        chapter=args.chapter,
    )
