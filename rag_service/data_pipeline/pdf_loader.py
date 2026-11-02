"""
data_pipeline/pdf_loader.py
Stage 1: Data Ingestion and Preprocessing.
Uses PyPDF2 + LangChain RecursiveCharacterTextSplitter.
"""
import re, json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from config.settings import settings


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    subject: str
    chapter: Optional[str]
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PDFExtractor:
    @staticmethod
    def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        reader = PdfReader(str(path))
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = PDFExtractor._clean_text(raw)
            if cleaned.strip():
                pages.append({"page_number": page_num, "text": cleaned})
        logger.info(f"Extracted {len(pages)} pages from {path.name}")
        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\x20-\x7E\n\u0900-\u097F]', '', text)
        return text.strip()


class DocumentChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_pages(self, pages: List[Dict], source_file: str,
                    subject: str, chapter: Optional[str] = None) -> List[TextChunk]:
        chunks = []
        idx = 0
        for page_data in pages:
            sub_chunks = self.splitter.split_text(page_data["text"])
            for text in sub_chunks:
                if len(text.strip()) < 50:
                    continue
                chunks.append(TextChunk(
                    chunk_id=f"{Path(source_file).stem}_p{page_data['page_number']}_c{idx}",
                    text=text.strip(),
                    source_file=source_file,
                    page_number=page_data["page_number"],
                    chunk_index=idx,
                    subject=subject,
                    chapter=chapter,
                    word_count=len(text.split()),
                ))
                idx += 1
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks


class DataIngestionPipeline:
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = DocumentChunker()

    def ingest(self, pdf_path: str, subject: str,
               chapter: Optional[str] = None, save_chunks: bool = True) -> List[TextChunk]:
        logger.info(f"Ingesting: {pdf_path}")
        pages = self.extractor.extract_pages(pdf_path)
        if not pages:
            raise ValueError(f"No readable text in {pdf_path}")
        chunks = self.chunker.chunk_pages(pages, Path(pdf_path).name, subject, chapter)
        if save_chunks:
            out = Path(settings.DATA_PROCESSED_DIR) / f"{Path(pdf_path).stem}_chunks.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
        return chunks

    def ingest_directory(self, directory: str, subject: str) -> List[TextChunk]:
        all_chunks = []
        for pdf in Path(directory).glob("*.pdf"):
            try:
                all_chunks.extend(self.ingest(str(pdf), subject=subject))
            except Exception as e:
                logger.error(f"Failed {pdf}: {e}")
        return all_chunks
