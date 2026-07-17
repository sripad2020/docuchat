import os
import json
import asyncio
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..storage import update_document_status, get_doc_dir
from .query_service import get_http_client, invalidate_index_cache

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> List[float]:
    """Call local Ollama nomic-embed-text to embed a single text string."""
    client = get_http_client()
    response = await client.post(
        f"{settings.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": settings.EMBEDDING_MODEL, "prompt": text},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


async def process_pdf(doc_id: str, file_path: str):
    """
    Background task: extracts text from PDF, chunks it, embeds via Ollama
    nomic-embed-text, builds a FAISS index, and saves everything to disk.
    No database — all state lives in data/<doc_id>/.
    """
    try:
        # ── Phase 1: Extract ─────────────────────────────────────────────────
        update_document_status(doc_id, "extracting")
        logger.info(f"[{doc_id}] Extracting text from PDF: {file_path}")

        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_text = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages_text.append({"page": page_num + 1, "text": text})

        logger.info(f"[{doc_id}] Extracted text from {len(pages_text)}/{total_pages} pages")

        if not pages_text:
            raise ValueError("PDF appears to have no extractable text (scanned/image-only PDF).")

        # ── Phase 2: Chunk ───────────────────────────────────────────────────
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = []
        for p in pages_text:
            splits = text_splitter.split_text(p["text"])
            for split in splits:
                chunks.append({"page": p["page"], "text": split})

        logger.info(f"[{doc_id}] Created {len(chunks)} chunks")

        # ── Phase 3: Embed via local Ollama ──────────────────────────────────
        update_document_status(doc_id, "embedding",
                               page_count=total_pages, chunk_count=len(chunks))

        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = await get_embedding(chunk["text"])
            embeddings.append(emb)
            if (i + 1) % 20 == 0:
                logger.info(f"[{doc_id}] Embedded {i + 1}/{len(chunks)} chunks")
            # Small yield to keep event loop responsive during large PDFs
            await asyncio.sleep(0)

        logger.info(f"[{doc_id}] Finished embedding all {len(chunks)} chunks")

        # ── Phase 4: Build & Persist FAISS Index ─────────────────────────────
        update_document_status(doc_id, "indexing")

        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)   # Inner Product (cosine-friendly with normalized vecs)

        vectors = np.array(embeddings, dtype="float32")
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        index.add(vectors)

        doc_dir = get_doc_dir(doc_id)
        faiss.write_index(index, os.path.join(doc_dir, "index.faiss"))

        with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        update_document_status(doc_id, "ready",
                               page_count=total_pages, chunk_count=len(chunks))
        # Invalidate the in-memory FAISS cache so the next query loads fresh data
        invalidate_index_cache(doc_id)
        logger.info(f"[{doc_id}] Indexing complete — ready for queries")

    except Exception as e:
        logger.exception(f"[{doc_id}] Indexing failed: {e}")
        update_document_status(doc_id, "failed", error=str(e))
