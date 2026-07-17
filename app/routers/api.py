import os
import json
import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..storage import (
    create_document, get_document, get_all_documents,
    delete_document, get_history,
)
from ..services.pdf_service import save_upload_file
from ..services.indexing_service import process_pdf
from ..services.query_service import query_document, stream_query_document

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


# ── Health Checks ─────────────────────────────────────────────────────────────

@router.get("/healthz", tags=["Health"])
async def health():
    """Liveness probe — returns 200 if process is alive."""
    return {"status": "ok"}


@router.get("/readyz", tags=["Health"])
async def ready():
    """Readiness probe — checks Ollama is reachable."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
        return {"status": "ready", "ollama": "reachable"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {e}")


# ── Documents ─────────────────────────────────────────────────────────────────

@router.post("/documents", tags=["Documents"])
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF. Kicks off background indexing immediately.
    Returns document_id + status='queued'.
    """
    try:
        # Validate by extension (more reliable than MIME type, which browsers report inconsistently)
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported. Please upload a .pdf file.")

        # Validate file size — wrap in try/except because SpooledTemporaryFile
        # on Windows may not support seek(0, 2) before the file is fully read.
        try:
            file.file.seek(0, 2)  # Seek to end
            size_bytes = file.file.tell()
            file.file.seek(0)     # Reset to start
            max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            if size_bytes > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {settings.MAX_FILE_SIZE_MB} MB."
                )
        except HTTPException:
            raise
        except Exception:
            # If seek fails (e.g. SpooledTemporaryFile in-memory), skip size check
            logger.warning(f"Could not determine file size for '{filename}', skipping size check.")

        doc_id = create_document(filename)
        file_path = await save_upload_file(file, doc_id)
        logger.info(f"Uploaded '{filename}' → doc_id={doc_id}, queuing indexing")

        background_tasks.add_task(process_pdf, doc_id, file_path)

        return {"id": doc_id, "status": "queued", "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents (newest first)."""
    return get_all_documents()


@router.get("/documents/{doc_id}/status", tags=["Documents"])
async def get_document_status(doc_id: str):
    """Poll indexing status: queued → extracting → embedding → indexing → ready | failed."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/documents/{doc_id}", tags=["Documents"])
async def remove_document(doc_id: str):
    """
    Delete a document — removes the PDF, FAISS index, chunks, cache, and all history.
    Nothing is stored in a database; deleting the folder is all it takes.
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_document(doc_id)
    logger.info(f"Deleted document {doc_id}")
    return {"status": "deleted", "id": doc_id}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@router.post("/documents/{doc_id}/query", tags=["Query"])
async def query_doc(doc_id: str, req: QueryRequest):
    """
    Ask a question against an indexed document.
    Returns: answer, citations (with page numbers), cached flag, confidence score.
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.get("status") != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready yet. Current status: '{doc.get('status')}'. "
                   "Please wait for indexing to complete."
        )

    try:
        result = await query_document(doc_id, req.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@router.post("/documents/{doc_id}/query/stream", tags=["Query"])
async def stream_query_doc(doc_id: str, req: QueryRequest):
    """
    Ask a question and receive tokens via Server-Sent Events (streaming).
    Each event is a JSON line: {"type": "token"|"meta"|"error", ...}
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.get("status") != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready yet. Current status: '{doc.get('status')}'",
        )

    async def event_generator():
        try:
            async for chunk in stream_query_document(doc_id, req.question):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.exception(f"Streaming error for {doc_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        finally:
            yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if behind a proxy
        },
    )


# ── History ───────────────────────────────────────────────────────────────────

@router.get("/documents/{doc_id}/history", tags=["History"])
async def get_doc_history(doc_id: str):
    """
    Return the full Q&A history for a document.
    History is stored in data/<doc_id>/history.json — no database required.
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"doc_id": doc_id, "filename": doc.get("filename"), "history": get_history(doc_id)}
