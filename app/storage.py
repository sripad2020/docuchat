import os
import json
import uuid
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from .config import settings


def get_doc_dir(doc_id: str) -> str:
    return os.path.join(settings.DATA_DIR, doc_id)


def create_document(filename: str) -> str:
    doc_id = str(uuid.uuid4())
    doc_dir = get_doc_dir(doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    meta = {
        "id": doc_id,
        "filename": filename,
        "status": "queued",
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "page_count": 0,
        "chunk_count": 0,
    }
    _save_metadata(doc_id, meta)
    _save_history(doc_id, [])
    return doc_id


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    meta_path = os.path.join(get_doc_dir(doc_id), "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_document_status(doc_id: str, status: str, error: str = None,
                           page_count: int = None, chunk_count: int = None):
    meta = get_document(doc_id)
    if meta:
        meta["status"] = status
        meta["error"] = error
        if page_count is not None:
            meta["page_count"] = page_count
        if chunk_count is not None:
            meta["chunk_count"] = chunk_count
        _save_metadata(doc_id, meta)


def delete_document(doc_id: str):
    doc_dir = get_doc_dir(doc_id)
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir)


def get_all_documents() -> List[Dict[str, Any]]:
    docs = []
    if not os.path.exists(settings.DATA_DIR):
        return docs
    for d in os.listdir(settings.DATA_DIR):
        doc = get_document(d)
        if doc:
            docs.append(doc)
    # Sort by created_at descending (newest first)
    docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return docs


def _save_metadata(doc_id: str, meta: Dict[str, Any]):
    meta_path = os.path.join(get_doc_dir(doc_id), "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ── History (Q&A log per document) ───────────────────────────────────────────

def add_to_history(doc_id: str, role: str, content: str,
                   citations: List[Dict] = None, cached: bool = False,
                   confidence: float = None):
    history = get_history(doc_id)
    msg: Dict[str, Any] = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if citations is not None:
        msg["citations"] = citations
    if cached is not None:
        msg["cached"] = cached
    if confidence is not None:
        msg["confidence"] = confidence
    history.append(msg)
    _save_history(doc_id, history)


def get_history(doc_id: str) -> List[Dict[str, Any]]:
    hist_path = os.path.join(get_doc_dir(doc_id), "history.json")
    if not os.path.exists(hist_path):
        return []
    with open(hist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_history(doc_id: str, history: List[Dict[str, Any]]):
    hist_path = os.path.join(get_doc_dir(doc_id), "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ── Q&A Cache (exact-match first, for instant repeats) ───────────────────────

def get_cache(doc_id: str) -> List[Dict[str, Any]]:
    cache_path = os.path.join(get_doc_dir(doc_id), "qa_cache.json")
    if not os.path.exists(cache_path):
        return []
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_to_cache(doc_id: str, question: str, answer: str, citations: List[Dict]):
    cache = get_cache(doc_id)
    cache.append({
        "question": question,
        "answer": answer,
        "citations": citations,
        "cached_at": datetime.utcnow().isoformat(),
    })
    cache_path = os.path.join(get_doc_dir(doc_id), "qa_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
