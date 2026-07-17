import os
import json
import asyncio
import logging
import faiss
import numpy as np
import httpx
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder

from ..config import settings
from ..storage import get_doc_dir, get_cache, add_to_cache, add_to_history

logger = logging.getLogger(__name__)

# ── Shared persistent HTTP client (avoids TCP handshake on every request) ─────
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=180.0)
    return _http_client


# ── In-memory FAISS index cache (avoids disk reads after first load) ──────────
# Structure: { doc_id: (faiss.Index, list[dict]) }
_index_cache: Dict[str, Tuple[faiss.Index, List[Dict]]] = {}


def get_index(doc_id: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load FAISS index + chunks from disk on first call, return cached copy thereafter."""
    if doc_id not in _index_cache:
        doc_dir = get_doc_dir(doc_id)
        index_path = os.path.join(doc_dir, "index.faiss")
        chunks_path = os.path.join(doc_dir, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise ValueError("Document index not found. Has the document finished indexing?")

        logger.info(f"[{doc_id}] Loading FAISS index into memory cache")
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        _index_cache[doc_id] = (index, chunks)
        logger.info(f"[{doc_id}] FAISS index cached ({len(chunks)} chunks)")

    return _index_cache[doc_id]


def invalidate_index_cache(doc_id: str):
    """Call this after re-indexing a document to force a fresh load."""
    _index_cache.pop(doc_id, None)


# ── CrossEncoder is loaded once at startup (not lazily) so the first query is fast.
_cross_encoder: CrossEncoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading CrossEncoder: {settings.RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(settings.RERANKER_MODEL)
        logger.info("CrossEncoder loaded and ready.")
    return _cross_encoder


# Pre-load at import time so startup absorbs the delay, not the first query.
try:
    get_cross_encoder()
except Exception as _e:
    logger.warning(f"CrossEncoder pre-load failed (will retry on first query): {_e}")


async def get_embedding(text: str) -> List[float]:
    """Import here to avoid circular imports — re-uses indexing_service's embedding call."""
    from .indexing_service import get_embedding as _get_emb
    return await _get_emb(text)


async def query_document(doc_id: str, question: str) -> Dict[str, Any]:
    """
    Full RAG query pipeline:
      1. Exact-match cache lookup (instant return)
      2. Embed query → FAISS retrieval (top-K)
      3. CrossEncoder rerank (top-N)
      4. Confidence gate (skip LLM if score too low)
      5. Build prompt with page citations
      6. Call local Qwen via Ollama /api/generate
      7. Cache result + write to history
    """

    # ── 1. Exact Cache Hit ────────────────────────────────────────────────────
    cache = get_cache(doc_id)
    for entry in cache:
        if entry["question"].lower().strip() == question.lower().strip():
            logger.info(f"[{doc_id}] Cache HIT for question: {question[:60]}")
            add_to_history(doc_id, "user", question)
            add_to_history(doc_id, "assistant", entry["answer"],
                           citations=entry["citations"], cached=True, confidence=1.0)
            return {
                "answer": entry["answer"],
                "citations": entry["citations"],
                "cached": True,
                "confidence": 1.0,
            }

    # ── 2. Load FAISS Index & Chunks (from in-memory cache) ──────────────────
    index, all_chunks = get_index(doc_id)

    if not all_chunks:
        return _not_found_response()

    # ── 3. Embed Query ────────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    query_emb = await get_embedding(question)
    query_vec = np.array([query_emb], dtype="float32")
    faiss.normalize_L2(query_vec)  # Must normalise to match IndexFlatIP index

    # ── 4. FAISS Retrieve ─────────────────────────────────────────────────────
    # Clamp k to actual number of chunks to avoid FAISS errors on small docs
    k = min(settings.TOP_K_RETRIEVE, len(all_chunks))
    distances, indices = index.search(query_vec, k)

    retrieved_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(all_chunks):
            retrieved_chunks.append(all_chunks[idx])

    if not retrieved_chunks:
        return _not_found_response()

    # ── 5. CrossEncoder Rerank ────────────────────────────────────────────────
    cross_encoder = get_cross_encoder()
    pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
    # Run blocking CrossEncoder in thread pool so we don't block the event loop
    scores = await loop.run_in_executor(None, cross_encoder.predict, pairs)

    ranked_indices = np.argsort(scores)[::-1]
    top_n_indices = ranked_indices[:settings.TOP_N_RERANK]

    top_chunks = [retrieved_chunks[i] for i in top_n_indices]
    best_score = float(scores[top_n_indices[0]])

    logger.info(f"[{doc_id}] Best rerank score: {best_score:.3f} (threshold: {settings.CONFIDENCE_THRESHOLD})")

    # ── 6. Confidence Gate ────────────────────────────────────────────────────
    if best_score < settings.CONFIDENCE_THRESHOLD:
        logger.info(f"[{doc_id}] Confidence too low ({best_score:.3f}) — skipping LLM")
        result = _not_found_response(confidence=best_score)
        add_to_history(doc_id, "user", question)
        add_to_history(doc_id, "assistant", result["answer"], confidence=best_score)
        return result

    # ── 7. Build Prompt with Page Citations ───────────────────────────────────
    context_parts = []
    citations_meta = []
    for i, chunk in enumerate(top_chunks):
        context_parts.append(
            f"[Source {i+1} — Page {chunk['page']}]\n{chunk['text']}"
        )
        citations_meta.append({
            "chunk": i + 1,
            "page": chunk["page"],
            "snippet": chunk["text"][:150] + ("..." if len(chunk["text"]) > 150 else ""),
        })

    context_block = "\n\n".join(context_parts)

    # Resume / document-friendly prompt that doesn't refuse on broad questions
    prompt = f"""<|im_start|>system
You are a helpful document assistant. Answer the user's question using ONLY the provided document context.
Rules:
1. Always try to give a helpful, accurate answer from the context provided.
2. Cite the page number for every fact using [Page X] inline wherever possible.
3. For resume documents: summarise skills, experience, education, and achievements clearly.
4. Only if the specific information is truly absent from ALL context sections, say: "I cannot answer this based on the provided document."
5. Do not hallucinate, guess, or use outside knowledge.
<|im_end|>
<|im_start|>user
Document Context:
{context_block}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

    # ── 8. Call Local Qwen via Ollama ─────────────────────────────────────────
    logger.info(f"[{doc_id}] Calling Ollama model: {settings.LLM_MODEL}")
    try:
        client = get_http_client()
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,      # Low temp for factual answers
                    "top_p": 0.9,
                    "num_predict": 512,      # Reduced from 1024 for faster responses
                },
            },
        )
        response.raise_for_status()
        llm_response: str = response.json()["response"].strip()
    except httpx.HTTPStatusError as e:
        logger.error(f"[{doc_id}] Ollama HTTP error: {e}")
        raise ValueError(f"Ollama returned an error: {e.response.status_code}. Is {settings.LLM_MODEL} pulled?")
    except httpx.ConnectError:
        raise ValueError("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434.")

    # If model says it can't answer, return not-found
    if "cannot answer this based on the provided document" in llm_response.lower():
        result = _not_found_response(confidence=best_score)
        add_to_history(doc_id, "user", question)
        add_to_history(doc_id, "assistant", result["answer"], confidence=best_score)
        return result

    # ── 9. Cache & Persist History ────────────────────────────────────────────
    add_to_cache(doc_id, question, llm_response, citations_meta)
    add_to_history(doc_id, "user", question)
    add_to_history(doc_id, "assistant", llm_response,
                   citations=citations_meta, cached=False, confidence=best_score)

    return {
        "answer": llm_response,
        "citations": citations_meta,
        "cached": False,
        "confidence": best_score,
    }


async def stream_query_document(doc_id: str, question: str):
    """
    Streaming RAG pipeline — same as query_document but yields tokens
    incrementally as they arrive from Ollama (stream=True).

    Yields dicts:
      {"type": "token",    "text": "..."}          — one token at a time
      {"type": "meta",     "citations": [...], "confidence": float, "cached": bool}
      {"type": "error",    "detail": "..."}
    """
    import json as _json

    # ── 1. Exact Cache Hit ────────────────────────────────────────────────────
    cache = get_cache(doc_id)
    for entry in cache:
        if entry["question"].lower().strip() == question.lower().strip():
            logger.info(f"[{doc_id}] Stream: Cache HIT")
            add_to_history(doc_id, "user", question)
            add_to_history(doc_id, "assistant", entry["answer"],
                           citations=entry["citations"], cached=True, confidence=1.0)
            # For cache hits, emit the full answer as a single token chunk
            yield {"type": "token", "text": entry["answer"]}
            yield {"type": "meta", "citations": entry["citations"],
                   "confidence": 1.0, "cached": True}
            return

    # ── 2. Load FAISS index (cached in memory) ────────────────────────────────
    try:
        index, all_chunks = get_index(doc_id)
    except ValueError as e:
        yield {"type": "error", "detail": str(e)}
        return

    if not all_chunks:
        yield {"type": "error", "detail": "No chunks found in document index."}
        return

    # ── 3. Embed Query ────────────────────────────────────────────────────────
    try:
        query_emb = await get_embedding(question)
    except Exception as e:
        yield {"type": "error", "detail": f"Embedding failed: {e}"}
        return

    query_vec = np.array([query_emb], dtype="float32")
    faiss.normalize_L2(query_vec)

    # ── 4. FAISS Retrieve ─────────────────────────────────────────────────────
    k = min(settings.TOP_K_RETRIEVE, len(all_chunks))
    distances, indices = index.search(query_vec, k)
    retrieved_chunks = [all_chunks[idx] for idx in indices[0] if 0 <= idx < len(all_chunks)]

    if not retrieved_chunks:
        yield {"type": "error", "detail": "No relevant chunks found."}
        return

    # ── 5. CrossEncoder Rerank ────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    cross_encoder = get_cross_encoder()
    pairs = [[question, chunk["text"]] for chunk in retrieved_chunks]
    scores = await loop.run_in_executor(None, cross_encoder.predict, pairs)

    ranked_indices = np.argsort(scores)[::-1]
    top_n_indices = ranked_indices[:settings.TOP_N_RERANK]
    top_chunks = [retrieved_chunks[i] for i in top_n_indices]
    best_score = float(scores[top_n_indices[0]])

    logger.info(f"[{doc_id}] Stream: Best rerank score: {best_score:.3f}")

    # ── 6. Confidence Gate ────────────────────────────────────────────────────
    if best_score < settings.CONFIDENCE_THRESHOLD:
        msg = _not_found_response(confidence=best_score)
        add_to_history(doc_id, "user", question)
        add_to_history(doc_id, "assistant", msg["answer"], confidence=best_score)
        yield {"type": "token", "text": msg["answer"]}
        yield {"type": "meta", "citations": [], "confidence": best_score,
               "cached": False, "not_found": True}
        return

    # ── 7. Build Prompt ───────────────────────────────────────────────────────
    context_parts = []
    citations_meta = []
    for i, chunk in enumerate(top_chunks):
        context_parts.append(f"[Source {i+1} — Page {chunk['page']}]\n{chunk['text']}")
        citations_meta.append({
            "chunk": i + 1,
            "page": chunk["page"],
            "snippet": chunk["text"][:150] + ("..." if len(chunk["text"]) > 150 else ""),
        })

    context_block = "\n\n".join(context_parts)
    prompt = f"""<|im_start|>system
You are a helpful document assistant. Answer the user's question using ONLY the provided document context.
Rules:
1. Always try to give a helpful, accurate answer from the context provided.
2. Cite the page number for every fact using [Page X] inline wherever possible.
3. For resume documents: summarise skills, experience, education, and achievements clearly.
4. Only if the specific information is truly absent from ALL context sections, say: "I cannot answer this based on the provided document."
5. Do not hallucinate, guess, or use outside knowledge.
<|im_end|>
<|im_start|>user
Document Context:
{context_block}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

    # ── 8. Stream tokens from Ollama ──────────────────────────────────────────
    logger.info(f"[{doc_id}] Stream: calling Ollama model: {settings.LLM_MODEL}")
    full_response = []
    try:
        client = get_http_client()
        async with client.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512,
                },
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk_data = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                token = chunk_data.get("response", "")
                if token:
                    full_response.append(token)
                    yield {"type": "token", "text": token}
                if chunk_data.get("done"):
                    break

    except Exception as e:
        logger.error(f"[{doc_id}] Stream: Ollama error: {e}")
        yield {"type": "error", "detail": f"LLM error: {e}"}
        return

    # ── 9. Cache & Persist ────────────────────────────────────────────────────
    final_answer = "".join(full_response).strip()
    add_to_cache(doc_id, question, final_answer, citations_meta)
    add_to_history(doc_id, "user", question)
    add_to_history(doc_id, "assistant", final_answer,
                   citations=citations_meta, cached=False, confidence=best_score)

    yield {"type": "meta", "citations": citations_meta,
           "confidence": best_score, "cached": False}


def _not_found_response(confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "answer": "The answer to your question was not found in this document. "
                  "The retrieved content either doesn't contain the information, "
                  "or the confidence score was too low to trust.",
        "citations": [],
        "cached": False,
        "confidence": confidence,
        "not_found": True,
    }
