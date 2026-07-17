import os
import warnings

# Prevent transformers from importing TensorFlow (optional dep we don't need)
# This stops the NumPy 1.x/2.x cv2 cascade error from polluting our logs
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", message=".*NumPy.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)

import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import api, views
from .config import settings

# ── Logging setup (structured, readable) ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="Production RAG PDF Q&A — no Docker, no database. File-system backed.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow same-origin for browser JS fetch calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ──────────────────────────────────────────────────────────────
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/static/css", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(views.router)    # HTML pages
app.include_router(api.router)      # JSON API

# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


async def _ensure_model_pulled(model: str) -> bool:
    """
    Check if an Ollama model is already downloaded; pull it if not.
    Returns True if the model is ready, False on failure.
    """
    import httpx as _httpx
    base = settings.OLLAMA_BASE_URL

    # Check if model already exists via /api/tags
    try:
        async with _httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base}/api/tags")
            r.raise_for_status()
            existing = [m["name"] for m in r.json().get("models", [])]
            # Match by name prefix (e.g. "qwen2.5" matches "qwen2.5:latest")
            if any(m == model or m.startswith(model + ":") or model.startswith(m.split(":")[0]) for m in existing):
                logger.info(f"   ✅ Model already present: {model}")
                return True
    except Exception as e:
        logger.warning(f"   Could not check Ollama model list: {e}")

    # Pull the model
    logger.info(f"   ⬇️  Pulling model '{model}' from Ollama (this may take a while)…")
    try:
        async with _httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST", f"{base}/api/pull",
                json={"name": model, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json as _json
                        data = _json.loads(line)
                        status = data.get("status", "")
                        if "pulling" in status or "verifying" in status or "writing" in status:
                            total    = data.get("total", 0)
                            completed = data.get("completed", 0)
                            if total:
                                pct = int(completed / total * 100)
                                logger.info(f"   {status}: {pct}%")
                        elif status:
                            logger.info(f"   {status}")
                    except Exception:
                        pass
        logger.info(f"   ✅ Model pulled successfully: {model}")
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to pull model '{model}': {e}")
        return False


@app.on_event("startup")
async def startup():
    logger.info(f"✅ {settings.APP_NAME} started")
    logger.info(f"   Ollama URL : {settings.OLLAMA_BASE_URL}")
    logger.info(f"   LLM model  : {settings.LLM_MODEL}")
    logger.info(f"   Embed model: {settings.EMBEDDING_MODEL}")
    logger.info(f"   Data dir   : {settings.DATA_DIR}")
    logger.info(f"   Docs       : http://localhost:8000/docs")

    # ── Auto-pull required Ollama models if not already downloaded ────────────
    logger.info("   Checking required Ollama models…")
    await _ensure_model_pulled(settings.EMBEDDING_MODEL)
    await _ensure_model_pulled(settings.LLM_MODEL)
