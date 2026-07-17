# 🧠 Production RAG PDF Q&A

> **Ask questions against any PDF. Get cited, grounded answers — powered entirely by local AI. No cloud. No database. No Docker.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat-square)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet?style=flat-square)](https://github.com/facebookresearch/faiss)

---

## ✦ What Is This?

A **fully offline, production-grade Retrieval-Augmented Generation (RAG)** system that lets you upload any PDF and interrogate it with natural language — with every answer pinned to exact page numbers.

No OpenAI. No paid APIs. No Docker containers. No external databases.  
Everything — LLM inference, embeddings, vector search, caching — runs **on your own machine**.

---

## ⚙️ How It Works — The Full Pipeline

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 — EXTRACT      PyMuPDF parses text per page            │
│  Phase 2 — CHUNK        LangChain splits text (400 chars/chunk) │
│  Phase 3 — EMBED        Ollama nomic-embed-text encodes chunks  │
│  Phase 4 — INDEX        FAISS IndexFlatIP stores vectors        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
User Question
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1 — CACHE CHECK   Exact-match (instant return)            │
│  Step 2 — EMBED QUERY   nomic-embed-text encodes question       │
│  Step 3 — FAISS SEARCH  Top-10 most relevant chunks retrieved   │
│  Step 4 — RERANK        CrossEncoder ms-marco-MiniLM re-scores  │
│  Step 5 — CONFIDENCE    Low-score gate filters bad results      │
│  Step 6 — PROMPT BUILD  Context + citations assembled           │
│  Step 7 — LLM GENERATE  Qwen2.5 answers via Ollama (streaming) │
│  Step 8 — CACHE + LOG   Answer saved to qa_cache + history      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Grounded Answer with Page Citations
```

---

## 🏗️ Project Structure

```
work_ll/
│
├── run.py                        # ← Entry point (run this!)
├── requirements.txt              # Python dependencies
├── setup_models.bat              # Windows: pull Ollama models
├── setup_models.sh               # Linux/macOS: pull Ollama models
│
├── app/
│   ├── main.py                   # FastAPI app factory + startup lifecycle
│   ├── config.py                 # Settings (env-configurable via .env)
│   ├── storage.py                # File-system storage: metadata, history, cache
│   │
│   ├── routers/
│   │   ├── api.py                # REST API: /api/v1/documents, /query, /stream
│   │   └── views.py              # HTML page routes (Jinja2 templates)
│   │
│   ├── services/
│   │   ├── indexing_service.py   # PDF extraction → chunking → embedding → FAISS
│   │   ├── query_service.py      # RAG query pipeline + streaming
│   │   └── pdf_service.py        # Upload file saving helper
│   │
│   ├── templates/
│   │   ├── base.html             # Base layout + Tailwind CDN
│   │   ├── index.html            # Dashboard (list documents)
│   │   ├── upload.html           # Drag-and-drop PDF uploader
│   │   ├── chat.html             # Chat interface with streaming
│   │   └── _navbar.html          # Navigation partial
│   │
│   └── static/
│       ├── js/chat.js            # SSE streaming chat client
│       └── js/upload.js          # Upload + polling logic
│
└── data/                         # Auto-created. One folder per document UUID
    └── <doc-uuid>/
        ├── metadata.json         # Status, page count, chunk count
        ├── original.pdf          # Uploaded PDF
        ├── chunks.json           # All text chunks (with page numbers)
        ├── index.faiss           # FAISS vector index
        ├── history.json          # Full Q&A conversation log
        └── qa_cache.json         # Exact-match answer cache
```

---

## 🔩 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web Framework** | FastAPI + Uvicorn | Async HTTP server, SSE streaming |
| **LLM** | Qwen2.5 via Ollama | Answer generation (100% local) |
| **Embeddings** | nomic-embed-text via Ollama | Semantic chunk encoding |
| **Vector Search** | FAISS `IndexFlatIP` | Cosine-similarity nearest-neighbour |
| **Reranker** | CrossEncoder `ms-marco-MiniLM-L-6-v2` | Precision ranking of retrieved chunks |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction per page |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | 400-char chunks, 50-char overlap |
| **Templating** | Jinja2 + Tailwind CSS | Server-side rendered UI |
| **Storage** | File system (JSON + FAISS) | Zero database dependency |

---

## 📋 Prerequisites

Before running anything, ensure the following are installed on your machine:

| Requirement | Minimum Version | Download |
|---|---|---|
| **Python** | 3.10+ | https://python.org |
| **Ollama** | Latest | https://ollama.com/download |
| **Git** | Any | https://git-scm.com |

> **Important:** Ollama must be **running as a background service** before you start the server. On Windows, launching the Ollama installer sets this up automatically. Verify with: `ollama list`

---

## 🚀 Execution Steps

Follow these steps **in order**. Do not skip.

---

### Step 1 — Clone or Open the Project

If you already have the project folder open, skip this step.

```bash
# Clone the repository
git clone <your-repo-url>
cd work_ll
```

---

### Step 2 — Create & Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt after activation.

---

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn`, `pymupdf`, `faiss-cpu`, `sentence-transformers`, `langchain-text-splitters`, `httpx`, `jinja2`, `python-multipart`, `pydantic-settings`, and `numpy`.

> ⚠️ `sentence-transformers` will download the CrossEncoder reranker model (~90 MB) on **first run**, not during `pip install`.

---

### Step 4 — Pull Required Ollama Models

This is a **one-time setup**. The setup scripts handle everything:

**Windows:**
```bat
setup_models.bat
```

**Linux / macOS:**
```bash
chmod +x setup_models.sh
./setup_models.sh
```

These scripts pull two models:

| Model | Size | Role |
|---|---|---|
| `qwen2.5` | ~4.7 GB | LLM — generates answers |
| `nomic-embed-text` | ~274 MB | Embeddings — encodes text to vectors |

> If `qwen2.5` fails (e.g., insufficient disk), the script automatically falls back to `qwen2.5:0.5b` (~390 MB). In that case, add `LLM_MODEL=qwen2.5:0.5b` to your `.env` file.

You can verify models were pulled successfully:
```bash
ollama list
```

---

### Step 5 — (Optional) Configure via `.env`

Create a `.env` file in the project root to override any default setting:

```env
# .env — all fields are optional, defaults shown below

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5
EMBEDDING_MODEL=nomic-embed-text
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K_RETRIEVE=10
TOP_N_RERANK=3
CONFIDENCE_THRESHOLD=-100.0

MAX_FILE_SIZE_MB=200
DATA_DIR=./data
```

> If no `.env` file is present, the app runs perfectly with built-in defaults.

---

### Step 6 — Start the Server

```bash
python run.py
```

Expected output:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     ✅ Production RAG PDF Q&A started
INFO:     Ollama URL : http://localhost:11434
INFO:     LLM model  : qwen2.5
INFO:     Embed model: nomic-embed-text
INFO:     Data dir   : /path/to/work_ll/data
INFO:     Docs       : http://localhost:8000/docs
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> The server auto-reloads on any code change (hot-reload is enabled by default via `reload=True`).

---

### Step 7 — Use the Application

Open your browser and navigate to:

| URL | Description |
|---|---|
| `http://localhost:8000` | 🏠 Dashboard — view all uploaded documents |
| `http://localhost:8000/upload` | 📤 Upload a new PDF |
| `http://localhost:8000/chat/<doc-id>` | 💬 Chat with an indexed document |
| `http://localhost:8000/docs` | 📖 Interactive Swagger API docs |
| `http://localhost:8000/redoc` | 📖 ReDoc API reference |

**Workflow:**
1. Go to `/upload` → drag-and-drop your PDF
2. Wait for indexing to complete (`queued → extracting → embedding → indexing → ready`)
3. Click **"Go to Chat"** when status shows **ready**
4. Ask questions — every answer cites exact page numbers

---

## 🌐 REST API Reference

All endpoints live under `/api/v1`.

### Health

```http
GET /api/v1/healthz          → { "status": "ok" }
GET /api/v1/readyz           → { "status": "ready", "ollama": "reachable" }
```

### Documents

```http
POST   /api/v1/documents                    Upload a PDF → returns { id, status, filename }
GET    /api/v1/documents                    List all documents
GET    /api/v1/documents/{id}/status        Poll indexing status
DELETE /api/v1/documents/{id}              Delete document + all data
```

### Query

```http
POST /api/v1/documents/{id}/query
Content-Type: application/json
{ "question": "What are the candidate's top skills?" }

→ { "answer": "...", "citations": [...], "cached": false, "confidence": 3.14 }
```

### Streaming Query (Server-Sent Events)

```http
POST /api/v1/documents/{id}/query/stream
Content-Type: application/json
{ "question": "Summarise the work experience." }

→ data: {"type": "token", "text": "The"}
→ data: {"type": "token", "text": " candidate"}
→ ...
→ data: {"type": "meta", "citations": [...], "confidence": 2.8, "cached": false}
→ data: {"type": "done"}
```

### History

```http
GET /api/v1/documents/{id}/history    → Full Q&A log for the document
```

---

## 🗂️ Document Indexing States

| Status | Meaning |
|---|---|
| `queued` | Upload received, background task queued |
| `extracting` | PyMuPDF is reading pages |
| `embedding` | Ollama is encoding chunks (slowest phase) |
| `indexing` | FAISS index being built and written to disk |
| `ready` | ✅ Document is fully indexed and queryable |
| `failed` | ❌ An error occurred — check logs for detail |

---

## 🔧 Troubleshooting

**`Cannot connect to Ollama`**
- Make sure Ollama is running: open Ollama app on Windows, or run `ollama serve` on Linux/macOS.
- Confirm it's accessible: `curl http://localhost:11434/api/tags`

**`Model not found` / HTTP 404 from Ollama**
- Run `ollama list` — if `qwen2.5` or `nomic-embed-text` are missing, re-run `setup_models.bat`

**`PDF appears to have no extractable text`**
- This means the PDF is scanned (image-only). PyMuPDF cannot OCR images. Use a PDF with embedded text.

**Slow first query after startup**
- The CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) downloads ~90 MB and loads into memory on first use. Subsequent queries are fast.

**`venv\Scripts\Activate.ps1` blocked by PowerShell**
- Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and try again.

---

## ⚡ Performance Notes

- **FAISS in-memory cache**: The vector index is loaded from disk once and kept in RAM for the lifetime of the server process — subsequent queries skip disk I/O entirely.
- **Exact-match Q&A cache**: Repeated identical questions are served instantly from `qa_cache.json` without touching the LLM.
- **Shared HTTP client**: A single persistent `httpx.AsyncClient` is reused across all Ollama requests, avoiding TCP reconnection overhead.
- **Async throughout**: The entire request path — upload, embedding, LLM call, streaming — is non-blocking, so the server handles concurrent requests without worker threads.

---

## 📁 Data Storage Layout

All document data is stored under `data/<doc-uuid>/` — no database required:

```
data/
└── 3f2a1b4c-xxxx-xxxx-xxxx-xxxxxxxxxxxx/
    ├── metadata.json     # { id, filename, status, page_count, chunk_count, created_at }
    ├── original.pdf      # The uploaded PDF file
    ├── chunks.json       # [ { page: 1, text: "..." }, ... ]
    ├── index.faiss       # Binary FAISS vector index
    ├── history.json      # Chronological Q&A log
    └── qa_cache.json     # Exact-match answer cache
```

To **fully delete** a document and all its data, use the `DELETE /api/v1/documents/{id}` endpoint — or simply delete the folder.

---
<div align="center">
  <sub>Built with FastAPI · Ollama · FAISS · PyMuPDF · sentence-transformers</sub>
</div>
