#!/usr/bin/env bash
# ============================================================
#  setup_models.sh  — Linux / macOS
#  Pulls all required Ollama models for this RAG app.
#  Run this ONCE before starting the server:
#    chmod +x setup_models.sh && ./setup_models.sh
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  RAG App — Ollama Model Setup (Linux / macOS)"
echo "============================================================"
echo ""

# Check if ollama is installed
if ! command -v ollama &>/dev/null; then
    echo "[ERROR] 'ollama' not found on PATH."
    echo "        Install it from: https://ollama.com/download"
    exit 1
fi

echo "[OK] Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
echo ""

# ── Pull LLM model ───────────────────────────────────────────
echo "[1/2] Pulling LLM model: qwen2.5"
echo "       (4.7 GB — this may take a few minutes)"
if ollama pull qwen2.5; then
    echo "[OK] qwen2.5 pulled."
else
    echo "[WARN] Failed to pull qwen2.5. Trying smaller fallback: qwen2.5:0.5b"
    ollama pull qwen2.5:0.5b
    echo "[OK] qwen2.5:0.5b pulled (faster, ~390 MB)."
    echo "     To use it, set LLM_MODEL=qwen2.5:0.5b in your .env file."
fi

echo ""

# ── Pull embedding model ─────────────────────────────────────
echo "[2/2] Pulling embedding model: nomic-embed-text"
echo "       (274 MB)"
ollama pull nomic-embed-text
echo "[OK] nomic-embed-text pulled."

echo ""
echo "============================================================"
echo "  All models ready!  Start the server with:"
echo "    python run.py"
echo "============================================================"
echo ""
