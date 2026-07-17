@echo off
REM ============================================================
REM  setup_models.bat  — Windows
REM  Pulls all required Ollama models for this RAG app.
REM  Run this ONCE before starting the server.
REM ============================================================

echo.
echo ============================================================
echo   RAG App — Ollama Model Setup (Windows)
echo ============================================================
echo.

REM Check if ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 'ollama' not found on PATH.
    echo         Download and install from: https://ollama.com/download
    pause
    exit /b 1
)

echo [OK] Ollama found.
echo.

REM ── Pull LLM model ──────────────────────────────────────────
echo [1/2] Pulling LLM model: qwen2.5
echo        (4.7 GB — this may take a few minutes)
ollama pull qwen2.5
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Failed to pull qwen2.5. Trying smaller fallback: qwen2.5:0.5b
    ollama pull qwen2.5:0.5b
)

echo.

REM ── Pull embedding model ─────────────────────────────────────
echo [2/2] Pulling embedding model: nomic-embed-text
echo        (274 MB)
ollama pull nomic-embed-text
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to pull nomic-embed-text.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   All models ready!  You can now start the server:
echo     python run.py
echo ============================================================
echo.
pause
