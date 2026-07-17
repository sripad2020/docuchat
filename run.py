"""
run.py  ── PyCharm / python run.py entry point
───────────────────────────────────────────────
Run this file directly from PyCharm (or the terminal):
    python run.py

Do NOT run app/main.py directly — Python relative imports require
the package to be executed as a module (python -m ...) or via uvicorn.
"""

import os
import sys

# Ensure the project root (the folder containing this file) is on sys.path
# so that `import app` works correctly as an absolute import.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Suppress noisy TF / NumPy warnings before any heavy imports
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",     # module path — works because ROOT is on sys.path
        host="0.0.0.0",
        port=8000,
        reload=True,        # auto-reload on code changes (great for development)
        reload_dirs=[ROOT], # watch the whole project folder
    )
