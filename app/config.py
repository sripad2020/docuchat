import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Production RAG PDF Q&A"
    DATA_DIR: str = os.path.join(os.getcwd(), "data")

    # Ollama settings — uses locally running Ollama (no Docker)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # Correct Ollama model tag for Qwen. Change to "qwen2.5" or "qwen3" based on what you have pulled.
    LLM_MODEL: str = "qwen2.5"

    # Reranking (runs locally via sentence-transformers, no server needed)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # RAG pipeline settings (all configurable via .env)
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVE: int = 10
    TOP_N_RERANK: int = 3
    # CrossEncoder logit scores on short resume docs are often very negative.
    # We set the gate to -100 (effectively OFF) and rely on the LLM system prompt
    # guardrail ("I cannot answer...") to handle out-of-scope questions instead.
    CONFIDENCE_THRESHOLD: float = -100.0

    # File upload limits
    MAX_FILE_SIZE_MB: int = 200

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure data directory exists on startup
os.makedirs(settings.DATA_DIR, exist_ok=True)
