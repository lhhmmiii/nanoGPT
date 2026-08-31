"""
API configuration — values can be overridden via environment variables.
"""
import os

# Model
MODEL_TYPE: str = os.getenv("MODEL_TYPE", "gpt2")  # "gpt2" | "gpt2-medium" | ...
CHECKPOINT_PATH: str | None = os.getenv("CHECKPOINT_PATH", None)  # optional local .pt file

# KV Cache
NUM_KV_BLOCKS: int = int(os.getenv("NUM_KV_BLOCKS", "512"))

# Generation defaults
DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "200"))
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
DEFAULT_TOP_K: int | None = int(os.getenv("DEFAULT_TOP_K", "50"))

# Server
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# CORS — origins allowed to call the API
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
