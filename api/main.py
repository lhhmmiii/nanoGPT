"""
FastAPI application entrypoint.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import CORS_ORIGINS
from api.inference import InferenceEngine
from api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    print("[startup] Loading model …")
    InferenceEngine.get()   # eagerly initialise the singleton
    print("[startup] Model ready.")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    print("[shutdown] Releasing model.")
    InferenceEngine.shutdown()


app = FastAPI(
    title="GPT-2 Paged Attention API",
    description=(
        "Inference API for GPT-2 with Paged Attention KV cache.\n\n"
        "Supports both blocking (`POST /api/chat`) and "
        "streaming (`GET /api/chat/stream`) generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "GPT-2 Paged Attention API — see /docs for Swagger UI."}
