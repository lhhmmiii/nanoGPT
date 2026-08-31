"""
API routes:
  GET  /api/health        — health check + model info
  POST /api/chat          — full (non-streaming) generation
  GET  /api/chat/stream   — SSE streaming generation
"""
import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api.config import MODEL_TYPE, NUM_KV_BLOCKS
from api.inference import InferenceEngine
from api.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    engine = InferenceEngine.get()
    return HealthResponse(
        status="ok",
        model_type=MODEL_TYPE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_free_blocks=engine.kv_cache_manager.num_free_blocks,
        device=str(engine.device),
    )


# --------------------------------------------------------------------------- #
# Full generation (non-streaming)                                              #
# --------------------------------------------------------------------------- #
@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    engine = InferenceEngine.get()
    generated_text, prompt_tokens, generated_tokens = engine.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return ChatResponse(
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )


# --------------------------------------------------------------------------- #
# Streaming generation (SSE)                                                   #
# --------------------------------------------------------------------------- #
@router.get("/chat/stream", tags=["chat"])
async def chat_stream(
    prompt: str = Query(..., min_length=1, description="Input prompt"),
    max_new_tokens: int = Query(200, ge=1, le=1000),
    temperature: float = Query(1.0, gt=0.0, le=2.0),
    top_k: Optional[int] = Query(50, ge=1),
):
    """
    Server-Sent Events (SSE) streaming endpoint.

    Each event is a JSON object:
      data: {"token": "<decoded text>"}        — intermediate token
      data: {"done": true, "stats": {...}}     — final event
    """
    engine = InferenceEngine.get()

    async def event_generator():
        generated_tokens = 0
        try:
            async for token_text in engine.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                generated_tokens += 1
                payload = json.dumps({"token": token_text})
                yield f"data: {payload}\n\n"

            # Final event
            stats = json.dumps({
                "done": True,
                "stats": {
                    "generated_tokens": generated_tokens,
                },
            })
            yield f"data: {stats}\n\n"
        except Exception as exc:
            error = json.dumps({"error": str(exc)})
            yield f"data: {error}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
