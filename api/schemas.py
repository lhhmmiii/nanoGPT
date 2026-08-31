"""
Pydantic schemas for the inference API.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(200, ge=1, le=1000, description="Max tokens to generate")
    temperature: float = Field(1.0, gt=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, description="Top-k sampling (None = disabled)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Once upon a time",
                "max_new_tokens": 100,
                "temperature": 0.8,
                "top_k": 50,
            }
        }
    }


class ChatResponse(BaseModel):
    generated_text: str = Field(..., description="Full generated text (prompt + new tokens)")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    generated_tokens: int = Field(..., description="Number of tokens generated")


class HealthResponse(BaseModel):
    status: str
    model_type: str
    num_kv_blocks: int
    num_free_blocks: int
    device: str
