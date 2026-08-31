"""
InferenceEngine — singleton that holds the loaded model and KV cache manager.

Supports:
  - generate()         : full generation, returns complete string
  - generate_stream()  : async generator yielding decoded tokens one-by-one
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from typing import AsyncGenerator, Optional

import tiktoken
import torch

from models.gpt2_paged import GPT2, GPT2Config
from paged_attention.kv_cache_manager import KVCacheManager
from schemas import Request

from api.config import (
    CHECKPOINT_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MODEL_TYPE,
    NUM_KV_BLOCKS,
)

# GPT-2 uses the "gpt2" tiktoken encoding for all variants
_ENCODING_NAME = "gpt2"


class InferenceEngine:
    """Thread-safe inference engine wrapping GPT2 with Paged Attention."""

    _instance: Optional["InferenceEngine"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"[InferenceEngine] Loading model '{MODEL_TYPE}' on {self.device} …")

        if CHECKPOINT_PATH:
            self.model, _, _, _ = GPT2.load_checkpoint(
                CHECKPOINT_PATH, map_location=str(self.device)
            )
        else:
            self.model = GPT2.from_pretrained(MODEL_TYPE)

        self.model.eval()
        self.model.to(self.device)

        self.kv_cache_manager = KVCacheManager(num_blocks=NUM_KV_BLOCKS)
        self.enc = tiktoken.get_encoding(_ENCODING_NAME)
        self._gen_lock = threading.Lock()

        print(f"[InferenceEngine] Ready — {NUM_KV_BLOCKS} KV cache blocks.")

    # ------------------------------------------------------------------ #
    # Singleton accessor                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls) -> "InferenceEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def _decode_token(self, token_id: int) -> str:
        return self.enc.decode([token_id])

    def _make_request(self, input_ids: list[int]) -> Request:
        return Request(
            request_id=str(uuid.uuid4()),
            input_ids=input_ids,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: Optional[int] = DEFAULT_TOP_K,
    ) -> tuple[str, int, int]:
        """
        Full (non-streaming) generation.

        Returns:
            (generated_text, prompt_token_count, generated_token_count)
        """
        input_ids = self._encode(prompt)
        request = self._make_request(list(input_ids))

        with self._gen_lock:
            with torch.no_grad():
                output = self.model.generate_with_cache(
                    request=request,
                    kv_cache_manager=self.kv_cache_manager,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )

        # output is a tensor of shape (1, total_len)
        all_ids = output[0].tolist()
        generated_ids = all_ids[len(input_ids):]
        generated_text = self.enc.decode(all_ids)

        return generated_text, len(input_ids), len(generated_ids)

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: Optional[int] = DEFAULT_TOP_K,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation — yields one decoded token string at a time.

        Runs the synchronous model in a thread so the event loop is not blocked.
        """
        input_ids = self._encode(prompt)
        request = self._make_request(list(input_ids))
        device = self.device
        model = self.model
        kv = self.kv_cache_manager
        enc = self.enc

        # Queue to pass tokens from the worker thread to this async generator
        token_queue: asyncio.Queue[int | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _worker():
            """Runs in a background thread; puts token IDs onto the queue."""
            try:
                from torch.nn import functional as F
                from utils.block import build_logical_blocks, append_decode_token
                from paged_attention.kv_cache_tensor import KVCacheTensor

                req = request
                req.num_computed_tokens = 0
                req.logical_blocks = []

                from utils.block import build_logical_blocks
                req = build_logical_blocks(req, kv_cache_block_size=8)
                kv.allocate(req)

                input_tensor = torch.tensor([req.input_ids], dtype=torch.long, device=device)

                with torch.no_grad():
                    # ── Prefill ──
                    kv_cache_tensor = model._ensure_kv_cache_tensor(kv, device)
                    logits, _ = model(input_tensor, request=req, kv_cache_manager=kv)

                    # ── Decode ──
                    for _ in range(max_new_tokens):
                        next_logits = logits[:, -1, :] / temperature
                        if top_k is not None:
                            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                            next_logits[next_logits < v[:, [-1]]] = float("-inf")
                        probs = F.softmax(next_logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)

                        token_id: int = idx_next.item()
                        req.generated_ids.append(token_id)
                        req = append_decode_token(req, token_id, kv_cache_block_size=8)
                        kv.allocate_last_block(req)

                        # Send token to async side
                        loop.call_soon_threadsafe(token_queue.put_nowait, token_id)

                        logits, _ = model(idx_next, request=req, kv_cache_manager=kv)

                    kv.free(req)
            except Exception as exc:
                print(f"[InferenceEngine._worker] Error: {exc}")
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

        # Run the worker in a thread pool so the event loop stays responsive
        loop.run_in_executor(None, _worker)

        # Yield tokens as they arrive
        while True:
            token_id = await token_queue.get()
            if token_id is None:
                break
            yield enc.decode([token_id])
