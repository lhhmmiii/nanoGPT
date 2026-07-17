"""
Benchmark: Prefix Caching with Paged Attention
================================================
Demonstrates the memory advantage of paged KV cache when multiple requests
share a common prefix.

Scenario:
  N requests share the first `prefix_len` tokens (e.g. system prompt) but
  have unique suffixes.  With paged attention the physical blocks for the
  shared prefix are allocated **once** and reused across requests (via block
  hash lookup), whereas the standard KV cache must store a full, independent
  copy per request.

Metrics:
  1. Memory — physical blocks allocated for N concurrent requests
  2. Time   — wall-clock time to process N requests sequentially
  3. Cache  — prefix cache hit rate in the paged KV cache manager
"""

import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.gpt2 import GPT2 as GPT2Standard, GPT2Config as GPT2StandardConfig
from models.gpt2_paged import GPT2 as GPT2Paged, GPT2Config as GPT2PagedConfig
from paged_attention.kv_cache_manager import KVCacheManager
from schemas import Request
from utils.block import build_logical_blocks

KV_BLOCK_SIZE = 8
NUM_PAGED_BLOCKS = 1024  # large pool so we never OOM during the benchmark


# ─────────────────── instrumented cache manager ───────────────────


class InstrumentedKVCacheManager(KVCacheManager):
    """Thin wrapper that counts cache hits / misses."""

    def __init__(self, num_blocks: int):
        super().__init__(num_blocks)
        self.cache_hits = 0
        self.cache_misses = 0

    def _allocate_block(self, block_hash=None):
        if block_hash is not None and block_hash in self.cached_blocks:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return super()._allocate_block(block_hash)

    def reset_counters(self):
        self.cache_hits = 0
        self.cache_misses = 0


# ─────────────────── model builders ───────────────────


def build_standard_model(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2StandardConfig(
        vocab_size=50257, block_size=1024,
        n_layer=12, n_head=12, n_embd=768, dropout=0.0,
    )
    return GPT2Standard(cfg).eval(), cfg


def build_paged_model(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2PagedConfig(
        vocab_size=50257, block_size=1024,
        n_layer=12, n_head=12, n_embd=768, dropout=0.0,
    )
    return GPT2Paged(cfg).eval(), cfg


# ─────────────────── request generators ───────────────────


def make_shared_prefix_requests(n, prefix_len, suffix_len, vocab_size):
    """N requests that share an identical prefix."""
    prefix = torch.randint(0, vocab_size, (prefix_len,)).tolist()
    return [
        prefix + torch.randint(0, vocab_size, (suffix_len,)).tolist()
        for _ in range(n)
    ]


def make_unique_requests(n, total_len, vocab_size):
    """N requests with completely independent tokens (no shared prefix)."""
    return [torch.randint(0, vocab_size, (total_len,)).tolist() for _ in range(n)]


# ─────────────────── memory benchmark ───────────────────


def memory_standard_concurrent(requests):
    """
    Standard KV cache: each request holds its own independent cache.
    Equivalent blocks = sum(ceil(len / block_size)) for every request.
    """
    return sum(math.ceil(len(ids) / KV_BLOCK_SIZE) for ids in requests)


def memory_paged_concurrent(requests):
    """
    Allocate paged blocks for ALL requests concurrently (without freeing
    in-between) so shared-prefix blocks are deduplicated via block-hash
    lookup.

    Returns (blocks_used, cache_hits, cache_misses).
    """
    kv_mgr = InstrumentedKVCacheManager(num_blocks=NUM_PAGED_BLOCKS)
    all_reqs = []

    for i, input_ids in enumerate(requests):
        req = Request(request_id=f"req-{i}", input_ids=list(input_ids))
        req = build_logical_blocks(req, kv_cache_block_size=KV_BLOCK_SIZE)
        kv_mgr.allocate(req)
        all_reqs.append(req)

    blocks_used = kv_mgr.num_allocated_blocks
    hits, misses = kv_mgr.cache_hits, kv_mgr.cache_misses

    # cleanup
    for req in all_reqs:
        kv_mgr.free(req)

    return blocks_used, hits, misses


# ─────────────────── time benchmark ───────────────────


@torch.no_grad()
def time_standard_cache_n(model, cfg, requests, gen_len, repeats=2):
    """Process N requests sequentially with the standard KV cache."""
    # warmup
    idx = torch.tensor([requests[0]], dtype=torch.long)
    model.generate_with_cache(idx, max_new_tokens=4)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for input_ids in requests:
            idx = torch.tensor([input_ids], dtype=torch.long)
            model.generate_with_cache(idx, max_new_tokens=gen_len)
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def time_paged_cache_n(model, cfg, requests, gen_len, repeats=2):
    """
    Process N requests sequentially with paged KV cache.

    The same KVCacheManager is kept across requests so that blocks freed by
    request i remain in the ``cached_blocks`` dict.  When request i+1 arrives
    with the same prefix, those blocks score cache hits and the physical
    memory is reused.
    """
    # warmup
    kv_mgr = KVCacheManager(num_blocks=NUM_PAGED_BLOCKS)
    req = Request(request_id="warmup", input_ids=list(requests[0]))
    model.generate_with_cache(req, kv_mgr, max_new_tokens=4)
    model.kv_cache_tensor = None

    t0 = time.perf_counter()
    for _ in range(repeats):
        kv_mgr = InstrumentedKVCacheManager(num_blocks=NUM_PAGED_BLOCKS)
        model.kv_cache_tensor = None
        for i, input_ids in enumerate(requests):
            req = Request(request_id=f"req-{i}", input_ids=list(input_ids))
            model.generate_with_cache(req, kv_mgr, max_new_tokens=gen_len)
    return (time.perf_counter() - t0) / repeats


# ─────────────────── main ───────────────────


def main():
    print("Building standard GPT-2 model …")
    model_std, cfg_std = build_standard_model()
    print("Building paged GPT-2 model …")
    model_paged, cfg_paged = build_paged_model()

    prefix_len = 64
    suffix_len = 16
    total_len = prefix_len + suffix_len  # 80
    gen_len = 16
    request_counts = [1, 2, 4, 8, 16]

    # ── Collect results ──
    mem_std_shared, mem_paged_shared = [], []
    mem_std_unique, mem_paged_unique = [], []
    hit_rates_shared, hit_rates_unique = [], []
    time_std_shared, time_paged_shared = [], []

    header = (
        f"{'N':>3} | {'std blocks':>10} | {'paged blocks':>12} | "
        f"{'saved %':>8} | {'hits':>5} | {'misses':>7} | {'hit rate':>9} | "
        f"{'std time (s)':>13} | {'paged time (s)':>15}"
    )

    print(f"\n{'=' * 20}  SHARED PREFIX  {'=' * 20}")
    print(f"prefix_len={prefix_len}  suffix_len={suffix_len}  gen_len={gen_len}\n")
    print(header)
    print("-" * len(header))

    for n in request_counts:
        reqs = make_shared_prefix_requests(n, prefix_len, suffix_len, cfg_std.vocab_size)

        # memory
        m_std = memory_standard_concurrent(reqs)
        m_paged, hits, misses = memory_paged_concurrent(reqs)
        saved = (1 - m_paged / m_std) * 100 if m_std > 0 else 0
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        mem_std_shared.append(m_std)
        mem_paged_shared.append(m_paged)
        hit_rates_shared.append(hr)

        # time
        t_std = time_standard_cache_n(model_std, cfg_std, reqs, gen_len)
        t_paged = time_paged_cache_n(model_paged, cfg_paged, reqs, gen_len)

        time_std_shared.append(t_std)
        time_paged_shared.append(t_paged)

        print(
            f"{n:>3} | {m_std:>10} | {m_paged:>12} | {saved:>7.1f}% | "
            f"{hits:>5} | {misses:>7} | {hr:>8.1f}% | "
            f"{t_std:>13.4f} | {t_paged:>15.4f}"
        )

    # ── Control: unique prefixes ──
    print(f"\n{'=' * 20}  UNIQUE PREFIX (control)  {'=' * 20}\n")
    print(
        f"{'N':>3} | {'std blocks':>10} | {'paged blocks':>12} | "
        f"{'saved %':>8} | {'hits':>5} | {'misses':>7} | {'hit rate':>9}"
    )
    print("-" * 72)

    for n in request_counts:
        reqs = make_unique_requests(n, total_len, cfg_std.vocab_size)
        m_std = memory_standard_concurrent(reqs)
        m_paged, hits, misses = memory_paged_concurrent(reqs)
        saved = (1 - m_paged / m_std) * 100 if m_std > 0 else 0
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        mem_std_unique.append(m_std)
        mem_paged_unique.append(m_paged)
        hit_rates_unique.append(hr)

        print(
            f"{n:>3} | {m_std:>10} | {m_paged:>12} | {saved:>7.1f}% | "
            f"{hits:>5} | {misses:>7} | {hr:>8.1f}%"
        )

    # ─────────────────── plot ───────────────────

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1 — Memory: blocks allocated (concurrent)
    ax = axes[0]
    ax.plot(request_counts, mem_std_shared,   "o-",  label="standard (shared prefix)",  color="#d62728")
    ax.plot(request_counts, mem_paged_shared, "s-",  label="paged (shared prefix)",     color="#2ca02c")
    ax.plot(request_counts, mem_std_unique,   "o--", label="standard (unique prefix)",  color="#d62728", alpha=0.5)
    ax.plot(request_counts, mem_paged_unique, "s--", label="paged (unique prefix)",     color="#2ca02c", alpha=0.5)
    ax.set_xlabel("number of concurrent requests (N)")
    ax.set_ylabel("physical blocks allocated")
    ax.set_title("Memory: blocks for N concurrent requests")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2 — Memory savings %
    ax = axes[1]
    savings_shared = [(1 - p / s) * 100 for s, p in zip(mem_std_shared, mem_paged_shared)]
    savings_unique = [(1 - p / s) * 100 for s, p in zip(mem_std_unique, mem_paged_unique)]
    ax.plot(request_counts, savings_shared, "D-", label="shared prefix", color="#1f77b4")
    ax.plot(request_counts, savings_unique, "D--", label="unique prefix", color="#9467bd")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("number of concurrent requests (N)")
    ax.set_ylabel("block savings (%)")
    ax.set_title("Paged KV cache: memory savings vs standard")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3 — Cache hit rate
    ax = axes[2]
    ax.plot(request_counts, hit_rates_shared, "^-", label="shared prefix", color="#2ca02c")
    ax.plot(request_counts, hit_rates_unique, "^--", label="unique prefix", color="#d62728")
    ax.set_xlabel("number of concurrent requests (N)")
    ax.set_ylabel("cache hit rate (%)")
    ax.set_title("Prefix cache hit rate")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Prefix Caching Benchmark  (prefix={prefix_len}, suffix={suffix_len}, gen={gen_len})",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = "prefix_caching_benchmark.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
