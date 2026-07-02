import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.gpt2 import GPT2, GPT2Config


def build_model(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
    )
    model = GPT2(cfg).eval()
    return model, cfg


@torch.no_grad()
def time_generation(model, cfg, prompt_len, gen_len, repeats=3):
    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len))

    # warmup (compiles kernels / warms caches, not timed)
    model.generate_with_cache(idx, max_new_tokens=4)
    model.generate_no_cache(idx, max_new_tokens=4)

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.generate_with_cache(idx, max_new_tokens=gen_len)
    t_cache = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        model.generate_no_cache(idx, max_new_tokens=gen_len)
    t_nocache = (time.perf_counter() - t0) / repeats

    return t_cache, t_nocache


def main():
    model, cfg = build_model()
    prompt_len = 32
    gen_lengths = [16, 32, 64, 128, 256]

    results = []
    print(f"{'new tokens':>10} | {'no cache (s)':>14} | {'kv cache (s)':>14} | {'speedup':>8}")
    print("-" * 56)
    for gen_len in gen_lengths:
        t_cache, t_nocache = time_generation(model, cfg, prompt_len, gen_len)
        speedup = t_nocache / t_cache
        results.append((gen_len, t_nocache, t_cache, speedup))
        print(f"{gen_len:>10} | {t_nocache:>14.4f} | {t_cache:>14.4f} | {speedup:>7.2f}x")

    gen_lens = [r[0] for r in results]
    nocache_times = [r[1] for r in results]
    cache_times = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(gen_lens, nocache_times, marker="o", label="no KV cache", color="#d62728")
    ax1.plot(gen_lens, cache_times, marker="o", label="with KV cache", color="#2ca02c")
    ax1.set_xlabel("tokens generated")
    ax1.set_ylabel("wall-clock time (s)")
    ax1.set_title(f"Generation latency (prompt_len={prompt_len})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    speedups = [r[3] for r in results]
    ax2.plot(gen_lens, speedups, marker="o", color="#1f77b4")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("tokens generated")
    ax2.set_ylabel("speedup (no-cache time / cache time)")
    ax2.set_title("KV cache speedup vs. sequence length")
    ax2.grid(alpha=0.3)

    fig.suptitle("nanoGPT: KV Cache vs. No Cache — Generation Benchmark", fontsize=12)
    fig.tight_layout()
    out_path = "kv_cache_benchmark.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()