# nanoGPT

A hands-on implementation of GPT-style models from first principles, designed for learning, experimentation, and practical fine-tuning. This project covers the full journey from tokenizer design and transformer internals to training and adaptation of GPT-2.

## Why this project stands out

- Built GPT-style architectures from scratch, including core transformer components rather than relying only on high-level abstractions.
- Implemented custom tokenizers to understand the preprocessing layer that powers modern language models.
- Created a reproducible training pipeline for data preparation, model training, and checkpointing.
- Recently completed a GPT-2 fine-tuning run and visualized the training loss curve as part of the experiment.

## Recent milestone: GPT-2 fine-tuning

I recently fine-tuned GPT-2 on the WikiText-103 dataset and monitored the learning curve to evaluate convergence. WikiText-103 contains approximately 109 million tokens. While relatively small for pretraining a language model from scratch, it is sufficient for fine-tuning. The training loss progression is shown below:

![Fine-tuning loss curve](assests/fine_tuning_gpt2.png)

This experiment highlights:
- a stable optimization trajectory during training,
- a practical end-to-end workflow for adapting a pretrained model,
- and a strong foundation for further experiments with domain-specific text generation.

## Performance Optimization: KV Cache

To improve inference latency, I implemented Key-Value (KV) caching for autoregressive token generation. By caching keys and values from previous tokens in the `CausalSelfAttention` layers, we avoid redundant calculations for past tokens. In subsequent generation steps, we feed only the newly generated token into the model instead of the entire context window.

### Why cache K and V — but not Q?
 
A natural question when implementing KV cache: why do we only store Key and Value tensors, and not Query?
 
Suppose Q and K both have shape `(seq_len, embed_size)`, e.g. `(3, 5)`. The attention score is computed as `Q · K^T`.
 
![Why cache K and V, not Q](assests/kv_cache_explainer.png)
 
Consider the step where we generate the 3rd token(new token):
- To compute the attention row for the new token (`A3`), we only need `Q3` multiplied against **all** of `K1^T, K2^T, K3^T`.
- `Q1` and `Q2` were each used exactly once, at the steps where `A1` and `A2` were computed — they are never reused afterward.
- In contrast, `K1, K2` (and `V1, V2`) get reused again and again at every subsequent generation step, since each new token's query must attend back over **all** past keys/values.

**Takeaway:** since past Query vectors are never reused, there's no benefit to caching Q. But past Key and Value vectors are reused at every future step, so caching them eliminates redundant recomputation — this is exactly why it's called a "KV cache" and not a "QKV cache."

### Benchmark Results

A generation latency benchmark was run on the CPU (prompt length of 32 tokens) comparing generation with and without the KV cache across different sequence lengths:

![KV Cache Speedup](assests/kv_cache_benchmark.png)

This benchmark demonstrates:
- A significant reduction in generation time, scaling from a **2x speedup** for short sequences to over **5.7x speedup** for 256 tokens.
- Constant time complexity per token for attention calculation with cache, compared to quadratic growth without cache.

## What is implemented

- Custom tokenizers: character, word, and BPE-style tokenization logic.
- GPT-1 implementation: attention, multi-head attention, decoder blocks, and the full model stack.
- GPT-2 implementation: pre-norm architecture, fused QKV-style projection logic, optimized batched attention, and **efficient Key-Value (KV) cache generation**.
- Training utilities: data preparation, batching, training entrypoints, and checkpoint support.

## Repository structure

- tokenizers/: tokenizer implementations and helpers
- models/: bigram, gpt1, and gpt2 implementations
- data/: example datasets such as wikitext-103
- Top-level scripts: prepare_data.py, dataset.py, and train_gpt2.py

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data (example using wikitext):

```bash
python prepare_data.py
```

3. Train a small experiment:

```bash
python train_gpt2.py
```

Adjust the hyperparameters in train_gpt2.py to match your hardware, dataset size, or desired model scale.

## Notes

- This project is primarily educational and experimental rather than production-ready.
- If you want, I can also expand this README with a more detailed fine-tuning guide, sample generations, or a benchmark section.

## Sources / Inspiration

This project was inspired by the excellent educational tutorials by Andrej Karpathy on building language models from scratch.

- Let's reproduce GPT-2 (124M): https://www.youtube.com/watch?v=l8pRSuU81PU
- Let's build the GPT Tokenizer: https://www.youtube.com/watch?v=zduSFxRajkE

