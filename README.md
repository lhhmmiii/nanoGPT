# nanoGPT

A from-scratch implementation of GPT1 and GPT2 to understand how the architecture works.

## Tokenizers

Implemented three tokenizers from scratch: character-level, word-level, and Byte Pair Encoding (BPE). The goal was to understand how tokenization works before using a production tokenizer (tiktoken) for actual training.

## GPT1

Built the full transformer stack from scratch: `Attention`, `MultiHeadAttention`, `TransformerDecoderBlock`, and the top-level `GPT1` model.

The main takeaway was understanding how self-attention actually computes relationships between tokens and how the decoder-only architecture differs from the original encoder-decoder transformer.

## GPT2

Key architectural differences I noticed when implementing GPT2 vs GPT1:

- **Pre-norm instead of post-norm** — LayerNorm is applied before the attention and MLP sublayers, not after.
- **Fused QKV projection** — instead of three separate linear layers for Q, K, V, a single linear layer projects to all three at once and then splits. Cleaner and slightly more efficient.
- **Batched multi-head attention** — instead of a ModuleList of attention heads concatenated at the end, the heads are computed in parallel using a single matrix by reshaping the tensor. This leverages matrix multiplication rather than looping.
- **Weight initialization** — weights drawn from N(0, 0.02²) and biases initialized to 0. A neutral starting point that matches the original GPT2 paper.
- Load pretrained model(openai/gpt2) from Huggingface
