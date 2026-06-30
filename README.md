# nanoGPT

A from-scratch implementation of GPT-style models intended to teach and explore the inner workings of Transformers, with focused implementations of GPT-1 and GPT-2 variants.

**What I implemented in this project**
- **Custom tokenizers**: from-scratch `character`, `word`, and `BPE` (Byte Pair Encoding) tokenizers to fully understand tokenization before using a production tokenizer like `tiktoken`.
- **GPT-1 implementation**: core components implemented by hand including `Attention`, `MultiHeadAttention`, `TransformerDecoderBlock`, and the top-level `GPT1` model.
- **GPT-2 implementation and optimizations**: adopted pre-norm, fused QKV projection, batched multi-head attention, and weight initialization consistent with the original GPT-2 settings.
- **Training and data utilities**: convenience scripts such as `prepare_data.py`, `dataset.py`, and `train_gpt2.py` for data preprocessing, batching, and training.

**Technical highlights**
- **Deep understanding of self-attention**: explicit implementation of attention computation and matrix-level optimizations to improve throughput.
- **Pre-norm vs Post-norm**: implemented pre-norm blocks (as used in GPT-2) to increase training stability for deeper networks.
- **Fused QKV & batched heads**: fused Q/K/V projection and parallelized head computation to utilize efficient matrix multiplication on accelerators.
- **End-to-end training flow**: tokenization → dataset batching → forward/backward passes → checkpointing.

**Repository structure**
- **`tokenizers/`**: custom tokenizer implementations (character, BPE, etc.)
- **`models/`**: `bigram.py`, `gpt1.py`, `gpt2.py` model implementations
- **`data/`**: contains example datasets (e.g. `wikitext-103`)
- Top-level scripts: `prepare_data.py`, `dataset.py`, `train_gpt2.py` — data pipeline and training entrypoints

**Quick start (example)**
1. Install dependencies:

```
pip install -r requirements.txt
```

2. Prepare data (example using wikitext):

```
python prepare_data.py --input data/wikitext-103/wiki.train.tokens --output data/processed
```

3. Train a small experiment:

```
python train_gpt2.py
```

Adjust training parameters and hyperparameters in `train_gpt2.py` to run on GPU/CPU or change model size.

**Notes**
- This project is primarily for learning and experimentation, not a production-ready implementation.

- If you want more detailed documentation (fine-tuning guide, example configs, or a demo notebook), tell me which you'd prefer and I will add it.

**Sources / Inspiration**

This project is inspired by and built while following the excellent educational tutorials by Andrej Karpathy on implementing language models from scratch.

* **Let's reproduce GPT-2 (124M)**
  https://www.youtube.com/watch?v=l8pRSuU81PU

* **Let's build the GPT Tokenizer**
  https://www.youtube.com/watch?v=zduSFxRajkE

These videos provide an in-depth explanation of transformer architectures, language modeling, and GPT implementation, and served as the primary learning resources for this project.



