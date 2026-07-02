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

## What is implemented

- Custom tokenizers: character, word, and BPE-style tokenization logic.
- GPT-1 implementation: attention, multi-head attention, decoder blocks, and the full model stack.
- GPT-2 implementation: pre-norm architecture, fused QKV-style projection logic, and optimized batched attention.
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

