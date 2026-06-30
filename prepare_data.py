import os
import requests
import tiktoken
import numpy as np

train_input_data = "data/wikitext-103/wiki.train.tokens"
val_input_data = "data/wikitext-103/wiki.valid.tokens"
test_input_data = "data/wikitext-103/wiki.test.tokens"

# Read the data from the files
with open(train_input_data, 'r', encoding='utf-8') as f:
    train_data = f.read()

with open(val_input_data, 'r', encoding='utf-8') as f:
    val_data = f.read()

with open(test_input_data, 'r', encoding='utf-8') as f:
    test_data = f.read()


# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
test_ids = enc.encode_ordinary(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile('data/wikitext-103/train.bin')
val_ids.tofile('data/wikitext-103/val.bin')
test_ids.tofile('data/wikitext-103/test.bin')