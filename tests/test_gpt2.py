import pytest
import torch
import tiktoken

from models.gpt2 import GPT2Config, GPT2


@pytest.fixture
def small_config():
    return GPT2Config(
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def small_model(small_config):
    torch.manual_seed(42)
    model = GPT2(small_config)
    model.eval()
    return model


def test_gpt2_forward_inference(small_model):
    idx = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        logits, loss = small_model(idx)
    assert logits.shape == (2, 1, 100)
    assert loss is None


def test_gpt2_forward_with_targets(small_model):
    idx = torch.randint(0, 100, (2, 10))
    targets = torch.randint(0, 100, (2, 10))
    logits, loss = small_model(idx, targets=targets)
    assert logits.shape == (2, 10, 100)
    assert loss is not None
    assert loss.item() > 0


def test_gpt2_generate_no_cache(small_model):
    idx = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = small_model.generate_no_cache(idx, max_new_tokens=6, top_k=1)
    assert output.shape == (1, 11)
    assert torch.equal(output[:, :5], idx)


def test_gpt2_generate_with_cache(small_model):
    idx = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = small_model.generate_with_cache(idx, max_new_tokens=6, top_k=1)
    assert output.shape == (1, 11)
    assert torch.equal(output[:, :5], idx)


def test_gpt2_cache_consistency(small_model):
    idx = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out_no_cache = small_model.generate_no_cache(idx.clone(), max_new_tokens=10, top_k=1)
        out_cache = small_model.generate_with_cache(idx.clone(), max_new_tokens=10, top_k=1)
    assert torch.equal(out_no_cache, out_cache)


def test_gpt2_sample_generation():
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "Hello world"
    input_ids = tokenizer.encode(sample_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    config = GPT2Config(vocab_size=50257, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT2(config).eval()
    with torch.no_grad():
        output = model.generate_with_cache(input_tensor, max_new_tokens=5)
    assert output.shape == (1, len(input_ids) + 5)


if __name__ == "__main__":
    config = GPT2Config()
    model = GPT2.from_pretrained("gpt2")
    # model = GPT2.load_checkpoint('checkpoints/gpt2/scratch/run1/best.pt', map_location='cpu')[0]
    sample_test = "Homarus gammarus is a large crustacean , with a body length up to 60"
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(sample_test)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    output = model.generate_with_cache(input_tensor, max_new_tokens=10)
    tokenized_output = tokenizer.decode(output.squeeze().tolist())
    print(f"Input: {sample_test}")
    print(f"Output: {tokenized_output}")