import pytest
import torch
import tiktoken

from models.gpt2_paged import GPT2Config, GPT2
from paged_attention.kv_cache_manager import KVCacheManager
from schemas import Request
from utils.block import build_logical_blocks, append_decode_token


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


def test_gpt2_paged_forward_inference(small_model):
    idx = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        logits, loss = small_model(idx)
    assert logits.shape == (2, 1, 100)
    assert loss is None


def test_gpt2_paged_forward_with_targets(small_model):
    idx = torch.randint(0, 100, (2, 10))
    targets = torch.randint(0, 100, (2, 10))
    logits, loss = small_model(idx, targets=targets)
    assert logits.shape == (2, 10, 100)
    assert loss is not None
    assert loss.item() > 0


def test_gpt2_paged_forward_with_kv_cache(small_model):
    kv_manager = KVCacheManager(num_blocks=16)
    input_ids = [10, 20, 30, 40, 50]
    req = Request(request_id="req_test_forward", input_ids=list(input_ids))

    req = build_logical_blocks(req, kv_cache_block_size=8)
    kv_manager.allocate(req)

    # Prefill step
    input_tensor = torch.tensor([req.input_ids], dtype=torch.long)
    with torch.no_grad():
        logits, _ = small_model(input_tensor, request=req, kv_cache_manager=kv_manager)

    assert logits.shape == (1, 1, small_model.config.vocab_size)
    assert req.num_computed_tokens == len(input_ids)

    # Single-token decode step
    next_token = torch.tensor([[60]], dtype=torch.long)
    req.generated_ids.append(60)
    req = append_decode_token(req, 60, kv_cache_block_size=8)
    kv_manager.allocate_last_block(req)

    with torch.no_grad():
        logits_decode, _ = small_model(next_token, request=req, kv_cache_manager=kv_manager)

    assert logits_decode.shape == (1, 1, small_model.config.vocab_size)
    assert req.num_computed_tokens == len(input_ids) + 1

    kv_manager.free(req)
    assert kv_manager.num_free_blocks == 16


def test_gpt2_paged_generate_no_cache(small_model):
    idx = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = small_model.generate_no_cache(idx, max_new_tokens=6, top_k=1)
    assert output.shape == (1, 11)
    assert torch.equal(output[:, :5], idx)


def test_gpt2_paged_generate_with_cache(small_model):
    kv_manager = KVCacheManager(num_blocks=32)
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    req = Request(request_id="req_gen_cache", input_ids=list(input_ids))

    with torch.no_grad():
        output = small_model.generate_with_cache(
            request=req,
            kv_cache_manager=kv_manager,
            max_new_tokens=12,
            top_k=1,
        )

    # 10 prompt tokens + 12 generated tokens = 22 tokens (unless stopped by EOS)
    assert output.shape[0] == 1
    assert output.shape[1] == 22
    assert output[0, :10].tolist() == input_ids
    # Ensure all blocks are freed after generation
    assert kv_manager.num_free_blocks == 32


def test_gpt2_paged_cache_consistency(small_model):
    # Verify that greedy decoding with paged KV cache matches generate_no_cache
    kv_manager = KVCacheManager(num_blocks=32)
    input_ids = [5, 12, 18, 24, 31, 40]
    idx = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        out_no_cache = small_model.generate_no_cache(idx.clone(), max_new_tokens=8, top_k=1)
        req = Request(request_id="req_consistency", input_ids=list(input_ids))
        out_cache = small_model.generate_with_cache(
            request=req,
            kv_cache_manager=kv_manager,
            max_new_tokens=8,
            top_k=1,
        )

    assert torch.equal(out_no_cache, out_cache)


def test_gpt2_paged_eos_stop(small_config):
    torch.manual_seed(42)
    model = GPT2(small_config).eval()
    kv_manager = KVCacheManager(num_blocks=16)

    # When max_new_tokens is 0, only prompt is returned and blocks are freed
    req = Request(request_id="req_eos", input_ids=[1, 2, 3])
    with torch.no_grad():
        output = model.generate_with_cache(req, kv_cache_manager=kv_manager, max_new_tokens=0)
    assert output.shape == (1, 3)
    assert kv_manager.num_free_blocks == 16


def test_gpt2_paged_sample_generation():
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "Homarus gammarus is a large crustacean"
    input_ids = tokenizer.encode(sample_text)
    config = GPT2Config(vocab_size=50257, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT2(config).eval()
    kv_manager = KVCacheManager(num_blocks=32)
    req = Request(request_id="sample_test_req", input_ids=input_ids)

    with torch.no_grad():
        output = model.generate_with_cache(
            request=req,
            kv_cache_manager=kv_manager,
            max_new_tokens=5,
        )
    assert output.shape == (1, len(input_ids) + 5)
    assert kv_manager.num_free_blocks == 32


if __name__ == "__main__":
    config = GPT2Config()
    model = GPT2.from_pretrained("gpt2")
    # model = GPT2.load_checkpoint('checkpoints/gpt2/scratch/run1/best.pt', map_location='cpu')[0]
    sample_test = "Homarus gammarus is a large crustacean , with a body length up to 60"
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(sample_test)
    req = Request(request_id="demo_req", input_ids=input_ids)
    kv_cache_manager = KVCacheManager(num_blocks=128)
    output = model.generate_with_cache(req, kv_cache_manager=kv_cache_manager, max_new_tokens=10)
    tokenized_output = tokenizer.decode(output.squeeze().tolist())
    print(f"Input: {sample_test}")
    print(f"Output: {tokenized_output}")
