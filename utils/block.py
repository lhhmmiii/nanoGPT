from hashlib import sha256
import math
from typing import Optional

from schemas import Request, LogicalBlock
from paged_attention.kv_cache_manager import KVCacheManager

KV_CACHE_BLOCK_SIZE = 8

def compute_block_hash(
    prev_block_hash: Optional[bytes],
    token_ids: list[int],
) -> bytes:
    """
    Compute the hash for a KV cache block.

    Args:
        prev_block_hash: Hash of the previous block.
        token_ids: Tokens contained in this block.

    Returns:
        32-byte SHA-256 digest.
    """
    h = sha256()

    if prev_block_hash is not None:
        h.update(prev_block_hash)

    for token in token_ids:
        h.update(token.to_bytes(4, byteorder="little", signed=False))

    return h.digest()

def build_logical_blocks(request: Request, kv_cache_manager: KVCacheManager) -> Request:
    """
    Build logical blocks for a request.

    Args:
        request: Request need to build logical blocks
        kv_cache_manager: KV cache manager.
    
    Returns:
        Request with logical blocks.
    """
    input_ids = request.input_ids
    num_blocks = math.ceil(len(input_ids) / KV_CACHE_BLOCK_SIZE)
    logical_blocks = []
    prev_block_hash = None
    for i in range(num_blocks):
        token_ids = input_ids[i*KV_CACHE_BLOCK_SIZE:(i+1)*KV_CACHE_BLOCK_SIZE]
        block_hash = compute_block_hash(prev_block_hash, token_ids)
        physical_block = kv_cache_manager.allocate(block_hash)
        kv_cache_manager.hash_block(physical_block.block_id, block_hash)
        logical_block = LogicalBlock(
            block_idx=i,
            token_ids=token_ids,
            block_hash=block_hash,
            physical_block=physical_block
        )
        logical_blocks.append(logical_block)
        prev_block_hash = block_hash

    request.logical_blocks = logical_blocks
    return request