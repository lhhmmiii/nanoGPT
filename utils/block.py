from hashlib import sha256
import math
from typing import Optional

from schemas import Request, LogicalBlock
from paged_attention.kv_cache_manager import KVCacheManager

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


def build_logical_blocks(request: Request, kv_cache_block_size: int = 8) -> Request:
    """
    Build the logical block structure and compute hashes only for full blocks.

    Physical block allocation is intentionally deferred and handled later by
    the Scheduler.
    """
    input_ids = request.input_ids
    needed_num_blocks = math.ceil(len(input_ids) / kv_cache_block_size)

    logical_blocks = []
    prev_block_hash = None

    for i in range(needed_num_blocks):
        token_ids = input_ids[i * kv_cache_block_size:(i + 1) * kv_cache_block_size]
        is_full = len(token_ids) == kv_cache_block_size

        # Compute a hash only for full blocks so they can participate
        # in prefix caching.
        if is_full:
            block_hash = compute_block_hash(prev_block_hash, token_ids)
            prev_block_hash = block_hash
        else:
            block_hash = None

        logical_block = LogicalBlock(
            block_idx=i,
            token_ids=token_ids,
            block_hash=block_hash,
            physical_block=None
        )
        logical_blocks.append(logical_block)

    request.logical_blocks = logical_blocks
    return request

def append_decode_token(
    request: Request,
    token_id: int,
    kv_cache_block_size: int = 8,
) -> Request:
    """
    Update logical blocks during the decode phase by appending one generated token.

    - Append the token to the last logical block if it is not full.
    - Otherwise, create a new logical block.
    - Compute the block hash only when a block becomes full.
    """

    # Keep input_ids consistent
    request.input_ids.append(token_id)

    logical_blocks = request.logical_blocks

    # No logical block yet
    if len(logical_blocks) == 0:
        logical_blocks.append(
            LogicalBlock(
                block_idx=0,
                token_ids=[token_id],
                block_hash=None,
                physical_block=None,
            )
        )
        return request

    last_block = logical_blocks[-1]

    # Current block still has space
    if len(last_block.token_ids) < kv_cache_block_size:
        last_block.token_ids.append(token_id)

        # Block becomes full after appending
        if len(last_block.token_ids) == kv_cache_block_size:
            prev_hash = (
                logical_blocks[-2].block_hash
                if len(logical_blocks) > 1
                else None
            )

            last_block.block_hash = compute_block_hash(
                prev_hash,
                last_block.token_ids,
            )

        return request

    # Last block is already full -> create a new block
    logical_blocks.append(
        LogicalBlock(
            block_idx=len(logical_blocks),
            token_ids=[token_id],
            block_hash=None,
            physical_block=None,
        )
    )

    return request