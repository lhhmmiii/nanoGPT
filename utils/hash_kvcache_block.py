from hashlib import sha256
from typing import Optional

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