from dataclasses import dataclass
from paged_attention.kv_cache_manager import KVCacheBlock

@dataclass
class LogicalBlock:
    block_idx: int
    token_ids: list[int]
    block_hash: bytes | None = None
    physical_block: KVCacheBlock | None = None

@dataclass
class Request:
    request_id: str
    request_content: str
    input_ids: list[int]
    generated_ids: list[int]
    logical_blocks: list[LogicalBlock]
    finished: bool