from dataclasses import dataclass, field
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
    input_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    logical_blocks: list[LogicalBlock] = field(default_factory=list)
    finished: bool = False