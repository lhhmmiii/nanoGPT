from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import Request


@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: bytes | None = None
    ref_cnt: int = 0
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    @property
    def is_free(self) -> bool:
        """Returns True if the block is free (reference count is 0)."""
        return self.ref_cnt == 0


class KVCacheManager:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        
        self.free_head: Optional[KVCacheBlock] = self.blocks[0]
        self.free_tail: Optional[KVCacheBlock] = self.blocks[-1]
        
        for i in range(num_blocks):
            if i > 0:
                self.blocks[i].prev_free_block = self.blocks[i - 1]
            if i < num_blocks - 1:
                self.blocks[i].next_free_block = self.blocks[i + 1]
                
        self.free_count = num_blocks
        
        # Hash to block mapping for prefix caching
        self.cached_blocks: Dict[bytes, KVCacheBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        """Returns the number of free blocks in the manager."""
        return self.free_count

    @property
    def num_allocated_blocks(self) -> int:
        """Returns the number of allocated blocks (reference count > 0)."""
        return self.num_blocks - self.free_count

    def allocate(self, request: Request):
        """
        Allocate KV cache blocks for a request.

        Args:
            request: Request need to allocate KV cache blocks.
        """
        for logical_block in request.logical_blocks:
            logical_block.physical_block = self._allocate_block(logical_block.block_hash)

    def free(self, request: Request):
        """
        Free KV cache blocks for a request.

        Args:
            request: Request need to free KV cache blocks.
        """
        for logical_block in request.logical_blocks:
            self._free_block(logical_block.physical_block.block_id)
            logical_block.physical_block = None

    def get_block_ids(self, request: Request) -> list[int]:
        """
        Get the block IDs for a request.

        Args:
            request: Request need to get block IDs.
        
        Returns:
            List of block IDs for the request.
        """
        return [logical_block.physical_block.block_id for logical_block in request.logical_blocks
                if logical_block.physical_block is not None]

    def _allocate_block(self, block_hash: Optional[bytes] = None) -> KVCacheBlock:
        """
        Allocates a physical block.
        """
        # Case 1: Cache hit
        if block_hash is not None and block_hash in self.cached_blocks:
            block = self.cached_blocks[block_hash]
            if block.is_free:
                self._remove_from_free_list(block)
            block.ref_cnt += 1
            return block

        # Case 2: Cache miss, allocate from the free list (evicts LRU block if cached)
        if self.free_head is None:
            raise MemoryError("No free blocks available in KV cache")

        block = self.free_head
        self._remove_from_free_list(block)

        # Evict old cache entry if it exists
        if block.block_hash:
            self.cached_blocks.pop(block.block_hash, None)

        # Update block metadata
        block.block_hash = block_hash if block_hash is not None else None
        block.ref_cnt = 1

        # Register new cache entry
        if block_hash is not None:
            self.cached_blocks[block_hash] = block

        return block

    def _free_block(self, block_id: int) -> None:
        """
        Decrements the reference count of a block.
        If the reference count reaches 0, the block is appended to the tail of the free list.
        
        Args:
            block_id: The ID of the block to free.
            
        Raises:
            ValueError: If the block ID is invalid or the block is already free.
        """
        if not (0 <= block_id < self.num_blocks):
            raise ValueError(f"Invalid block_id: {block_id}. Must be between 0 and {self.num_blocks - 1}.")

        block = self.blocks[block_id]
        if block.is_free:
            raise ValueError(f"Double free error: Block {block_id} is already free.")

        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self._append_to_free_list(block)

    def _remove_from_free_list(self, block: KVCacheBlock) -> None:
        """
        Removes a block from the doubly linked free list.
        """
        if block.prev_free_block is not None:
            block.prev_free_block.next_free_block = block.next_free_block
        else:
            self.free_head = block.next_free_block

        if block.next_free_block is not None:
            block.next_free_block.prev_free_block = block.prev_free_block
        else:
            self.free_tail = block.prev_free_block

        block.prev_free_block = None
        block.next_free_block = None
        self.free_count -= 1

    def _append_to_free_list(self, block: KVCacheBlock) -> None:
        """
        Appends a block to the tail of the doubly linked free list.
        """
        block.next_free_block = None
        block.prev_free_block = self.free_tail

        if self.free_tail is not None:
            self.free_tail.next_free_block = block
        else:
            self.free_head = block

        self.free_tail = block
        self.free_count += 1

if __name__ == "__main__":
    pass