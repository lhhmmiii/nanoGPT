import pytest
from schemas import Request, LogicalBlock
from paged_attention.kv_cache_manager import KVCacheManager, KVCacheBlock

def test_kv_cache_manager_initialization():
    num_blocks = 4
    manager = KVCacheManager(num_blocks=num_blocks)
    
    assert manager.num_blocks == num_blocks
    assert manager.num_free_blocks == num_blocks
    assert manager.num_allocated_blocks == 0
    assert len(manager.blocks) == num_blocks
    
    # Check doubly-linked free list links
    for i in range(num_blocks):
        block = manager.blocks[i]
        assert block.block_id == i
        assert block.ref_cnt == 0
        assert block.is_free is True
        assert block.block_hash is None
        
        if i > 0:
            assert block.prev_free_block == manager.blocks[i - 1]
        else:
            assert block.prev_free_block is None
            
        if i < num_blocks - 1:
            assert block.next_free_block == manager.blocks[i + 1]
        else:
            assert block.next_free_block is None

    assert manager.free_head == manager.blocks[0]
    assert manager.free_tail == manager.blocks[-1]
    assert len(manager.cached_blocks) == 0

def test_request_allocation_and_free():
    manager = KVCacheManager(num_blocks=5)
    
    # Create request with logical blocks
    block_hash_0 = b"hash_block_0"
    block_hash_1 = b"hash_block_1"
    
    lb0 = LogicalBlock(block_idx=0, token_ids=[1, 2, 3, 4], block_hash=block_hash_0)
    lb1 = LogicalBlock(block_idx=1, token_ids=[5, 6, 7, 8], block_hash=block_hash_1)
    
    req = Request(
        request_id="req_test",
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        logical_blocks=[lb0, lb1]
    )
    
    # Test allocate
    manager.allocate(req)
    
    assert lb0.physical_block is not None
    assert lb1.physical_block is not None
    assert lb0.physical_block.block_id == 0
    assert lb1.physical_block.block_id == 1
    assert lb0.physical_block.block_hash == block_hash_0
    assert lb1.physical_block.block_hash == block_hash_1
    assert manager.num_free_blocks == 3
    
    # Test get_block_ids
    block_ids = manager.get_block_ids(req)
    assert block_ids == [0, 1]
    
    # Test free
    manager.free(req)
    assert lb0.physical_block is None
    assert lb1.physical_block is None
    assert manager.num_free_blocks == 5
    
    # Verify both physical blocks are free in the manager
    assert manager.blocks[0].is_free is True
    assert manager.blocks[1].is_free is True

def test_get_block_ids():
    manager = KVCacheManager(num_blocks=5)
    
    # 1. Empty logical blocks
    req_empty = Request(request_id="req_empty", input_ids=[])
    assert manager.get_block_ids(req_empty) == []
    
    # 2. Logical blocks are not allocated yet (physical_block is None)
    lb0 = LogicalBlock(block_idx=0, token_ids=[1, 2, 3, 4], block_hash=b"hash_0")
    lb1 = LogicalBlock(block_idx=1, token_ids=[5, 6, 7, 8], block_hash=b"hash_1")
    req = Request(
        request_id="req_test",
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        logical_blocks=[lb0, lb1]
    )
    assert manager.get_block_ids(req) == []
    
    # 3. Partially allocated
    manager.allocate(req)
    print(manager.get_block_ids(req))
    assert manager.get_block_ids(req) == [lb0.physical_block.block_id, lb1.physical_block.block_id]