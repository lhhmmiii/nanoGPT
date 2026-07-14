

class KVCacheBlock:
    block_id: int
    block_hash: bytes = b''
    ref_cnt: int = 0

    def __init__(self, block_id: int, block_hash: bytes = b''):
        self.block_id = block_id
        self.block_hash = block_hash

    @property
    def is_free(self):
        return self.ref_cnt == 0

class KVCacheManager:
    def __init__(self, num_blocks: int):
        self.blocks = [
            KVCacheBlock(i) for i in range(num_blocks)
        ]
        self.num_allocated_blocks = 0
    
    def allocate_block(self, block_hash: bytes):
        if self.num_allocated_blocks == len(self.blocks):
            raise Exception("No free blocks available")
        
        block = self.blocks[self.num_allocated_blocks]
        block.block_hash = block_hash
        block.ref_cnt = 1
        self.num_allocated_blocks += 1
        return block.block_id
    
    def free(self, block_id: int):
        """
        Free the block by block id
        """
        self.blocks[block_id].ref_cnt = 0
        self.blocks[block_id].block_hash = b''
        self.num_allocated_blocks -= 1

if __name__ == "__main__":
    pass 


    