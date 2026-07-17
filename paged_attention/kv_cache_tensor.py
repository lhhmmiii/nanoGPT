import torch

class KVCacheTensor:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_heads: int,
        kv_block_size: int,
        head_dim: int,
        dtype: torch.float16,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_block_size = kv_block_size
        self.head_dim = head_dim
        self.num_blocks = num_blocks

        # Initialize real tensors
        self.data = torch.empty(
            (num_layers, 2, num_blocks, num_heads, kv_block_size, head_dim),
            dtype=dtype,
            device=device,
        )

    def read_block(
        self,
        layer_id: int,
        block_id: int,
        token_offset: int,
        length: int,
    ) -> torch.Tensor:
        """
        Read KV cache for a single block.
        
        Args:
            layer_id: Layer index
            block_id: Block index
            token_offset: Token offset in the block
            length: Number of tokens to read
        
        Returns:
            Tensor of shape (2, num_heads, length, head_dim)
        """
        k = self.data[layer_id, 0, block_id, :, token_offset:token_offset + length, :] 
        v = self.data[layer_id, 1, block_id, :, token_offset:token_offset + length, :] 
        return k, v
    
    def write_block(
        self,
        layer_id: int,
        block_id: int,
        token_offset: int,
        data: torch.Tensor,
    ) -> None:
        """
        Write KV cache for a single block.
        
        Args:
            layer_id: Layer index
            block_id: Block index
            token_offset: Token offset in the block
            data: Tensor of shape (2, num_heads, length, head_dim)
        """
        length = data.size(2)
        self.data[layer_id, :, block_id, :, token_offset:token_offset + length, :] = data    
        