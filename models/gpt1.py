import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from train import train_model
from tokenizers.character import CharacterTokenizer

# Hyperparameters
embedding_dim = 768
block_size = 512
num_heads = 12
decoder_layers = 12

learning_rate = 2.5e-4
batch_size = 64
num_epochs = 100 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPT1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, block_size, decoder_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, block_size) for _ in range(decoder_layers)
        ])
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass of the GPT-1 model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T) where B is the batch size and T is the sequence length.
        """
        x = self.embedding(x)  # (B, T, C)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, T)
        position_embedding = self.position_embedding(positions)  # (1, T, C)
        x = x + position_embedding  # (B, T, C)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.linear(x)  # (B, T, vocab_size)
        return x
    
    def generate(self, tokenizer, prompt, block_size, max_length=100):
        """
        Generate text using the GPT-1 model.
        
        Args:
            tokenizer (CharacterTokenizer): Tokenizer for encoding and decoding text.
            prompt (str): Input prompt for text generation.
            block_size (int): Maximum sequence length for the model.
            max_length (int): Maximum length of the generated text.
        
        Returns:
            str: Generated text.
        """
        self.eval()
        with torch.no_grad():
            input_ids = tokenizer.encode(prompt)
            input_ids = input_ids[-block_size:]  # Truncate to block size
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
            
            for _ in range(max_length):
                logits = self.forward(input_tensor)  # (1, T, vocab_size)
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Get the index of the highest logit
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)  # Append to input
                
            generated_ids = input_tensor.squeeze(0).tolist()
            generated_text = tokenizer.decode(generated_ids)
            return generated_text 

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(
            embedding_dim, embedding_dim // num_heads, block_size, num_heads=num_heads
        )
        self.ffn = FeedForward(embedding_dim, embedding_dim * 4)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """
        Forward pass of the decoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is the batch size, T is the sequence length, and C is the embedding dimension.
        """
        # Multi-head attention
        attn_output = self.multihead_attention(x)  # (B, T, C)
        x = self.layernorm1(x + attn_output)  # (B, T, C)

        # Feedforward network
        ffn_output = self.ffn(x)  # (B, T, C)
        x = self.layernorm2(x + ffn_output)  # (B, T, C)

        return x
        

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size, num_heads):
        super().__init__()
        self.multihead_attn = nn.ModuleList([Attention(embedding_dim, head_dim, block_size) for _ in range(num_heads)])
        
    def forward(self, x):
        """
        Forward pass of the multi-head attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is the batch size, T is the sequence length, and C is the embedding dimension.
        """
        attn_outputs = [attn(x) for attn in self.multihead_attn]  # List of (B, T, head_dim)
        out = torch.cat(attn_outputs, dim=-1)  # (B, T, head_dim * num_heads)
        return out
        
class Attention(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size=8):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.register_buffer("tril_mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Forward pass of the attention block.
        """
        T = x.size(1)
        q = self.query(x) # (B, T, head_dim)
        k = self.key(x) # (B, T, head_dim)
        v = self.value(x) # (B, T, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5) # (B, T, T)
        attn_scores = attn_scores.masked_fill(
            self.tril_mask[:T, :T] == 0,
            float("-inf")
        ) # (B, T, T)
        attn_weights = torch.softmax(attn_scores, dim =-1) # (B, T, T)

        # Apply attention weights to values
        out = torch.matmul(attn_weights, v) # (B, T, head_dim)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass of the feedforward block.
        """
        x = self.linear1(x) # (B, T, hidden_dim)
        x = self.relu(x) # (B, T, hidden_dim)
        x = self.dropout(x)
        x = self.linear2(x) # (B, T, embedding_dim)
        return x