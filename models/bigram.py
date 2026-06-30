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
embedding_dim = 128
batch_size = 32
learning_rate = 1e-3
num_epochs = 10 
block_size = 8 
batch_size = 32

class Bigram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass of the bigram model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = self.linear(x)    # (batch_size, sequence_length, vocab_size)
        return x