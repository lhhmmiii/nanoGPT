import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from generate import generate_text
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

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load and preprocess the dataset
    tokenizer = CharacterTokenizer()
    tokenizer.load_vocab("vocabs/char_vocab.json")
    vocab_size = len(tokenizer.char_to_index)

    # Load the text data
    with open("data/wikitext-103/wiki.valid.tokens", "r") as f:
        text = f.read()

    # Initialize the dataset and dataloader
    dataset = TextDataset(text, block_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Bigram(vocab_size, embedding_dim)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # Generate text using the trained model
    prompt = "The quick brown fox"
    generated_text = generate_text(model, tokenizer, prompt, block_size, max_length=100)
    print(f"Generated text: {generated_text}")