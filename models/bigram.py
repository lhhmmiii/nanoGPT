import torch
import torch.nn as nn

from tokenizers.character import CharacterTokenizer

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
    # Example usage
    tokenizer = CharacterTokenizer()
    tokenizer.load_vocab("vocabs/char_vocab.json")
    vocab_size = len(tokenizer.char_to_index)
    embedding_dim = 128

    model = Bigram(vocab_size, embedding_dim)
    print(model)

    # Example input tensor
    text = "Hello, world!"
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    output = model(input_tensor)
    print(tokenizer.decode(output.argmax(dim=-1).squeeze().tolist()))  # Decode the output to text