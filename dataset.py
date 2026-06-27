import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers.character import CharacterTokenizer

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.load_vocab("vocabs/char_vocab.json")
        self.tokens = torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.text) - self.block_size 

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y

if __name__ == "__main__":
    with open("data/wikitext-103/wiki.valid.tokens", "r") as f:
        text = f.read()

    # Initialize the dataset and dataloader
    dataset = TextDataset(text, 8)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y = next(iter(train_loader))
    print(x)
    # print(y)