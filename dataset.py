import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx:idx+self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx+1:idx+1+self.block_size].astype(np.int64)
        )
        return x, y

if __name__ == "__main__":
    dataset = TextDataset("data/wikitext-103/train.bin", 8)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y = next(iter(train_loader))
    print(x)
    # print(y)