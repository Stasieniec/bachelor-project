import torch
from torch.utils.data import Dataset

class WikipediaDataModule(Dataset):
    def __init__(self, data, block_size = 512):
        self.data = data
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        # idx is the start position of the context window
        # x is a slice of block_size length
        x = self.data[idx : idx+self.block].long() # long() because embedding layers expect int64
        y = self.data[idx+1 : idx+self.block+1].long() # shift by 1
