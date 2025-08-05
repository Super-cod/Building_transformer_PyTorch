import torch
from torch.utils.data import Dataset
from vocabulary import Vocabulary


class WordLevelDataset(Dataset):
    def __init__(self, text_path, vocab: Vocabulary, block_size: int):
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.vocab = vocab
        self.block_size = block_size

        text = " ".join(lines)
        self.tokens = vocab.numericalize(text)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1: idx + self.block_size + 1], dtype=torch.long)
        return x, y
