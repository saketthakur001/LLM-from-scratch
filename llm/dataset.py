"""Turns a flat token-id stream into fixed-length (input, target) chunks for LM training."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def save_token_ids(ids: list[int], path: str) -> None:
    with open(path, "w") as f:
        json.dump(ids, f)


def load_token_ids(path: str) -> np.ndarray:
    with open(path, "r") as f:
        ids = json.load(f)
    return np.array(ids, dtype=np.int64)


class TokenBlockDataset(Dataset):
    """Each item is a (block_size,) input sequence and its next-token target,
    both sliced from a single flat array of token ids."""

    def __init__(self, ids: np.ndarray, block_size: int):
        if len(ids) <= block_size:
            raise ValueError("token stream shorter than block_size")
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.ids[idx: idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
