"""Read prepared uint32 tokens as fixed-length causal-LM training blocks."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokens(Dataset):
    def __init__(self, directory, split="train", sequence_length=None, include_tail=False, pad_token_id=None):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "ready" or split not in manifest["splits"]:
            raise ValueError("Prepared data must be ready and contain the requested split")
        self.sequence_length = sequence_length or manifest["block_length"]
        if self.sequence_length < 2:
            raise ValueError("Causal LM sequences must contain at least two tokens")
        self.tokens = np.load(directory / f"{split}_tokens.npy", mmap_mode="r")
        if self.tokens.dtype != np.uint32 or len(self.tokens) != manifest["splits"][split]["tokens"]:
            raise ValueError("Token array does not match the manifest")
        self.include_tail = include_tail
        self.pad_token_id = pad_token_id
        if include_tail and pad_token_id is None:
            raise ValueError("Including the final partial block requires a padding token")

    def __len__(self):
        return (len(self.tokens) + self.sequence_length - 1) // self.sequence_length if self.include_tail else len(self.tokens) // self.sequence_length

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        ids = torch.from_numpy(self.tokens[start:start + self.sequence_length].astype(np.int64))
        labels = ids.clone()
        if len(ids) < self.sequence_length:
            padding = self.sequence_length - len(ids)
            ids = torch.cat((ids, torch.full((padding,), self.pad_token_id, dtype=torch.int64)))
            labels = torch.cat((labels, torch.full((padding,), -100, dtype=torch.int64)))
        # Padding is strictly at the end; causal attention prevents real tokens
        # from attending to it, and ignored labels exclude padding from loss.
        return {"input_ids": ids, "labels": labels}
