"""IDS sequence dataset for training a character-level transformer."""

import json
import os
import random
from typing import Optional

from .ids_parser import IDS_OPERATORS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class IDSVocab:
    """Token vocabulary for IDS sequences."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def build(self, sequences: list[str]) -> None:
        """Build vocab from a list of IDS strings."""
        tokens = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        # Add all IDS operators
        tokens.update(IDS_OPERATORS.keys())
        # Add all component characters
        for seq in sequences:
            for ch in seq:
                tokens.add(ch)

        for i, tok in enumerate(sorted(tokens)):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

    def encode(self, ids_string: str) -> list[int]:
        """Encode an IDS string as [BOS, ...tokens..., EOS]."""
        ids = [self.token_to_id[BOS_TOKEN]]
        for ch in ids_string:
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
        ids.append(self.token_to_id[EOS_TOKEN])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to an IDS string (strips BOS/EOS/PAD)."""
        special = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        chars = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if tok in special:
                if tok == EOS_TOKEN:
                    break
                continue
            chars.append(tok)
        return "".join(chars)

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]


def load_ids_sequences(path: Optional[str] = None) -> list[str]:
    """Load all valid IDS decomposition strings from dictionary.txt."""
    path = path or os.path.join(DATA_DIR, "dictionary.txt")
    sequences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            decomp = entry.get("decomposition", "")
            if decomp and not decomp.startswith("？") and len(decomp) >= 3:
                sequences.append(decomp)
    return sequences


def prepare_dataset(
    sequences: list[str],
    vocab: IDSVocab,
    max_len: int = 16,
    train_frac: float = 0.9,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]]]:
    """Encode sequences and split into train/val sets.

    Returns (train_encoded, val_encoded) where each is a list of
    padded token ID sequences.
    """
    encoded = []
    for seq in sequences:
        ids = vocab.encode(seq)
        if len(ids) <= max_len:
            # Pad to max_len
            ids = ids + [vocab.pad_id] * (max_len - len(ids))
            encoded.append(ids)

    random.seed(seed)
    random.shuffle(encoded)

    split = int(len(encoded) * train_frac)
    return encoded[:split], encoded[split:]
