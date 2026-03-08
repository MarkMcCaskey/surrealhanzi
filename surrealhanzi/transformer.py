"""Character-level autoregressive transformer for IDS generation.

Trainable on CPU in minutes:
- Up to ~96K training sequences from BabelStone IDS (or ~9.5K from MMAH)
- Vocab ~10K+ tokens (17 IDS operators + component characters)
- 4 layers, 4 heads, dim 192 → ~4M parameters
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDSTransformer(nn.Module):
    """Small autoregressive transformer for generating IDS sequences."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 16,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)
        h = self.drop(tok_emb + pos_emb)

        # Causal mask: each position can only attend to itself and earlier
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Padding mask
        pad_mask = (x == self.pad_id)

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    @torch.no_grad()
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        max_tokens: int = 14,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[int]:
        """Generate a single IDS sequence autoregressively."""
        self.eval()
        tokens = [bos_id]

        for _ in range(max_tokens):
            x = torch.tensor([tokens], dtype=torch.long)
            logits = self(x)
            next_logits = logits[0, -1] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            tokens.append(next_id)

            if next_id == eos_id:
                break

        return tokens

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
