"""Causal transformer for event forecasting and distress scoring."""

from __future__ import annotations

import torch
from torch import nn


class CausalEventTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        bucket_sizes: dict,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.event_embed = nn.Embedding(vocab_size, hidden_size)
        self.amount_embed = nn.Embedding(bucket_sizes["amount"], hidden_size)
        self.balance_embed = nn.Embedding(bucket_sizes["balance"], hidden_size)
        self.util_embed = nn.Embedding(bucket_sizes["util"], hidden_size)
        self.gap_embed = nn.Embedding(bucket_sizes["gap"], hidden_size)
        self.pos_embed = nn.Embedding(256, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.next_event_head = nn.Linear(hidden_size, vocab_size)
        self.distress_head = nn.Linear(hidden_size, 1)

    def forward(self, seq_tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_tokens.size(1), device=seq_tokens.device).unsqueeze(0)
        hidden = (
            self.event_embed(seq_tokens[:, :, 0])
            + self.amount_embed(seq_tokens[:, :, 1])
            + self.balance_embed(seq_tokens[:, :, 2])
            + self.util_embed(seq_tokens[:, :, 3])
            + self.gap_embed(seq_tokens[:, :, 4])
            + self.pos_embed(positions)
        )
        hidden = self.dropout(hidden)
        encoded = self.transformer(hidden, src_key_padding_mask=~mask)

        last_index = mask.sum(dim=1).clamp(min=1) - 1
        pooled = encoded[torch.arange(encoded.size(0), device=encoded.device), last_index]
        next_event_logits = self.next_event_head(pooled)
        distress_logits = self.distress_head(pooled).squeeze(-1)
        return next_event_logits, distress_logits
