"""LSTM baseline for distress classification."""

from __future__ import annotations

import torch
from torch import nn


class LSTMDistressModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, dense_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(dense_features)
        last_index = mask.sum(dim=1).clamp(min=1) - 1
        pooled = outputs[torch.arange(outputs.size(0), device=outputs.device), last_index]
        return self.classifier(pooled).squeeze(-1)
