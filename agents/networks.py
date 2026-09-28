"""Matched-capacity Q-networks. Only the input-channel count differs between agents."""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Small CNN over the (H, W, C) grid + MLP head. H/W agnostic via pooling."""

    def __init__(self, rows: int, cols: int, in_channels: int,
                 n_actions: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.head(self.encoder(x))


class MLPQNetwork(nn.Module):
    """Fallback flat MLP (useful for tiny grids / ablations)."""

    def __init__(self, rows: int, cols: int, in_channels: int,
                 n_actions: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols * in_channels, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
