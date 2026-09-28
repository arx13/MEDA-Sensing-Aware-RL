"""Confidence-aware DQN: learns how much to trust the sensing map.

Observation channels: [reported_health, confidence, droplet_onehot, target_onehot].
Same network capacity class as the baseline; the agent must LEARN the trust
weighting (no hard-coded cutoff).
"""
from __future__ import annotations

from agents.dqn import DQNAgent

IN_CHANNELS = 4


def make_confidence_aware(rows: int, cols: int, **kwargs) -> DQNAgent:
    kwargs.pop("in_channels", None)
    return DQNAgent(rows, cols, in_channels=IN_CHANNELS, **kwargs)
