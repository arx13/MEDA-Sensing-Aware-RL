"""Baseline DQN: trusts the sensing map fully (Elfar-style assumption).

Observation channels: [reported_health, droplet_onehot, target_onehot].
Identical architecture/hyperparams to the confidence-aware agent; the only
difference is the ABSENCE of the confidence channel.
"""
from __future__ import annotations

from agents.dqn import DQNAgent

IN_CHANNELS = 3


def make_baseline(rows: int, cols: int, **kwargs) -> DQNAgent:
    kwargs.pop("in_channels", None)
    return DQNAgent(rows, cols, in_channels=IN_CHANNELS, **kwargs)
