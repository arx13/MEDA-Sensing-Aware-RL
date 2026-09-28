"""Sensing faults: act ONLY on *reported* readings, never on ground truth.

Each function is pure w.r.t. the physical array: it transforms the reported
value the agent observes. Ground-truth MicroCell.health is never touched here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def apply_sensing_noise(reported_value: float, sigma: float = 0.05,
                        rng: Optional[np.random.Generator] = None) -> float:
    noise = rng.normal(0.0, sigma) if rng is not None else np.random.normal(0.0, sigma)
    return float(np.clip(reported_value + noise, 0.0, 1.0))


def apply_sensing_spoof(reported_health: float, spoofed_value: float = 1.0) -> float:
    """Reports 'healthy' regardless of true health (false-negative sensor)."""
    return float(spoofed_value)


def apply_sensing_dropout(reported_value: float, drop_prob: float = 0.1,
                          rng: Optional[np.random.Generator] = None) -> Optional[float]:
    r = rng.random() if rng is not None else np.random.rand()
    return None if r < drop_prob else reported_value


def apply_sensing_delay(history_buffer: list, lag_steps: int = 2):
    """Return a stale reading from `lag_steps` ago (models sensor latency)."""
    if not history_buffer:
        return None
    if len(history_buffer) > lag_steps:
        return history_buffer[-lag_steps - 1] if lag_steps >= 0 else history_buffer[-1]
    return history_buffer[0]
