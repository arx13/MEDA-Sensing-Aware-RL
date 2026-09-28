"""Residual-based reliability (confidence) score per MC.

Idea: the droplet's *expected* next position is derivable from the last
commanded actuation; if the *reported* position deviates persistently, that
MC's sensor is suspect -- independent of electrode health. A rolling residual
average separates one-off noise from persistent spoofing/delay.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def compute_confidence(cell_id: tuple[int, int],
                       expected_position: tuple[int, int],
                       reported_position: tuple[int, int],
                       history: dict[tuple[int, int], deque],
                       window: int = 5) -> float:
    residual = euclidean(expected_position, reported_position)
    buf = history[cell_id]
    buf.append(residual)
    while len(buf) > window:
        buf.popleft()
    avg_residual = float(sum(buf) / len(buf))
    return float(1.0 / (1.0 + avg_residual))  # squash to (0, 1]


class ConfidenceTracker:
    """Maintains per-cell residual histories + a global confidence grid."""

    def __init__(self, rows: int, cols: int, window: int = 5):
        self.rows = rows
        self.cols = cols
        self.window = window
        self.history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=window))
        self.grid = np.ones((rows, cols), dtype=np.float32)

    def reset(self) -> None:
        self.history.clear()
        self.grid = np.ones((self.rows, self.cols), dtype=np.float32)

    def update(self, expected_position: tuple[int, int],
               reported_position: tuple[int, int]) -> np.ndarray:
        """Update confidence for the *reported* cell; decay others toward 1.

        Only the cell the sensor implicates gets penalised; the rest relax
        back toward full trust so stale suspicion does not accumulate.
        """
        r, c = reported_position
        r = int(np.clip(r, 0, self.rows - 1))
        c = int(np.clip(c, 0, self.cols - 1))
        conf = compute_confidence((r, c), expected_position, reported_position,
                                  self.history, self.window)
        self.grid[r, c] = conf
        # relax all other cells slightly toward 1.0
        mask = np.ones_like(self.grid, dtype=bool)
        mask[r, c] = False
        self.grid[mask] = np.clip(self.grid[mask] + 0.02, 0.0, 1.0)
        return self.grid

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()
