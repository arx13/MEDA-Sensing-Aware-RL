"""Grid of microelectrode (MC) cells with ground-truth health state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MicroCell:
    row: int
    col: int
    health: float = 1.0            # 1.0 = fully healthy, 0.0 = dead (ground truth)
    fault_type: Optional[str] = None  # None | 'dielectric_breakdown' | 'stuck_at' | 'degraded'
    wear: float = 0.0              # cumulative actuation count -> charge-trapping model
    true_droplet_present: bool = False


class MEDAArray:
    """2D grid of MicroCells with vectorised health/wear accessors.

    The object-of-record for *ground truth*. Sensing code may read it but must
    never mutate it; sensing faults live in a separate reported-reading plane.
    """

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.cells = [[MicroCell(r, c) for c in range(cols)] for r in range(rows)]

    # -- basic helpers ----------------------------------------------------
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get(self, row: int, col: int) -> MicroCell:
        return self.cells[row][col]

    def reset(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                cell.health = 1.0
                cell.fault_type = None
                cell.wear = 0.0
                cell.true_droplet_present = False

    # -- vectorised views -------------------------------------------------
    def health_grid(self) -> np.ndarray:
        return np.array([[c.health for c in row] for row in self.cells], dtype=np.float32)

    def wear_grid(self) -> np.ndarray:
        return np.array([[c.wear for c in row] for row in self.cells], dtype=np.float32)

    def occupancy_grid(self) -> np.ndarray:
        return np.array(
            [[1.0 if c.true_droplet_present else 0.0 for c in row] for row in self.cells],
            dtype=np.float32,
        )

    def set_occupancy(self, position: tuple[int, int] | None) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c].true_droplet_present = False
        if position is not None:
            self.cells[position[0]][position[1]].true_droplet_present = True

    def dead_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Ground-truth dead cells (actuation force insufficient)."""
        return self.health_grid() <= threshold

    def random_free_cell(self, rng: np.random.Generator, exclude: set | None = None) -> tuple[int, int]:
        exclude = exclude or set()
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in exclude and self.cells[r][c].health > 0.3
        ]
        if not candidates:
            candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in exclude]
        return candidates[int(rng.integers(len(candidates)))]
