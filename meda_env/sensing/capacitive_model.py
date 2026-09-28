"""Simulated capacitive sensing: ground truth -> reported position/health.

The sensor plane is a *read-only view* of the array. Sensing faults corrupt
only this reported plane, never MicroCell.health.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from meda_env.fault_injection import sensing_faults as sf


class CapacitiveSensorModel:
    def __init__(self, rows: int, cols: int,
                 noise_sigma: float = 0.05,
                 drop_prob: float = 0.1,
                 spoof_value: float = 1.0,
                 lag_steps: int = 2,
                 seed: int = 0):
        self.rows = rows
        self.cols = cols
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self.spoof_value = spoof_value
        self.lag_steps = lag_steps
        self.rng = np.random.default_rng(seed)
        # per-cell history of reported health (for delay faults) + last reported pos
        self.health_history: dict[tuple[int, int], list[float]] = defaultdict(list)
        self.last_reported_position: tuple[int, int] | None = None

    def reset(self, true_position: tuple[int, int] | None = None) -> None:
        self.health_history.clear()
        self.last_reported_position = true_position

    def sense(self, array, true_position: tuple[int, int],
              sensing_cells: dict[tuple[int, int], str]) -> tuple[np.ndarray, tuple[int, int]]:
        """Return (reported_health_grid, reported_position).

        reported_position is derived from the occupancy reading: with clean
        sensing it equals the true position; corrupted position sensors shift
        or freeze it, which is what the confidence tracker detects.
        All position fault modes currently hold the last reported position
        (zero-order hold); a true lag (trailing by lag_steps) is future work --
        health-delay already trails via per-cell history.
        """
        true_health = array.health_grid()
        reported = true_health.copy()

        for (r, c), kind in sensing_cells.items():
            if kind == "noise":
                reported[r, c] = sf.apply_sensing_noise(float(true_health[r, c]), self.noise_sigma, self.rng)
            elif kind == "spoof":
                reported[r, c] = sf.apply_sensing_spoof(float(true_health[r, c]), self.spoof_value)
            elif kind == "dropout":
                v = sf.apply_sensing_dropout(float(true_health[r, c]), self.drop_prob, self.rng)
                reported[r, c] = 0.0 if v is None else v  # dropout reads as "no signal"
            elif kind == "delay":
                hist = self.health_history[(r, c)]
                hist.append(float(true_health[r, c]))
                reported[r, c] = float(sf.apply_sensing_delay(hist, self.lag_steps))
            for cell_id in [(r, c)]:
                if cell_id not in self.health_history or kind != "delay":
                    self.health_history[cell_id].append(float(reported[cell_id]))

        # Reported position: corrupt if the droplet's own cell sensor is faulty.
        pos_kind = sensing_cells.get(tuple(true_position))
        if pos_kind == "spoof":
            # spoofed position sensor freezes at last reported position
            reported_pos = self.last_reported_position or true_position
        elif pos_kind == "dropout" and self.rng.random() < self.drop_prob:
            reported_pos = self.last_reported_position or true_position
        elif pos_kind == "noise":
            # jitter by one cell with prob proportional to sigma
            if self.rng.random() < min(0.5, self.noise_sigma * 5):
                dr, dc = int(self.rng.integers(-1, 2)), int(self.rng.integers(-1, 2))
                reported_pos = (int(np.clip(true_position[0] + dr, 0, self.rows - 1)),
                                int(np.clip(true_position[1] + dc, 0, self.cols - 1)))
            else:
                reported_pos = true_position
        elif pos_kind == "delay":
            reported_pos = self.last_reported_position or true_position
        else:
            reported_pos = true_position

        self.last_reported_position = reported_pos
        return reported.astype(np.float32), reported_pos
