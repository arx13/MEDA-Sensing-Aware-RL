"""Droplet state + kinematics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Droplet:
    position: tuple[int, int]       # (row, col) ground-truth MC coordinate
    volume: float = 1.0             # multiples of unit droplet volume
    target: Optional[tuple[int, int]] = None

    def at_target(self) -> bool:
        return self.target is not None and self.position == self.target
