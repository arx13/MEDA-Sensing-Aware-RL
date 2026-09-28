"""Fluidic faults: act ONLY on ground-truth MicroCell.health.

Kept fully decoupled from sensing faults (which touch reported readings only).
Charge trapping is cumulative/monotonic and can progress into dielectric
breakdown, matching fault taxonomy (B): degradation -> breakdown.
"""
from __future__ import annotations


def inject_dielectric_breakdown(cell) -> None:
    """Catastrophic, permanent."""
    cell.health = 0.0
    cell.fault_type = "dielectric_breakdown"


def inject_stuck_at(cell) -> None:
    """Catastrophic, permanent."""
    cell.health = 0.0
    cell.fault_type = "stuck_at"


def apply_charge_trapping(cell, wear_increment: float = 0.01,
                          breakdown_wear_threshold: float = 1.0) -> None:
    """Parametric, cumulative degradation."""
    cell.wear += wear_increment
    cell.health = max(0.0, 1.0 - cell.wear)
    if cell.wear >= breakdown_wear_threshold:
        inject_dielectric_breakdown(cell)
    elif cell.fault_type is None:
        cell.fault_type = "degraded"
