"""Per-episode fault scheduling: when/where each fault type activates.

Fluidic dead cells and corrupted-sensor cells are sampled *independently*;
the overlap between the two sets is an explicit experiment parameter (the
regime the trust-the-sensing baseline handles worst).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from meda_env.fault_injection import fluidic_faults as ff


@dataclass
class FaultSchedule:
    fluidic_cells: list[tuple[int, int]] = field(default_factory=list)
    sensing_cells: dict[tuple[int, int], str] = field(default_factory=dict)  # cell -> fault kind
    fluidic_types: dict[tuple[int, int], str] = field(default_factory=dict)


class FaultScheduler:
    def __init__(self, rows: int, cols: int,
                 n_fluidic: int = 5,
                 fluidic_mix: dict | None = None,
                 sensing_fault_density: float = 0.1,
                 sensing_mix: dict | None = None,
                 overlap_fraction: float = 0.2,
                 sensing_noise_sigma: float = 0.05,
                 sensing_drop_prob: float = 0.1,
                 sensing_spoof_value: float = 1.0,
                 sensing_lag_steps: int = 2,
                 seed: int = 0):
        self.rows = rows
        self.cols = cols
        self.n_fluidic = n_fluidic
        self.fluidic_mix = fluidic_mix or {"dielectric_breakdown": 0.5, "stuck_at": 0.3, "charge_trapping": 0.2}
        self.sensing_fault_density = sensing_fault_density
        self.sensing_mix = sensing_mix or {"noise": 0.5, "spoof": 0.2, "dropout": 0.2, "delay": 0.1}
        self.overlap_fraction = overlap_fraction
        self.sensing_noise_sigma = sensing_noise_sigma
        self.sensing_drop_prob = sensing_drop_prob
        self.sensing_spoof_value = sensing_spoof_value
        self.sensing_lag_steps = sensing_lag_steps
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_profile(cls, rows: int, cols: int, profile: dict, seed: int | None = None) -> "FaultScheduler":
        return cls(
            rows, cols,
            n_fluidic=profile.get("n_fluidic", 5),
            fluidic_mix=profile.get("fluidic_mix"),
            sensing_fault_density=profile.get("sensing_fault_density", 0.1),
            sensing_mix=profile.get("sensing_mix"),
            overlap_fraction=profile.get("overlap_fraction", 0.2),
            sensing_noise_sigma=profile.get("sensing_noise_sigma", 0.05),
            sensing_drop_prob=profile.get("sensing_drop_prob", 0.1),
            sensing_spoof_value=profile.get("sensing_spoof_value", 1.0),
            sensing_lag_steps=profile.get("sensing_lag_steps", 2),
            seed=profile.get("seed", 0) if seed is None else seed,
        )

    # -- sampling ---------------------------------------------------------
    def _sample_kind(self, mix: dict) -> str:
        kinds = list(mix.keys())
        probs = np.array([mix[k] for k in kinds], dtype=float)
        probs /= probs.sum()
        return kinds[int(self.rng.choice(len(kinds), p=probs))]

    def sample_schedule(self, exclude: set[tuple[int, int]] | None = None) -> FaultSchedule:
        """Sample fluidic + sensing fault sets. `exclude` = start/target cells."""
        exclude = set(exclude or set())
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if (r, c) not in exclude]
        sched = FaultSchedule()

        # Fluidic faults
        n_f = min(self.n_fluidic, len(all_cells))
        fluidic: list[tuple[int, int]] = []
        if all_cells and n_f > 0:
            idx = self.rng.choice(len(all_cells), size=n_f, replace=False)
            fluidic = [all_cells[i] for i in np.atleast_1d(idx)]
        for cell in fluidic:
            kind = self._sample_kind(self.fluidic_mix)
            sched.fluidic_cells.append(cell)
            sched.fluidic_types[cell] = kind

        # Sensing faults: total count from density; a fraction forced to overlap fluidic set
        n_s = int(round(self.rows * self.cols * self.sensing_fault_density))
        if self.overlap_fraction >= 1.0:
            # full co-occurrence: sensing set is a subset of the fluidic set
            n_s = min(n_s, len(sched.fluidic_cells))
        n_overlap = int(round(n_s * self.overlap_fraction))
        n_overlap = min(n_overlap, len(sched.fluidic_cells))
        overlap_cells = [c for c in sched.fluidic_cells if c not in exclude]
        if overlap_cells:
            idx = self.rng.choice(len(overlap_cells), size=min(n_overlap, len(overlap_cells)), replace=False)
            chosen_overlap = [overlap_cells[i] for i in np.atleast_1d(idx)]
        else:
            chosen_overlap = []
        # Non-overlap sensing faults are drawn from cells that are NEITHER
        # already-chosen overlap NOR fluidic, so overlap_fraction is exact
        # (up to pool exhaustion, guarded by min() below) rather than a lower
        # bound. Exactness is load-bearing for Exp4 co-occurrence analysis.
        fluidic_set = set(sched.fluidic_cells)
        remaining_pool = [c for c in all_cells
                          if c not in set(chosen_overlap) and c not in fluidic_set]
        n_rest = n_s - len(chosen_overlap)
        rest = []
        if n_rest > 0 and remaining_pool:
            idx = self.rng.choice(len(remaining_pool), size=min(n_rest, len(remaining_pool)), replace=False)
            rest = [remaining_pool[i] for i in np.atleast_1d(idx)]
        for cell in list(chosen_overlap) + list(rest):
            if cell in exclude:
                continue
            sched.sensing_cells[tuple(cell)] = self._sample_kind(self.sensing_mix)
        return sched

    def apply_fluidic(self, array, schedule: FaultSchedule,
                      wear_increment: float = 0.3) -> None:
        """Stamp fluidic faults onto ground-truth cells."""
        for cell_id in schedule.fluidic_cells:
            cell = array.get(cell_id[0], cell_id[1])
            kind = schedule.fluidic_types.get(cell_id, "dielectric_breakdown")
            if kind == "dielectric_breakdown":
                ff.inject_dielectric_breakdown(cell)
            elif kind == "stuck_at":
                ff.inject_stuck_at(cell)
            else:  # charge_trapping / degraded: parametric hit
                ff.apply_charge_trapping(cell, wear_increment=wear_increment,
                                         breakdown_wear_threshold=1.0)
