"""Fluidic vs sensing fault separation tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from meda_env.array import MEDAArray
from meda_env.fault_injection import fluidic_faults as ff
from meda_env.fault_injection import sensing_faults as sf
from meda_env.fault_injection.fault_scheduler import FaultScheduler


def test_fluidic_breakdown_mutates_ground_truth():
    arr = MEDAArray(4, 4)
    cell = arr.get(1, 1)
    ff.inject_dielectric_breakdown(cell)
    assert cell.health == 0.0 and cell.fault_type == "dielectric_breakdown"


def test_charge_trapping_progresses_to_breakdown():
    arr = MEDAArray(4, 4)
    cell = arr.get(0, 0)
    ff.apply_charge_trapping(cell, wear_increment=0.6, breakdown_wear_threshold=1.0)
    assert 0.0 < cell.health < 1.0
    ff.apply_charge_trapping(cell, wear_increment=0.6, breakdown_wear_threshold=1.0)
    assert cell.health == 0.0 and cell.fault_type == "dielectric_breakdown"


def test_sensing_faults_never_touch_ground_truth():
    arr = MEDAArray(4, 4)
    before = arr.health_grid().copy()
    sf.apply_sensing_noise(1.0, sigma=0.1)
    sf.apply_sensing_spoof(0.0, spoofed_value=1.0)
    sf.apply_sensing_dropout(1.0, drop_prob=0.5)
    sf.apply_sensing_delay([0.5, 0.6], lag_steps=1)
    np.testing.assert_array_equal(arr.health_grid(), before)


def test_spoof_reports_healthy():
    assert sf.apply_sensing_spoof(0.0, spoofed_value=1.0) == 1.0


def test_scheduler_overlap_parameter():
    sched = FaultScheduler(10, 10, n_fluidic=8, sensing_fault_density=0.2,
                           overlap_fraction=1.0, seed=0)
    plan = sched.sample_schedule()
    fluidic = set(plan.fluidic_cells)
    sensing = set(plan.sensing_cells)
    # full overlap: every sensing cell must be a fluidic cell
    assert sensing.issubset(fluidic)


def test_scheduler_no_overlap():
    sched = FaultScheduler(10, 10, n_fluidic=8, sensing_fault_density=0.2,
                           overlap_fraction=0.0, seed=0)
    plan = sched.sample_schedule()
    # exact-disjoint: non-overlap sensing faults never land on fluidic cells
    assert len(plan.fluidic_cells) == 8
    assert len(plan.sensing_cells) == 20
    assert set(plan.sensing_cells).isdisjoint(set(plan.fluidic_cells))


def test_exact_overlap_fraction():
    sched = FaultScheduler(10, 10, n_fluidic=8, sensing_fault_density=0.2,
                           overlap_fraction=0.25, seed=0)
    plan = sched.sample_schedule()
    assert len(plan.sensing_cells) == 20
    assert len(set(plan.sensing_cells) & set(plan.fluidic_cells)) == 5  # round(20*0.25)
