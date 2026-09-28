"""Array + actuation legality tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meda_env.array import MEDAArray
from meda_env.actuation import apply_move, apply_wear, ACTION_DELTAS


def test_legal_move_updates_position():
    arr = MEDAArray(5, 5)
    new_pos, stuck = apply_move((2, 2), 3, arr)  # E
    assert new_pos == (2, 3) and not stuck


def test_move_into_dead_cell_silently_fails():
    arr = MEDAArray(5, 5)
    arr.get(2, 3).health = 0.0
    new_pos, stuck = apply_move((2, 2), 3, arr)
    assert new_pos == (2, 2) and stuck


def test_out_of_bounds_silently_fails():
    arr = MEDAArray(5, 5)
    new_pos, stuck = apply_move((0, 0), 0, arr)  # N off grid
    assert new_pos == (0, 0) and stuck


def test_diagonals_available():
    arr = MEDAArray(5, 5)
    new_pos, stuck = apply_move((2, 2), 7, arr)  # SE
    assert new_pos == (3, 3) and not stuck
    assert len(ACTION_DELTAS) == 8


def test_wear_degrades_and_breaks_down():
    arr = MEDAArray(3, 3)
    for _ in range(5):
        apply_wear(arr, (1, 1), wear_increment=0.2, breakdown_wear_threshold=1.0)
    cell = arr.get(1, 1)
    assert cell.health == 0.0
    assert cell.fault_type == "dielectric_breakdown"


def test_straight_line_unobstructed():
    arr = MEDAArray(5, 5)
    pos = (0, 0)
    for _ in range(4):
        pos, stuck = apply_move(pos, 3, arr)  # E x4
        assert not stuck
    assert pos == (0, 4)
