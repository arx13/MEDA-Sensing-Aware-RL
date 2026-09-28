"""Legal moves + voltage/wear model for MEDA free-direction routing.

8-directional moves (N/S/E/W + diagonals) are the point of MEDA over EWOD;
dropping diagonals would forfeit the platform's defining capability.
"""
from __future__ import annotations

# Action indexing: 0=N(-1,0) 1=S(+1,0) 2=W(0,-1) 3=E(0,+1)
#                  4=NW(-1,-1) 5=NE(-1,+1) 6=SW(+1,-1) 7=SE(+1,+1)
ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

ACTION_NAMES = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")


def neighbour(position: tuple[int, int], action: int) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    return (position[0] + dr, position[1] + dc)


def is_in_bounds(position: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= position[0] < rows and 0 <= position[1] < cols


def is_actuable(true_health: float, threshold: float = 0.3) -> bool:
    """Whether the destination MC can physically pull the droplet."""
    return true_health > threshold


def apply_move(position: tuple[int, int], action: int, array, threshold: float = 0.3):
    """Attempt a move. Returns (new_position, stuck_flag).

    A move into a ground-truth dead cell *silently fails* (droplet stays put),
    modelling insufficient EWOD force -- not a crash. Out-of-bounds moves also
    fail silently.
    """
    dest = neighbour(position, action)
    if not is_in_bounds(dest, array.rows, array.cols):
        return position, True
    if not is_actuable(array.get(dest[0], dest[1]).health, threshold):
        return position, True
    return dest, False


def apply_wear(array, position: tuple[int, int], wear_increment: float = 0.01,
               breakdown_wear_threshold: float = 1.0) -> None:
    """Charge-trapping wear on the cell a droplet actuates/passes over.

    Cumulative + monotonic; crossing the wear threshold progresses into
    dielectric breakdown (degradation -> breakdown, not independent classes).
    """
    cell = array.get(position[0], position[1])
    cell.wear += wear_increment
    cell.health = max(0.0, 1.0 - cell.wear)
    if cell.wear >= breakdown_wear_threshold:
        cell.health = 0.0
        cell.fault_type = "dielectric_breakdown"
    elif cell.wear > 0.0 and cell.fault_type is None:
        cell.fault_type = "degraded"
