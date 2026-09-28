"""Evaluation metrics: success, degradation, false-trust, overhead."""
from __future__ import annotations

import time

import numpy as np


def routing_success_rate(successes: list | np.ndarray) -> float:
    return float(np.mean(successes)) if len(successes) else 0.0


def degradation_curve(results_by_density: dict[float, float]) -> dict:
    """{density: success_rate} sorted by density (headline figure data)."""
    return dict(sorted(results_by_density.items()))


def false_trust_rate(false_trust_moves: int, total_moves: int) -> float:
    if total_moves <= 0:
        return 0.0
    return false_trust_moves / total_moves


def steps_to_goal(steps_list: list) -> dict:
    arr = np.array([s for s in steps_list if s is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}


def summarize_episodes(records: list[dict]) -> dict:
    successes = [r["success"] for r in records]
    return {
        "success_rate": routing_success_rate(successes),
        "n": len(records),
        "steps": steps_to_goal([r["steps"] if r["success"] else None for r in records]),
        "false_trust_rate": false_trust_rate(
            sum(r.get("false_trust_moves", 0) for r in records),
            sum(r.get("total_moves", 0) for r in records)),
        "mean_reward": float(np.mean([r["reward"] for r in records])) if records else 0.0,
    }


def time_decision(fn, repeats: int = 200) -> dict:
    """Honest overhead measurement: mean decision latency of `fn()` in ms."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - t0) / repeats
    return {"mean_ms": dt * 1000.0, "repeats": repeats}
