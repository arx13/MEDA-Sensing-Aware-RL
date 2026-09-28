"""Gym env: obs shapes, rewards, confidence, sensing-fault behaviour."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from meda_env.fault_injection.fault_scheduler import FaultSchedule
from meda_env.gym_env import MEDARoutingEnv


def _env(**kw):
    base = dict(rows=6, cols=6, max_steps=20, use_confidence=True,
                fault_profile={"n_fluidic": 0, "sensing_fault_density": 0.0},
                seed=0)
    base.update(kw)
    return MEDARoutingEnv(**base)


def test_obs_shapes_baseline_vs_aware():
    assert _env(use_confidence=True).reset()[0].shape == (6, 6, 4)
    assert _env(use_confidence=False).reset()[0].shape == (6, 6, 3)


def test_local_obs_shape():
    env = _env(obs_mode="local", local_window=5)
    assert env.reset()[0].shape == (5, 5, 4)


def test_goal_reward_and_termination():
    env = _env()
    env.reset(options={"start": (0, 0), "target": (0, 1),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    _, reward, terminated, _, _ = env.step(3)  # E into target
    assert terminated and reward > 5.0


def test_stuck_penalty_on_dead_cell():
    env = _env()
    obs, _ = env.reset(options={"start": (0, 0), "target": (5, 5),
                                "schedule": FaultSchedule(), "skip_fluidic": True})
    env.array.get(0, 1).health = 0.0  # kill E neighbour
    _, reward, terminated, _, _ = env.step(3)
    assert not terminated
    # stuck penalty fires; reported position frozen so shaping contributes only
    # the drift term k*d*(1-gamma) with d = Chebyshev((0,0),(5,5)) = 5.
    d = 5
    expected = (env.reward_step + env.reward_stuck
                + env.reward_shaping_scale * d * (1 - env.discount))
    assert abs(reward - expected) < 1e-6
    assert env.stuck_moves == 1


def test_confidence_drops_on_spoofed_position():
    env = _env()
    env.reset(options={"start": (2, 2), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    # spoof the DESTINATION cell sensor so the reported position freezes behind
    env.schedule.sensing_cells[(2, 3)] = "spoof"
    env.step(3)
    # confidence at the (wrong) reported cell should have dropped
    assert (env.tracker.get_grid() < 1.0).any()


def test_false_trust_counted():
    env = _env()
    env.reset(options={"start": (0, 0), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    env.array.get(0, 1).health = 0.0
    env.reported_health[0, 1] = 1.0  # sensor claims healthy
    env.step(3)
    assert env.false_trust_moves == 1
