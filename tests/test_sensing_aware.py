"""Sensing-awareness invariants: reported-position obs, RNG seeding, gating.

Covers the P0/P1 simulation fixes:
- droplet channel shows the REPORTED position, never ground truth (no-leak)
- reset(seed) fully determines schedule + sensing stream (spawn order pinned)
- identical reset seed + identical actions -> identical reported readings
  across baseline/aware envs (paired-episode lockstep for Exp2/Exp4)
- tracker gating is reported-only: frozen on truthfully-reported dead dest,
  decaying on spoofed-healthy dest
- dropout/delay hold the last reported position (zero-order hold)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from meda_env.fault_injection.fault_scheduler import FaultScheduler, FaultSchedule
from meda_env.gym_env import MEDARoutingEnv


def _env(**kw):
    base = dict(rows=6, cols=6, max_steps=20, use_confidence=True,
                fault_profile={"n_fluidic": 0, "sensing_fault_density": 0.0},
                seed=0)
    base.update(kw)
    return MEDARoutingEnv(**base)


def _droplet_channel(obs, use_confidence=True):
    # aware: [health, conf, droplet, target]; baseline: [health, droplet, target]
    return obs[..., -2]


def test_obs_droplet_channel_is_reported_position():
    env = _env()
    env.reset(options={"start": (2, 2), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    env.schedule.sensing_cells[(2, 3)] = "spoof"  # dest position sensor frozen
    obs, _, _, _, _ = env.step(3)  # E: true (2,3), reported frozen at (2,2)
    assert env.droplet.position == (2, 3)
    assert env.reported_position == (2, 2)
    drop = _droplet_channel(obs)
    assert drop.sum() == 1.0 and drop[2, 2] == 1.0 and drop[2, 3] == 0.0


def test_no_leak_across_fault_modes():
    for kind in ["noise", "spoof", "dropout", "delay"]:
        env = _env()
        env.reset(seed=1, options={"start": (1, 1), "target": (5, 5),
                                   "schedule": FaultSchedule(), "skip_fluidic": True})
        env.schedule.sensing_cells[(1, 2)] = kind
        for a in [3, 3, 1, 0, 2]:
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
            rp = (int(env.reported_position[0]), int(env.reported_position[1]))
            drop = _droplet_channel(obs)
            assert drop.sum() == 1.0 and drop[rp] == 1.0
            if rp != env.droplet.position:
                assert drop[env.droplet.position] == 0.0


def test_lockstep_scripted_actions_baseline_vs_aware():
    """Same reset seed + same actions -> identical reported readings.

    This is the paired-episode invariant behind the Exp2/Exp4 comparisons:
    both agents face bit-identical sensing conditions.
    """
    profile = {"n_fluidic": 4, "sensing_fault_density": 0.2, "overlap_fraction": 0.25,
               "sensing_noise_sigma": 0.08, "sensing_drop_prob": 0.2, "seed": 7}
    actions = [3, 3, 1, 7, 0, 2, 5, 4, 3, 1]
    trajs = []
    for use_conf in (False, True):
        env = MEDARoutingEnv(rows=6, cols=6, max_steps=50, use_confidence=use_conf,
                             fault_profile=profile, seed=7)
        env.reset(seed=42)
        sched = (sorted(env.schedule.fluidic_cells),
                 sorted(env.schedule.sensing_cells.items()))
        health = [env.reported_health.copy()]
        rpos = [env.reported_position]
        for a in actions:
            _, _, term, trunc, _ = env.step(a)
            health.append(env.reported_health.copy())
            rpos.append(env.reported_position)
            if term or trunc:
                break
        trajs.append((sched, health, rpos))
    (s0, h0, p0), (s1, h1, p1) = trajs
    assert s0 == s1
    assert len(h0) == len(h1) and p0 == p1
    for g0, g1 in zip(h0, h1):
        np.testing.assert_array_equal(g0, g1)


def test_spawn_order_pinned():
    """SeedSequence(seed).spawn(2): index 0 -> scheduler, 1 -> sensors.

    Uses a zero-fault profile so neither stream is consumed during reset/sense
    and both RNGs are directly comparable to fresh references.
    """
    env = _env()
    env.reset(seed=123)
    s0, s1 = np.random.SeedSequence(123).spawn(2)
    assert env.scheduler.rng.random() == np.random.default_rng(s0).random()
    assert env.sensors.rng.random() == np.random.default_rng(s1).random()


def test_reset_seed_reproducibility():
    e1, e2 = _env(), _env()
    e1.reset(seed=99)
    e2.reset(seed=99)
    assert sorted(e1.schedule.fluidic_cells) == sorted(e2.schedule.fluidic_cells)
    assert e1.schedule.sensing_cells == e2.schedule.sensing_cells
    np.testing.assert_array_equal(e1.reported_health, e2.reported_health)
    assert e1.reported_position == e2.reported_position
    # ... and reseeding an old env replays the same episode, not a continuation
    e1.reset(seed=7)
    e1.reset(seed=99)
    assert sorted(e1.schedule.fluidic_cells) == sorted(e2.schedule.fluidic_cells)
    np.testing.assert_array_equal(e1.reported_health, e2.reported_health)


def test_tracker_frozen_on_truthfully_reported_dead_dest():
    env = _env()
    env.reset(options={"start": (0, 0), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    env.array.get(0, 1).health = 0.0  # truly dead ...
    # ... but the decision-time reading is only fresh after a re-sense, so walk
    # away and back first (this also pins the gate to decision-time info).
    env.step(1)  # S to (1,0)
    env.step(0)  # N back to (0,0); reported[0,1] is now truthfully 0.0
    assert env.reported_health[0, 1] == 0.0
    before = env.tracker.get_grid().copy()
    env.step(3)  # E: stuck; reported map already explains the failure
    assert env.stuck_moves == 1
    np.testing.assert_array_equal(env.tracker.get_grid(), before)


def test_tracker_decays_on_spoofed_healthy_dest():
    env = _env()
    env.reset(options={"start": (2, 2), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    env.array.get(2, 3).health = 0.0            # truly dead ...
    env.schedule.sensing_cells[(2, 3)] = "spoof"  # ... but reported healthy
    env.step(3)
    assert env.false_trust_moves == 1
    assert (env.tracker.get_grid() < 1.0).any()  # gate passes, update proceeds


def test_dropout_holds_last_reported_position():
    env = _env()
    env.reset(options={"start": (1, 1), "target": (5, 5),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    env.schedule.sensing_cells[(1, 2)] = "dropout"
    env.sensors.drop_prob = 1.0  # force the dropout branch
    env.step(3)  # true (1,2); position reading drops -> zero-order hold
    assert env.droplet.position == (1, 2)
    assert env.reported_position == (1, 1)
    assert env.reported_health[1, 2] == 0.0  # dropout reads as "no signal"


def test_sensing_window_param_reaches_tracker():
    env = MEDARoutingEnv(rows=6, cols=6, sensing_window=3,
                         fault_profile={"n_fluidic": 0, "sensing_fault_density": 0.0})
    assert env.tracker.window == 3
