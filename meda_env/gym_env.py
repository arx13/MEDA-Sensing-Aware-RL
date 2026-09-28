"""Gymnasium wrapper tying array + droplet + faults + sensing together."""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from meda_env.actuation import ACTION_DELTAS, apply_move, apply_wear, neighbour
from meda_env.array import MEDAArray
from meda_env.droplet import Droplet
from meda_env.fault_injection.fault_scheduler import FaultScheduler, FaultSchedule
from meda_env.sensing.capacitive_model import CapacitiveSensorModel
from meda_env.sensing.confidence_signal import ConfidenceTracker


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class MEDARoutingEnv(gym.Env):
    """Single-droplet routing on a MEDA grid under fluidic + sensing faults.

    Observation (full mode): float32 grid (H, W, C) where C = 3 baseline
      [reported_health, droplet_onehot, target_onehot] or C = 4 aware
      [reported_health, confidence, droplet_onehot, target_onehot].
    The droplet one-hot is rendered at the REPORTED position (possibly
    corrupted sensing), never ground truth; the target is a commanded goal and
    stays ground truth. Both agents see the same reported position -- the
    fairness invariant behind the baseline-vs-aware comparison.
    Local mode: same channels cropped to a window centred on the REPORTED
      droplet position (zero-padded at borders).
    Action: Discrete(8) -- 8-directional MEDA moves.

    Reproducibility / paired-episode invariant: reset(seed=int) derives
    per-episode RNGs for the fault scheduler (spawn index 0) and the sensor
    model (spawn index 1) from that seed, so a published seed fully determines
    the fault schedule and sensing-noise stream of the episode. Two envs reset
    with the same seed and fed identical actions produce identical reported
    readings step-for-step (see tests/test_sensing_aware.py::test_lockstep).
    reset(seed=None) keeps advancing the persistent streams.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 rows: int = 10, cols: int = 10,
                 obs_mode: str = "full", local_window: int = 5,
                 use_confidence: bool = True,
                 max_steps: int = 100,
                 reward_goal: float = 10.0, reward_step: float = -0.05,
                 reward_stuck: float = -1.0, reward_suspect: float = -0.1,
                 confidence_threshold: float = 0.5,
                 reward_shaping_scale: float = 0.2, discount: float = 0.99,
                 sensing_window: int = 5,
                 health_threshold: float = 0.3,
                 wear_per_step: float = 0.01,
                 fault_profile: dict | None = None,
                 fault_scheduler: FaultScheduler | None = None,
                 seed: int = 0):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.obs_mode = obs_mode
        self.local_window = local_window
        self.use_confidence = use_confidence
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_stuck = reward_stuck
        self.reward_suspect = reward_suspect
        self.confidence_threshold = confidence_threshold
        self.reward_shaping_scale = reward_shaping_scale
        self.discount = discount  # must match the agent's gamma for strict PBRS invariance
        self.sensing_window = sensing_window
        self.health_threshold = health_threshold
        self.wear_per_step = wear_per_step

        profile = fault_profile or {}
        self.scheduler = fault_scheduler or FaultScheduler(
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
            seed=profile.get("seed", seed),
        )
        self.sensors = CapacitiveSensorModel(
            rows, cols,
            noise_sigma=self.scheduler.sensing_noise_sigma,
            drop_prob=self.scheduler.sensing_drop_prob,
            spoof_value=self.scheduler.sensing_spoof_value,
            lag_steps=self.scheduler.sensing_lag_steps,
            seed=seed,
        )
        window = sensing_window  # explicit env param: observation geometry, not a fault
        self.tracker = ConfidenceTracker(rows, cols, window=window)

        self.array = MEDAArray(rows, cols)
        self.droplet = Droplet(position=(0, 0))
        self.schedule: FaultSchedule = FaultSchedule()
        self.reported_health = np.ones((rows, cols), dtype=np.float32)
        self.reported_position = (0, 0)
        self.step_count = 0
        self._rng = np.random.default_rng(seed)
        self.false_trust_moves = 0
        self.stuck_moves = 0

        self.action_space = spaces.Discrete(8)
        n_ch = 4 if use_confidence else 3
        if obs_mode == "full":
            self.observation_space = spaces.Box(0.0, 1.0, shape=(rows, cols, n_ch), dtype=np.float32)
        else:
            w = local_window
            self.observation_space = spaces.Box(0.0, 1.0, shape=(w, w, n_ch), dtype=np.float32)

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Pinned derivation order (frozen by test_spawn_order_pinned):
            #   spawn index 0 -> fault-scheduler stream, index 1 -> sensor stream.
            # A published int seed therefore fully determines the episode's fault
            # schedule AND sensing-noise realizations. seed=None keeps advancing
            # the persistent streams (SB3-style sampling).
            sched_seed, sens_seed = np.random.SeedSequence(int(seed)).spawn(2)
            self.scheduler.rng = np.random.default_rng(sched_seed)
            self.sensors.rng = np.random.default_rng(sens_seed)
        opts = options or {}
        self.array.reset()
        self.tracker.reset()
        self.step_count = 0
        self.false_trust_moves = 0
        self.stuck_moves = 0

        start = opts.get("start")
        target = opts.get("target")
        if start is None or target is None:
            start = (int(self._rng.integers(self.rows)), int(self._rng.integers(self.cols)))
            target = (int(self._rng.integers(self.rows)), int(self._rng.integers(self.cols)))
            while target == start:
                target = (int(self._rng.integers(self.rows)), int(self._rng.integers(self.cols)))

        if "schedule" in opts and opts["schedule"] is not None:
            self.schedule = opts["schedule"]
        else:
            self.schedule = self.scheduler.sample_schedule(exclude={tuple(start), tuple(target)})
        if "skip_fluidic" not in opts:
            self.scheduler.apply_fluidic(self.array, self.schedule)

        self.droplet = Droplet(position=tuple(start), target=tuple(target))
        self.array.set_occupancy(self.droplet.position)
        self.sensors.reset(self.droplet.position)
        self.reported_health, self.reported_position = self.sensors.sense(
            self.array, self.droplet.position, self.schedule.sensing_cells)
        self.tracker.grid[:, :] = 1.0
        return self._obs(), {"start": start, "target": target}

    def step(self, action: int):
        action = int(action)
        self.step_count += 1
        expected = neighbour(self.droplet.position, action)
        expected = (int(np.clip(expected[0], 0, self.rows - 1)),
                    int(np.clip(expected[1], 0, self.cols - 1)))

        new_pos, stuck = apply_move(self.droplet.position, action, self.array,
                                    self.health_threshold)
        dest = neighbour(self.droplet.position, action)
        dest_in = 0 <= dest[0] < self.rows and 0 <= dest[1] < self.cols

        # Ground-truth dead cell at destination? (out-of-bounds counts as stuck, not dead)
        dest_dead = False
        if dest_in:
            dest_dead = self.array.get(dest[0], dest[1]).health <= self.health_threshold

        # False trust: agent stepped toward a ground-truth dead cell that the
        # *reported* map claimed was healthy.
        if dest_in and dest_dead and float(self.reported_health[dest[0], dest[1]]) > self.health_threshold:
            self.false_trust_moves += 1
        if stuck:
            self.stuck_moves += 1

        # Decision-time (pre-move) reported reading of the destination, captured
        # before re-sensing below. Same reading the false-trust check uses.
        dest_reported_dead = (
            dest_in
            and float(self.reported_health[dest[0], dest[1]]) <= self.health_threshold
        )

        self.droplet.position = new_pos
        self.array.set_occupancy(new_pos)
        apply_wear(self.array, new_pos, wear_increment=self.wear_per_step)

        pre_reported = (int(self.reported_position[0]), int(self.reported_position[1]))
        # Re-sense + update confidence (expected vs reported position).
        self.reported_health, self.reported_position = self.sensors.sense(
            self.array, new_pos, self.schedule.sensing_cells)
        # Gating source is REPORTED-only (decision-time, pre-move reading),
        # never ground truth: if the sensor map itself already marks the
        # destination dead, the arrival residual carries no information about
        # the position sensor, so skip the update. Spoofed-healthy
        # destinations pass through and update.
        # Note: a dead + spoofed-healthy destination (Exp4 co-occurrence) also
        # passes: the resulting misattribution is the irreducible ambiguity
        # that makes co-occurrence the hard case. Dropout reads 0.0 ("dead"),
        # so dropout cells are skipped here -- harmless, since the agent
        # avoids them via the health channel anyway.
        if dest_reported_dead:
            conf_grid = self.tracker.get_grid()
        else:
            conf_grid = self.tracker.update(expected, self.reported_position)

        # -- rewards ------------------------------------------------------
        reward = self.reward_step
        # Potential-based shaping (Ng et al.): F = discount*Phi(post) - Phi(pre),
        # Phi = -k * Chebyshev distance. Policy-invariant, so the optimal policy
        # and the baseline-vs-aware comparison are untouched. Computed on
        # REPORTED positions -- a function of the observation, never ground
        # truth -- so shaping stalls honestly when the position sensor is blind
        # (frozen reports give F ~ 0), consistent with the paper's thesis.
        if self.reward_shaping_scale and self.droplet.target is not None:
            post_reported = (int(self.reported_position[0]), int(self.reported_position[1]))
            d_pre = _chebyshev(pre_reported, self.droplet.target)
            d_post = _chebyshev(post_reported, self.droplet.target)
            k = self.reward_shaping_scale
            reward += self.discount * (-k * d_post) - (-k * d_pre)
        terminated = False
        if self.droplet.at_target():
            reward += self.reward_goal
            terminated = True
        if stuck and dest_dead:
            reward += self.reward_stuck
        if self.use_confidence and dest_in:
            if float(conf_grid[dest[0], dest[1]]) < self.confidence_threshold:
                reward += self.reward_suspect
        truncated = self.step_count >= self.max_steps

        info = {"stuck": stuck, "dest_dead": dest_dead,
                "false_trust_moves": self.false_trust_moves}
        return self._obs(), float(reward), terminated, truncated, info

    # -- observations -----------------------------------------------------
    def _obs(self) -> np.ndarray:
        droplet_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        target_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        # The agent localizes THROUGH sensing: droplet one-hot at the reported
        # (possibly corrupted) position. Ground-truth position never enters
        # the observation (no-leak invariant, see tests/test_sensing_aware.py).
        # Target is a commanded goal and stays ground truth.
        rp = (int(self.reported_position[0]), int(self.reported_position[1]))
        droplet_map[rp[0], rp[1]] = 1.0
        if self.droplet.target is not None:
            target_map[self.droplet.target[0], self.droplet.target[1]] = 1.0
        conf = self.tracker.get_grid()
        if self.use_confidence:
            full = np.stack([self.reported_health, conf, droplet_map, target_map], axis=-1)
        else:
            full = np.stack([self.reported_health, droplet_map, target_map], axis=-1)
        if self.obs_mode == "full":
            return full.astype(np.float32)
        w = self.local_window
        pad = w // 2
        padded = np.pad(full, ((pad, pad), (pad, pad), (0, 0)))
        # Centre the window on the reported position: centring on truth would
        # reveal the true position through the window alignment.
        r, c = rp[0] + pad, rp[1] + pad
        return padded[r - pad:r + pad + 1, c - pad:c + pad + 1].astype(np.float32)

    def render(self):
        from meda_env.renderer import render_state
        return render_state(self.array, self.reported_health, self.tracker.get_grid(),
                            self.droplet)
