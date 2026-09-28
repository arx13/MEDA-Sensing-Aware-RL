"""Training-schedule continuity, best-checkpoint selection, fail-closed eval."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import yaml

from agents.baseline_dqn import make_baseline
from agents.dqn import ReplayBuffer
from evaluation.run_eval import evaluate_condition
from meda_env.fault_injection.fault_scheduler import FaultSchedule
from meda_env.gym_env import MEDARoutingEnv
from training.common import train


def _tiny_env(**kw):
    base = dict(rows=4, cols=4, max_steps=5, use_confidence=False,
                fault_profile={"n_fluidic": 0, "sensing_fault_density": 0.0},
                seed=0)
    base.update(kw)
    return MEDARoutingEnv(**base)


def test_epsilon_continuous_across_phases(tmp_path):
    """Phase B must continue the epsilon schedule, not restart at 1.0."""
    env = _tiny_env()
    agent = make_baseline(4, 4, train_start=10 ** 9, epsilon_decay_episodes=800)
    kw = dict(log_dir=str(tmp_path / "tb"), checkpoint_dir=str(tmp_path / "ckpt"),
              save_every=0, log_every=1, seed=0, eval_every=0)
    train(agent, env, 2, run_name="pha", start_episode=0, **kw)
    eps_after_a = agent.epsilon
    train(agent, env, 2, run_name="phb", start_episode=2, **kw)
    # last episode ran at global index 3: strictly decayed, never restarted
    expected = 1.0 + (3 / 800) * (0.05 - 1.0)
    assert agent.epsilon == pytest.approx(expected)
    assert agent.epsilon < eps_after_a < 1.0


def test_best_checkpoint_saved_on_greedy_eval(tmp_path):
    env = _tiny_env()
    agent = make_baseline(4, 4, train_start=10 ** 9)
    h = train(agent, env, 3, run_name="ev",
              log_dir=str(tmp_path / "tb"), checkpoint_dir=str(tmp_path / "ckpt"),
              save_every=0, log_every=1, seed=0,
              start_episode=0, eval_every=1, eval_episodes=2)
    assert len(h["eval_success"]) == 3
    assert os.path.exists(os.path.join(str(tmp_path / "ckpt"), "ev_best.pt"))


def _mini_cfgs():
    array_cfg = {"array": {"rows": 4, "cols": 4},
                 "actuation": {"health_threshold": 0.3, "wear_per_step": 0.0},
                 "env": {"max_steps": 5, "seed": 0}}
    hp = {"defaults": {"lr": 1e-3, "gamma": 0.99, "buffer_size": 100,
                       "batch_size": 8, "hidden_dim": 32, "seed": 0}}
    profile = {"n_fluidic": 0, "sensing_fault_density": 0.0}
    return array_cfg, hp, profile


def test_missing_checkpoint_fail_closed():
    array_cfg, hp, profile = _mini_cfgs()
    with pytest.raises(FileNotFoundError):
        evaluate_condition(array_cfg, profile, "nope_base.pt", "nope_aware.pt",
                           1, hp, [], "t", require_ckpt=True)


def test_missing_checkpoint_warns_and_reports_latency():
    array_cfg, hp, profile = _mini_cfgs()
    rows: list = []
    evaluate_condition(array_cfg, profile, "nope_base.pt", "nope_aware.pt",
                       1, hp, rows, "t", require_ckpt=False)
    assert len(rows) == 2  # baseline + aware random-policy rows
    for r in rows:
        assert r["decision_ms"] > 0.0


def test_buffer_clear_drops_transitions():
    buf = ReplayBuffer(10)
    o = np.zeros((4, 4, 3), dtype=np.float32)
    for _ in range(5):
        buf.add(o, 0, 0.0, o, 0.0)
    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0


def test_phase_config_pins_fixes():
    """Pin the full-budget guards: phase-2 training (cleared buffer) and
    n=100 selection probes (~+-5pp SE instead of ~+-11pp)."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..",
                            "configs", "training_baseline.yaml")
    with open(cfg_path) as f:
        hp = yaml.safe_load(f)
    assert hp["phases"].get("clear_buffer_between_phases") is True
    assert hp["phases"].get("eval_episodes") == 100


def test_shaping_rewards_approach_penalizes_retreat():
    """PBRS on reported positions: F = gamma*Phi(post) - Phi(pre), Phi = -k*d."""
    env = MEDARoutingEnv(rows=6, cols=6, max_steps=20, use_confidence=False,
                         fault_profile={"n_fluidic": 0, "sensing_fault_density": 0.0},
                         reward_shaping_scale=0.2, discount=0.99, seed=0)
    env.reset(options={"start": (0, 0), "target": (0, 2),
                       "schedule": FaultSchedule(), "skip_fluidic": True})
    _, r_fwd, _, _, _ = env.step(3)  # E: Chebyshev distance 2 -> 1
    assert r_fwd == pytest.approx(-0.05 + 0.2 * (2 - 0.99 * 1))
    _, r_back, _, _, _ = env.step(2)  # W: 1 -> 2
    assert r_back == pytest.approx(-0.05 + 0.2 * (1 - 0.99 * 2))
    assert r_fwd > 0 > r_back


def test_double_huber_train_step_finite_on_outliers():
    """Sparse +10 jackpots among -0.05s: the old MSE/max-bootstrap explosion
    trigger. Huber + Double must keep every update finite."""
    agent = make_baseline(4, 4, hidden_dim=32, train_start=0,
                          buffer_size=512, batch_size=32)
    rng = np.random.default_rng(0)
    o = rng.random((4, 4, 3)).astype(np.float32)
    for i in range(200):
        r = 10.0 if i % 20 == 0 else -0.05
        agent.remember(o, int(rng.integers(8)), r, o, 0.0)
    losses = [agent.train_step() for _ in range(20)]
    assert all(l is not None and np.isfinite(l) for l in losses)
    for p in agent.q.parameters():
        assert torch.isfinite(p).all()
