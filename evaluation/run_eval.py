"""Greedy-policy rollout evaluation for both agents across fault conditions."""
from __future__ import annotations

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from agents.baseline_dqn import make_baseline
from agents.confidence_aware_dqn import make_confidence_aware
from evaluation.metrics import summarize_episodes, time_decision
from training.common import build_env


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rollout(agent, env, episodes: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        total_reward, moves = 0.0, 0
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            moves += 1
        records.append({
            "success": 1.0 if env.droplet.at_target() else 0.0,
            "steps": moves,
            "reward": total_reward,
            "false_trust_moves": env.false_trust_moves,
            "total_moves": moves,
            "stuck_moves": env.stuck_moves,
        })
    return records


def measure_latency(agent, env, repeats: int = 200, seed: int = 0,
                    warmup: int = 20) -> float:
    """Honest overhead metric: mean greedy decision latency in ms.

    Times forward passes over real observations from the env so the reported
    cost includes the confidence channel's extra input width for the aware net.
    Untimed warm-up calls run first so oneDNN/MKL init and cold caches (which
    would otherwise land on whichever agent is measured first) don't bias the
    comparison. Same process, back-to-back; report hardware alongside numbers
    since absolutes are machine-specific (the aware/baseline ratio travels).
    """
    rng = np.random.default_rng(seed)
    obs_pool = [env.reset(seed=int(rng.integers(1 << 30)))[0] for _ in range(10)]
    state = {"i": 0}

    def _fn():
        agent.act(obs_pool[state["i"] % len(obs_pool)], greedy=True)
        state["i"] += 1

    for _ in range(warmup):
        _fn()
    return time_decision(_fn, repeats)["mean_ms"]


def _load_or_fail(agent, ckpt: str | None, name: str, require_ckpt: bool) -> None:
    if ckpt and os.path.exists(ckpt):
        agent.load(ckpt)
        return
    msg = (f"[WARN] missing checkpoint '{ckpt}': evaluating {name} agent with "
           f"RANDOM weights -- numbers are not a trained result.")
    print(msg)
    if require_ckpt:
        raise FileNotFoundError(msg)


def evaluate_condition(array_cfg, fault_profile, ckpt_baseline: str | None,
                       ckpt_aware: str | None, episodes: int, hp: dict, out_rows: list,
                       tag: str, require_ckpt: bool = False) -> None:
    d = hp["defaults"]
    rows, cols = array_cfg["array"]["rows"], array_cfg["array"]["cols"]
    env_b = build_env(array_cfg, fault_profile, use_confidence=False)
    env_a = build_env(array_cfg, fault_profile, use_confidence=True)
    agent_b = make_baseline(rows, cols, lr=d["lr"], gamma=d["gamma"],
                            buffer_size=d["buffer_size"], batch_size=d["batch_size"],
                            hidden_dim=d["hidden_dim"], seed=d["seed"])
    agent_a = make_confidence_aware(rows, cols, lr=d["lr"], gamma=d["gamma"],
                                    buffer_size=d["buffer_size"], batch_size=d["batch_size"],
                                    hidden_dim=d["hidden_dim"], seed=d["seed"])
    _load_or_fail(agent_b, ckpt_baseline, "baseline", require_ckpt)
    _load_or_fail(agent_a, ckpt_aware, "aware", require_ckpt)
    for name, agent, env in [("baseline", agent_b, env_b), ("aware", agent_a, env_a)]:
        recs = rollout(agent, env, episodes, seed=hp["defaults"]["seed"])
        s = summarize_episodes(recs)
        ms = measure_latency(agent, env)
        out_rows.append({"condition": tag, "agent": name,
                         "success_rate": s["success_rate"],
                         "mean_steps": s["steps"]["mean"],
                         "false_trust_rate": s["false_trust_rate"],
                         "mean_reward": s["mean_reward"],
                         "decision_ms": ms, "n": s["n"]})
        print(f"[{tag}] {name}: success={s['success_rate']:.3f} "
              f"false_trust={s['false_trust_rate']:.3f} steps={s['steps']['mean']:.1f} "
              f"decision_ms={ms:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="configs/array_default.yaml")
    ap.add_argument("--faults", default="configs/fault_profiles.yaml")
    ap.add_argument("--hp", default="configs/training_baseline.yaml")
    ap.add_argument("--ckpt-baseline", default="results/checkpoints/baseline_faulty_best.pt")
    ap.add_argument("--ckpt-aware", default="results/checkpoints/aware_faulty_best.pt")
    ap.add_argument("--require-ckpt", action="store_true",
                    help="fail instead of evaluating random weights when a checkpoint is missing")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default="results/eval_summary.csv")
    args = ap.parse_args()

    array_cfg = load_yaml(args.array)
    fault_cfg = load_yaml(args.faults)
    hp = load_yaml(args.hp)
    episodes = args.episodes or hp["eval"]["episodes_per_condition"]
    rows_out: list[dict] = []
    for tag, profile in fault_cfg["profiles"].items():
        evaluate_condition(array_cfg, profile, args.ckpt_baseline, args.ckpt_aware,
                           episodes, hp, rows_out, tag, require_ckpt=args.require_ckpt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
