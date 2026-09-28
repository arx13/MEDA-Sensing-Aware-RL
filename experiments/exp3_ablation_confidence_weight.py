"""Exp3 -- ablation: zero the confidence signal at test time.

Runs the aware agent with its confidence channel zeroed out. If the gain is
really from the signal (not just a bigger observation space), performance
should fall back toward baseline.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from agents.confidence_aware_dqn import make_confidence_aware
from evaluation.metrics import summarize_episodes
from evaluation.run_eval import load_yaml, rollout
from training.common import build_env

array_cfg = load_yaml("configs/array_default.yaml")
fault_cfg = load_yaml("configs/fault_profiles.yaml")
hp = load_yaml("configs/training_baseline.yaml")
d = hp["defaults"]
rows, cols = array_cfg["array"]["rows"], array_cfg["array"]["cols"]

profile = fault_cfg["profiles"]["sensing_medium"]
env = build_env(array_cfg, profile, use_confidence=True)
agent = make_confidence_aware(rows, cols, hidden_dim=d["hidden_dim"], seed=d["seed"])
ckpt = "results/checkpoints/aware_faulty_best.pt"
if os.path.exists(ckpt):
    agent.load(ckpt)
else:
    print(f"[WARN] missing checkpoint '{ckpt}': ablating a RANDOMLY INITIALIZED policy.")

# Normal rollout
recs_normal = rollout(agent, env, hp["eval"]["episodes_per_condition"], seed=d["seed"])

# Ablated rollout: zero confidence channel (index 1) after each reset/step.
from meda_env.gym_env import MEDARoutingEnv

orig_build = MEDARoutingEnv._obs
def _obs_zeroed(self):
    o = orig_build(self)
    oc = o.copy()
    oc[..., 1] = 0.0
    return oc
MEDARoutingEnv._obs = _obs_zeroed
try:
    recs_ablated = rollout(agent, env, hp["eval"]["episodes_per_condition"], seed=d["seed"])
finally:
    MEDARoutingEnv._obs = orig_build

rows_out = []
for tag, recs in [("aware_intact", recs_normal), ("aware_conf_zeroed", recs_ablated)]:
    s = summarize_episodes(recs)
    rows_out.append({"condition": "sensing_medium", "agent": tag,
                     "success_rate": s["success_rate"], "mean_steps": s["steps"]["mean"],
                     "false_trust_rate": s["false_trust_rate"],
                     "mean_reward": s["mean_reward"], "n": s["n"]})
    print(tag, s)

os.makedirs("results", exist_ok=True)
with open("results/exp3_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("wrote results/exp3_ablation.csv")
