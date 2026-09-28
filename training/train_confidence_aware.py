"""Train the confidence-aware agent (identical loop/hyperparams to baseline)."""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.confidence_aware_dqn import make_confidence_aware
from training.common import build_env, load_yaml, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="configs/array_default.yaml")
    ap.add_argument("--faults", default="configs/fault_profiles.yaml")
    ap.add_argument("--hp", default="configs/training_baseline.yaml")
    ap.add_argument("--episodes-clean", type=int, default=None)
    ap.add_argument("--episodes-faulty", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--run-name", default="aware")
    args = ap.parse_args()

    array_cfg = load_yaml(args.array)
    fault_cfg = load_yaml(args.faults)
    hp = load_yaml(args.hp)
    d = hp["defaults"]
    n_clean = args.episodes_clean if args.episodes_clean is not None else hp["phases"]["clean_episodes"]
    n_faulty = args.episodes_faulty if args.episodes_faulty is not None else hp["phases"]["faulty_episodes"]

    agent = make_confidence_aware(array_cfg["array"]["rows"], array_cfg["array"]["cols"],
                                  lr=d["lr"], gamma=d["gamma"], buffer_size=d["buffer_size"],
                                  batch_size=d["batch_size"], epsilon_start=d["epsilon_start"],
                                  epsilon_end=d["epsilon_end"],
                                  epsilon_decay_episodes=d["epsilon_decay_episodes"],
                                  target_update_every=d["target_update_every"],
                                  train_start=d["train_start"], hidden_dim=d["hidden_dim"],
                                  seed=d["seed"])

    ckpt_dir = hp["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_every = args.eval_every if args.eval_every is not None else hp["phases"].get("eval_every", 0)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else hp["phases"].get("eval_episodes", 20)

    env_clean = build_env(array_cfg, fault_cfg["profiles"]["clean"], use_confidence=True)
    print(f"[aware] phase A: clean sensing, {n_clean} episodes")
    train(agent, env_clean, n_clean, run_name=args.run_name + "_clean",
          log_dir=hp["logging"]["log_dir"], checkpoint_dir=ckpt_dir,
          save_every=hp["logging"]["save_every_episodes"], seed=d["seed"],
          start_episode=0, eval_every=eval_every, eval_episodes=eval_episodes)

    # Phase 2 is training, not fine-tuning: weights carry over, replay data
    # does not (cleared below). Epsilon schedule stays continuous.
    if hp["phases"].get("clear_buffer_between_phases", False):
        n_buf = len(agent.buffer)
        agent.buffer.clear()
        print(f"[aware] phase boundary: replay buffer cleared "
              f"({n_buf} clean-phase transitions dropped; phase 2 = training, not fine-tuning)")
    faulty_profile = fault_cfg["profiles"][hp["phases"]["faulty_profile"]]
    env_faulty = build_env(array_cfg, faulty_profile, use_confidence=True)
    print(f"[aware] phase B: {hp['phases']['faulty_profile']}, {n_faulty} episodes")
    train(agent, env_faulty, n_faulty, run_name=args.run_name + "_faulty",
          log_dir=hp["logging"]["log_dir"], checkpoint_dir=ckpt_dir,
          save_every=hp["logging"]["save_every_episodes"], seed=d["seed"] + 1,
          start_episode=n_clean, eval_every=eval_every, eval_episodes=eval_episodes)


if __name__ == "__main__":
    main()
