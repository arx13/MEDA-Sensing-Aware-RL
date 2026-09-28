"""Shared training loop: env rollout -> replay -> DQN updates + logging."""
from __future__ import annotations

import os

import numpy as np
import yaml

from meda_env.gym_env import MEDARoutingEnv
from training.callbacks import TrainLogger


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(array_cfg: dict, fault_profile: dict, use_confidence: bool,
              env_overrides: dict | None = None) -> MEDARoutingEnv:
    env_cfg = dict(array_cfg.get("env", {}))
    if env_overrides:
        env_cfg.update(env_overrides)
    return MEDARoutingEnv(
        rows=array_cfg["array"]["rows"], cols=array_cfg["array"]["cols"],
        obs_mode=env_cfg.get("obs_mode", "full"),
        local_window=env_cfg.get("local_window", 5),
        use_confidence=use_confidence,
        max_steps=env_cfg.get("max_steps", 100),
        reward_goal=env_cfg.get("reward_goal", 10.0),
        reward_step=env_cfg.get("reward_step", -0.05),
        reward_stuck=env_cfg.get("reward_stuck", -1.0),
        reward_suspect=env_cfg.get("reward_suspect", -0.1),
        confidence_threshold=env_cfg.get("confidence_threshold", 0.5),
        reward_shaping_scale=env_cfg.get("reward_shaping_scale", 0.2),
        discount=env_cfg.get("discount", 0.99),
        sensing_window=env_cfg.get("sensing_window", 5),
        health_threshold=array_cfg["actuation"].get("health_threshold", 0.3),
        wear_per_step=array_cfg["actuation"].get("wear_per_step", 0.01),
        fault_profile=fault_profile,
        seed=env_cfg.get("seed", 0),
    )


def greedy_success_rate(agent, env: MEDARoutingEnv, episodes: int, seed: int) -> float:
    """Greedy-policy success rate without training (model-selection probe)."""
    rng = np.random.default_rng(seed)
    wins = 0.0
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(agent.act(obs, greedy=True))
            done = terminated or truncated
        wins += 1.0 if env.droplet.at_target() else 0.0
    return wins / max(1, episodes)


def train(agent, env: MEDARoutingEnv, episodes: int, run_name: str = "run",
          log_dir: str = "results/tensorboard", checkpoint_dir: str = "results/checkpoints",
          save_every: int = 500, log_every: int = 10, seed: int = 0,
          start_episode: int = 0, eval_every: int = 0, eval_episodes: int = 20) -> dict:
    """Run `episodes` training episodes. `start_episode` offsets the epsilon
    schedule and all logging/checkpoint numbering so multi-phase curricula
    (clean -> faulty) share one continuous exploration schedule instead of
    restarting at eps=1.0 per phase. Every `eval_every` episodes (0 disables),
    a greedy probe is logged and the best-by-success checkpoint is saved."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = TrainLogger(log_dir, run_name)
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 777)
    history = {"episode_reward": [], "success": [], "steps": [], "loss": [],
               "eval_episode": [], "eval_success": []}
    best_eval = -1.0
    for ep in range(episodes):
        global_ep = start_episode + ep
        agent.update_epsilon(global_ep)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        ep_reward, steps, losses = 0.0, 0, []
        while not done:
            action = agent.act(obs)
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, obs2, float(terminated))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = obs2
            ep_reward += reward
            steps += 1
        success = 1.0 if env.droplet.at_target() else 0.0
        history["episode_reward"].append(ep_reward)
        history["success"].append(success)
        history["steps"].append(steps)
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        if ep % log_every == 0:
            logger.log("reward/episode", ep_reward, global_ep)
            logger.log("success/episode", success, global_ep)
            logger.log("steps/episode", steps, global_ep)
            logger.log("exploration/epsilon", agent.epsilon, global_ep)
            if losses:
                logger.log("loss/mean", float(np.mean(losses)), global_ep)
        if save_every and (global_ep + 1) % save_every == 0:
            agent.save(os.path.join(checkpoint_dir, f"{run_name}_ep{global_ep + 1}.pt"))
        if eval_every and (global_ep + 1) % eval_every == 0:
            s = greedy_success_rate(agent, env, eval_episodes,
                                    int(eval_rng.integers(1 << 30)))
            history["eval_episode"].append(global_ep)
            history["eval_success"].append(s)
            logger.log("eval/success", s, global_ep)
            if s > best_eval:
                best_eval = s
                agent.save(os.path.join(checkpoint_dir, f"{run_name}_best.pt"))
    agent.save(os.path.join(checkpoint_dir, f"{run_name}_final.pt"))
    logger.close()
    return history
