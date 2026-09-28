"""Shared DQN implementation used by both agents.

Baseline and confidence-aware agents are the SAME class with different
`in_channels` (3 vs 4) and `use_confidence` env flag. Keep hyperparameters
identical so any performance gap is attributable to the confidence signal.
"""
from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from agents.networks import QNetwork


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, o, a, r, o2, d) -> None:
        self.buf.append((o, a, r, o2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        o, a, r, o2, d = zip(*batch)
        return (np.stack(o), np.array(a), np.array(r, dtype=np.float32),
                np.stack(o2), np.array(d, dtype=np.float32))

    def clear(self) -> None:
        """Drop all stored transitions (phase boundary: weights transfer, data does not)."""
        self.buf.clear()

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    """Double DQN with target network + epsilon-greedy + Adam + Huber loss.

    Double selection (online argmax, target eval) plus Huber contains the
    max-bootstrap feedback that explodes vanilla MSE-DQN under sparse goal
    rewards. Shared verbatim by baseline and aware agents (Elfar-style)."""

    def __init__(self, rows: int, cols: int, in_channels: int, n_actions: int = 8,
                 lr: float = 1e-3, gamma: float = 0.99,
                 buffer_size: int = 20000, batch_size: int = 64,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 800,
                 target_update_every: int = 500, train_start: int = 1000,
                 hidden_dim: int = 256, seed: int = 0,
                 device: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.target_update_every = target_update_every
        self.train_start = train_start
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(rows, cols, in_channels, n_actions, hidden_dim).to(self.device)
        self.target = copy.deepcopy(self.q).eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.epsilon = epsilon_start
        self.grad_steps = 0
        self.in_channels = in_channels

    def update_epsilon(self, episode: int) -> None:
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        return int(self.q(t).argmax(dim=1).item())

    def remember(self, o, a, r, o2, d) -> None:
        self.buffer.add(o, a, r, o2, d)

    def train_step(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.train_start):
            return None
        o, a, r, o2, d = self.buffer.sample(self.batch_size)
        o = torch.as_tensor(o, dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(o2, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            # Double-DQN: online net selects, target net evaluates.
            a_star = self.q(o2).argmax(dim=1)
            q_next = self.target(o2).gather(1, a_star[:, None]).squeeze(1)
            target = r + self.gamma * (1.0 - d) * q_next
        # Huber (delta=1): quadratic near zero, linear tails -- bounds the
        # per-sample gradient that MSE lets sparse +10 rewards explode with.
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict(), "target": self.target.state_dict(),
                    "epsilon": self.epsilon, "grad_steps": self.grad_steps}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", 0.0)
        self.grad_steps = ckpt.get("grad_steps", 0)
