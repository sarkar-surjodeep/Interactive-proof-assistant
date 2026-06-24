

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Tuple, List


# ── Network ───────────────────────────────────────────────────────────────────

class WorkerNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(acts),
            torch.FloatTensor(rews),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ── Worker agent ──────────────────────────────────────────────────────────────

class WorkerAgent:
    """
    DQN worker that learns to apply inference rules.
    Epsilon-greedy exploration decays over training.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        target_update_freq: int = 50,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.steps_done = 0

        self.policy_net = WorkerNet(obs_dim, n_actions)
        self.target_net = WorkerNet(obs_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_history: List[float] = []

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
              np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if not greedy and random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(obs).unsqueeze(0))
            return int(q.argmax().item())

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size)

        # Current Q values
        q_vals = self.policy_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_obs).argmax(1)
            next_q = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rews + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.steps_done = ckpt["steps_done"]
        self.epsilon = ckpt["epsilon"]
