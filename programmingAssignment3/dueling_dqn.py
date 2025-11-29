"""
dueling_dqn.py

Complete Dueling DQN implementation (PyTorch) with:
 - dueling Q-network (variant: 'mean' or 'max' normalization)
 - replay buffer
 - epsilon-greedy policy
 - target network (hard update)
 - per-episode return logging saved to disk: results/{env}_{algo}_{variant}_seed{seed}.npy

Example:
    python dueling_dqn.py --env CartPole-v1 --variant mean --seed 100 --episodes 500
"""
import argparse
import os
import random
from collections import deque, namedtuple
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Utilities: seeding & env
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA / GPU is used and deterministic needed, additional config required (not done here)

def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# ---------------------------
# Replay buffer
# ---------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Dueling Q-Network
# ---------------------------
class DuelingQNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden=128, variant='mean'):
        """
        variant: 'mean' or 'max' (advantage normalization)
        """
        super().__init__()
        assert variant in ('mean', 'max'), "variant must be 'mean' or 'max'"
        self.variant = variant
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        # advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)       # (batch, 1)
        a = self.adv_stream(h)         # (batch, n_actions)
        if self.variant == 'mean':
            a_norm = a - a.mean(dim=1, keepdim=True)
        else:  # 'max'
            a_norm = a - a.max(dim=1, keepdim=True).values
        q = v + a_norm
        return q

# ---------------------------
# Agent & training logic
# ---------------------------
class DuelingDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        variant: str = 'mean',
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_size: int = 100_000,
        device: str = 'cpu',
        target_update_every: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_steps: int = 50_000,
    ):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.online = DuelingQNet(obs_dim, n_actions, variant=variant).to(self.device)
        self.target = DuelingQNet(obs_dim, n_actions, variant=variant).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

        self.replay = ReplayBuffer(replay_size)
        self.learn_steps = 0

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(state_v)
            action = int(q.argmax(dim=1).item())
        return action

    def compute_epsilon(self, step: int) -> float:
        if step >= self.epsilon_decay_steps:
            return self.epsilon_final
        frac = step / max(1, self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def push_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None  # not enough samples yet

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q-values
        q_values = self.online(states).gather(1, actions)  # (batch, 1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target(next_states)
            next_q_max = next_q.max(dim=1, keepdim=True).values
            q_target = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping (helpful)
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # target network hard update every N learn steps
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

# ---------------------------
# Training loop
# ---------------------------
def train(
    env_name: str,
    variant: str,
    seed: int,
    episodes: int = 500,
    max_steps_per_episode: int = 500,
    lr: float = 1e-3,
    batch_size: int = 64,
    replay_size: int = 100_000,
    start_train_after: int = 1000,
    update_every: int = 1,
    target_update_every: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
    epsilon_decay_steps: int = 50_000,
    save_dir: str = 'results',
    device: str = 'cpu',
):
    set_seed(seed)
    env = make_env(env_name, seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DuelingDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        variant=variant,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        replay_size=replay_size,
        device=device,
        target_update_every=target_update_every,
        epsilon_start=epsilon_start,
        epsilon_final=epsilon_final,
        epsilon_decay_steps=epsilon_decay_steps,
    )

    os.makedirs(save_dir, exist_ok=True)
    results_fname = os.path.join(save_dir, f"{env_name}_dueling_{variant}_seed{seed}.npy")

    total_steps = 0
    episode_returns = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + ep)  # small variation per episode seed
        ep_return = 0.0
        for t in range(max_steps_per_episode):
            epsilon = agent.compute_epsilon(total_steps)
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.push_transition(state, action, reward, next_state, float(done))
            state = next_state
            ep_return += reward
            total_steps += 1

            # training step
            if total_steps > start_train_after and (total_steps % update_every == 0):
                agent.update()

            if done:
                break

        episode_returns.append(ep_return)

        # Logging to console (concise)
        if ep % max(1, episodes // 20) == 0 or ep <= 10:
            recent = np.mean(episode_returns[-20:]) if len(episode_returns) >= 1 else ep_return
            print(f"[Seed {seed}] Ep {ep:04d}/{episodes}  Return: {ep_return:.2f}  Recent20: {recent:.2f}  Eps: {epsilon:.3f}")

    # Save returns to .npy
    np.save(results_fname, np.array(episode_returns))
    print(f"Saved returns to {results_fname}")
    env.close()
    return np.array(episode_returns)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='CartPole-v1')
    p.add_argument('--variant', type=str, choices=['mean', 'max'], default='mean',
                   help="dueling variant: 'mean' or 'max' advantage normalization")
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--replay_size', type=int, default=100_000)
    p.add_argument('--start_train_after', type=int, default=1000)
    p.add_argument('--update_every', type=int, default=1)
    p.add_argument('--target_update_every', type=int, default=1000)
    p.add_argument('--epsilon_final', type=float, default=0.05)
    p.add_argument('--epsilon_decay_steps', type=int, default=50_000)
    p.add_argument('--max_steps_per_episode', type=int, default=500)
    p.add_argument('--save_dir', type=str, default='results')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Running Dueling DQN with args:", args)
    train(
        env_name=args.env,
        variant=args.variant,
        seed=args.seed,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        start_train_after=args.start_train_after,
        update_every=args.update_every,
        target_update_every=args.target_update_every,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        save_dir=args.save_dir,
        device=args.device,
    )
