"""
reinforce.py

REINFORCE (Monte-Carlo policy gradient) implementation in PyTorch.

Features:
 - PolicyNet (categorical) for discrete action spaces
 - Optional baseline: ValueNet trained to predict returns (MSE)
 - Options: normalize returns, normalize advantages, entropy regularization
 - Saves per-episode returns to `results/` as .npy
 - CLI friendly

Example:
    python reinforce.py --env CartPole-v1 --episodes 1000 --seed 100 --baseline
    python reinforce.py --env CartPole-v1 --episodes 500 --seed 100 --baseline False --normalize_returns True
"""
import argparse
import os
import random
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------
# Utilities: seeding & env
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# ---------------------------
# Networks
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits  # we'll apply Categorical(logits=...) externally

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # return shape: (batch,)

# ---------------------------
# Helpers: discount returns
# ---------------------------
def compute_discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """
    Given list of rewards [r0, r1, ..., r_T], compute returns G_t = sum_{k=0}^{T-t} gamma^k r_{t+k}
    Returns numpy array shape (T+1,)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = 0.0
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns

def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)

# ---------------------------
# REINFORCE agent / trainer
# ---------------------------
def train_reinforce(
    env_name: str,
    seed: int,
    episodes: int = 1000,
    max_steps_per_episode: int = 500,
    gamma: float = 0.99,
    lr_policy: float = 1e-3,
    lr_value: float = 1e-3,
    baseline: bool = False,
    normalize_returns: bool = False,
    normalize_advantages: bool = True,
    entropy_coeff: float = 0.0,
    value_loss_coeff: float = 1.0,
    device: str = "cpu",
    save_dir: str = "results",
    hidden: int = 128,
):
    set_seed(seed)
    env = make_env(env_name, seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device(device)

    policy_net = PolicyNet(obs_dim, n_actions, hidden=hidden).to(device)
    policy_opt = optim.Adam(policy_net.parameters(), lr=lr_policy)

    value_net = None
    value_opt = None
    if baseline:
        value_net = ValueNet(obs_dim, hidden=hidden).to(device)
        value_opt = optim.Adam(value_net.parameters(), lr=lr_value)

    os.makedirs(save_dir, exist_ok=True)
    suffix = "baseline" if baseline else "nobaseline"
    results_fname = os.path.join(save_dir, f"{env_name}_reinforce_{suffix}_seed{seed}.npy")

    episode_returns = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + ep)
        states = []
        actions = []
        rewards = []

        ep_return = 0.0
        for t in range(max_steps_per_episode):
            states.append(state)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
            logits = policy_net(state_t)  # (1, n_actions)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            actions.append(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(float(reward))
            ep_return += reward

            state = next_state
            if done:
                break

        # compute discounted returns
        returns = compute_discounted_returns(rewards, gamma)  # shape (T,)
        if normalize_returns:
            returns = normalize(returns)

        # convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)  # (T, obs_dim)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)  # (T,)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)  # (T,)

        # compute baselines if any
        if baseline:
            with torch.no_grad():
                values_pred = value_net(states_t)  # (T,)
            advantages = returns_t - values_pred
        else:
            advantages = returns_t.clone()

        if normalize_advantages:
            # convert to numpy for stable normalization then back
            adv_np = advantages.cpu().numpy()
            adv_np = normalize(adv_np)
            advantages = torch.tensor(adv_np, dtype=torch.float32, device=device)

        # POLICY UPDATE
        policy_opt.zero_grad()
        logits_all = policy_net(states_t)  # (T, n_actions)
        dists = Categorical(logits=logits_all)
        log_probs = dists.log_prob(actions_t)  # (T,)
        policy_loss = -(log_probs * advantages).sum()  # negative for gradient ascent

        # entropy regularization (encourages exploration)
        if entropy_coeff and entropy_coeff > 0.0:
            entropy = dists.entropy().sum()
            policy_loss = policy_loss - entropy_coeff * entropy

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        policy_opt.step()

        # VALUE (baseline) UPDATE
        if baseline:
            value_opt.zero_grad()
            values_pred = value_net(states_t).squeeze(-1)  # (T,)
            value_loss = nn.functional.mse_loss(values_pred, returns_t) * value_loss_coeff
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
            value_opt.step()
        else:
            value_loss = None

        episode_returns.append(ep_return)

        # concise logging
        if ep % max(1, episodes // 20) == 0 or ep <= 10:
            recent20 = np.mean(episode_returns[-20:]) if len(episode_returns) >= 1 else ep_return
            if baseline:
                print(f"[Seed {seed}] Ep {ep:04d}/{episodes}  Return: {ep_return:.2f}  Recent20: {recent20:.2f}  PolicyLoss: {policy_loss.item():.3f}  ValueLoss: {value_loss.item():.3f}")
            else:
                print(f"[Seed {seed}] Ep {ep:04d}/{episodes}  Return: {ep_return:.2f}  Recent20: {recent20:.2f}  PolicyLoss: {policy_loss.item():.3f}")

    # save returns
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
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--episodes', type=int, default=1000,
                   help='Number of episodes to run (default 1000)')
    p.add_argument('--max_steps_per_episode', type=int, default=500)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr_policy', type=float, default=1e-3)
    p.add_argument('--lr_value', type=float, default=1e-3)
    p.add_argument('--baseline', action='store_true', help='Enable value baseline (ValueNet)')
    p.add_argument('--no-baseline', dest='baseline', action='store_false', help='Disable baseline explicitly')
    p.set_defaults(baseline=False)
    p.add_argument('--normalize_returns', action='store_true', help='Normalize discounted returns before using')
    p.add_argument('--normalize_advantages', action='store_true', default=True,
                   help='Normalize advantages (recommended). Default: True')
    p.add_argument('--entropy_coeff', type=float, default=0.0, help='Entropy regularization coefficient (default 0)')
    p.add_argument('--value_loss_coeff', type=float, default=1.0, help='Scale for value MSE loss')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--save_dir', type=str, default='results')
    p.add_argument('--hidden', type=int, default=128)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Running REINFORCE with args:", args)
    train_reinforce(
        env_name=args.env,
        seed=args.seed,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        gamma=args.gamma,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        baseline=args.baseline,
        normalize_returns=args.normalize_returns,
        normalize_advantages=args.normalize_advantages,
        entropy_coeff=args.entropy_coeff,
        value_loss_coeff=args.value_loss_coeff,
        device=args.device,
        save_dir=args.save_dir,
        hidden=args.hidden,
    )
