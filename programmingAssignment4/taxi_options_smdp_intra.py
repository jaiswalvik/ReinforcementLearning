#!/usr/bin/env python3
"""
taxi_options_smdp_intra_fixed.py

SMDP Q-learning and Intra-option Q-learning on Taxi-v3 (fixed BFS & improvements).

Usage:
  python taxi_options_smdp_intra_fixed.py --n_episodes 2000 --epsilon 1.0 --epsilon_decay 0.99995

Dependencies:
  - gymnasium (or gym)
  - numpy
  - matplotlib
"""
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os
import time
import sys
from collections import deque

# Try gymnasium first, fallback to gym
try:
    import gymnasium as gym  # type: ignore
except Exception:
    import gym  # type: ignore

# ---------------------------
# Helper: map taxi env landmark idx to coordinates
LANDMARKS = {
    0: ("R", (0, 0)),
    1: ("G", (0, 4)),
    2: ("Y", (4, 0)),
    3: ("B", (4, 3)),
}

ACTION_MEANS = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff"
}

# --- API compatibility wrappers ---------------------------------------------
def reset_env(env):
    """
    Returns: integer observation/state
    Handles both gym (obs) and gymnasium (obs, info)
    """
    result = env.reset()
    if isinstance(result, tuple) and len(result) >= 1:
        obs = result[0]
    else:
        obs = result
    return obs

def step_env(env, action):
    """
    Step wrapper that returns (obs, reward, done, info)
    Handles both gym (obs, r, done, info) and gymnasium (obs, r, terminated, truncated, info)
    """
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done, info
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    else:
        raise RuntimeError(f"Unknown env.step() return signature of length {len(result)}")

def decode_state(env, s):
    """
    Robust decoder for Taxi states.
    Returns a tuple: (taxi_row, taxi_col, pass_loc, dest_idx)
    Tries multiple decode locations (env.decode, env.unwrapped.decode, env.env.decode).
    """
    s_int = int(s)
    calls = []
    if hasattr(env, "decode"):
        calls.append(lambda: env.decode(s_int))
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "decode"):
        calls.append(lambda: env.unwrapped.decode(s_int))
    if hasattr(env, "env") and hasattr(env.env, "decode"):
        calls.append(lambda: env.env.decode(s_int))

    last_exc = None
    for call in calls:
        try:
            val = call()
            # normalize val to tuple
            if hasattr(val, "__iter__") and not isinstance(val, (tuple, list)):
                val = tuple(val)
            if isinstance(val, (tuple, list)) and len(val) == 4:
                return tuple(val)
            raise ValueError(f"decode returned unexpected type/shape: {type(val)} / {val}")
        except Exception as e:
            last_exc = e
            continue

    raise AttributeError(f"Could not find a working `decode`. Last error: {last_exc}")

# ---------------------------
def manhattan_action_towards(taxi_pos, target_pos):
    tr, tc = target_pos
    r, c = taxi_pos
    if r < tr:
        return 0  # south
    if r > tr:
        return 1  # north
    if c < tc:
        return 2  # east
    if c > tc:
        return 3  # west
    return None

def next_nav_action_bfs(env, state, target_coord):
    """
    BFS on discrete state graph (using env.P) to find a shortest-path first action (0..3)
    from `state` to any state where taxi position == target_coord.
    Returns None if already at target or if no path found.
    Falls back to manhattan if env.P or decode is unavailable.
    """
    # Try to decode taxi pos (if this fails we will fall back to manhattan with (0,0))
    try:
        taxi_r, taxi_c, _, _ = decode_state(env, state)
    except Exception:
        # fallback: can't decode, use manhattan from (0,0) conservatively
        return manhattan_action_towards((0, 0), target_coord)

    if (taxi_r, taxi_c) == target_coord:
        return None

    if not hasattr(env, "P"):
        return manhattan_action_towards((taxi_r, taxi_c), target_coord)

    start = int(state)
    parents = {}  # node -> (parent_node, action_from_parent_to_node)
    q = deque()
    visited = set([start])

    # enqueue neighbors of start
    for a in (0, 1, 2, 3):
        trans = env.P[start].get(a) if isinstance(env.P[start], dict) else env.P[start][a]
        if not trans:
            continue
        nextstate = trans[0][1]
        if nextstate is None or nextstate in visited:
            continue
        visited.add(nextstate)
        parents[nextstate] = (start, a)
        q.append(nextstate)

    while q:
        cur = q.popleft()
        try:
            tr, tc, _, _ = decode_state(env, cur)
        except Exception:
            # If we can't decode an explored state, bail to manhattan fallback
            return manhattan_action_towards((taxi_r, taxi_c), target_coord)
        if (tr, tc) == target_coord:
            # walk up parents to find first action from start
            cur_node = cur
            parent, action = parents[cur_node]
            # If parent is start, return action, otherwise climb
            while parent != start:
                cur_node = parent
                parent, action = parents[cur_node]
            return action
        for a in (0, 1, 2, 3):
            trans = env.P[cur].get(a) if isinstance(env.P[cur], dict) else env.P[cur][a]
            if not trans:
                continue
            ns = trans[0][1]
            if ns is None or ns in parents or ns == start:
                continue
            # set parent to current & the action used from cur->ns (not inherited)
            parents[ns] = (cur, a)
            q.append(ns)
    # no path found; fall back to simple manhattan direction
    return manhattan_action_towards((taxi_r, taxi_c), target_coord)

def option_policy_action(env, state, option_id):
    """
    Navigation toward the option's landmark using BFS. At landmark, do pickup/dropoff as appropriate.
    """
    taxi_r, taxi_c, pass_loc, dest_idx = decode_state(env, state)
    landmark_coord = LANDMARKS[option_id][1]

    if (taxi_r, taxi_c) == landmark_coord:
        # if passenger present and matches option landmark, pickup
        if pass_loc != 4 and pass_loc == option_id:
            return 4
        # if passenger in taxi and destination is this landmark, dropoff
        if pass_loc == 4 and dest_idx == option_id:
            return 5
        return None

    a = next_nav_action_bfs(env, state, landmark_coord)
    if a is None:
        # BFS said we are at the landmark or couldn't find meaningful action; try manhattan
        return manhattan_action_towards((taxi_r, taxi_c), landmark_coord)
    return a

def option_terminates(env, state, option_id):
    taxi_r, taxi_c, _, _ = decode_state(env, state)
    landmark_coord = LANDMARKS[option_id][1]
    return (taxi_r, taxi_c) == landmark_coord

# ---------------------------
def epsilon_greedy(Q_row, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(Q_row))
    else:
        Q_row = np.asarray(Q_row)
        max_val = Q_row.max()
        candidates = np.flatnonzero(Q_row == max_val)
        return int(random.choice(candidates))

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# ---------------------------
class SMDP_QLearner:
    def __init__(self, env, n_options=4, alpha=0.1, gamma=0.9, epsilon=1.0,
                 option_step_cap=200, enable_debug=False, optimistic_init=0.1):
        self.env = env
        self.n_options = n_options
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Ensure discrete state space
        if not hasattr(env.observation_space, "n"):
            raise ValueError("Environment must have a discrete observation_space with .n")
        self.Q = np.ones((env.observation_space.n, n_options), dtype=np.float32) * optimistic_init
        self.option_step_cap = option_step_cap
        self.debug = enable_debug

        # diagnostics
        self.fallback_count = 0
        self.post_pick_choice_counts = np.zeros(n_options, dtype=int)

    def run_episode(self, max_steps_per_episode=1000, render=False):
        s = reset_env(self.env)
        total_reward = 0.0
        steps = 0
        done = False
        delivered = 0

        while not done and steps < max_steps_per_episode:
            o = epsilon_greedy(self.Q[int(s)], self.epsilon)

            cum_reward = 0.0
            k = 0
            s0 = s
            prim_steps = 0
            saw_pickup_in_option = False

            while True:
                a = option_policy_action(self.env, s, o)
                if a is None:
                    break

                s_next, r, done, info = step_env(self.env, a)
                if r == 20:
                    delivered = 1

                # blocked nav fallback (should be rarely needed with BFS)
                if a in (0,1,2,3) and s_next == s:
                    # fallback primitive random move
                    fallback = random.choice([0,1,2,3])
                    s_fb, r_fb, done_fb, info_fb = step_env(self.env, fallback)
                    self.fallback_count += 1
                    # add current step reward and fallback reward (discounted appropriately)
                    cum_reward += (self.gamma ** k) * r
                    k += 1
                    steps += 1
                    total_reward += r
                    cum_reward += (self.gamma ** k) * r_fb
                    k += 1
                    steps += 1
                    total_reward += r_fb
                    if r_fb == 20:
                        delivered = 1
                    s = s_fb
                else:
                    if a == 4:
                        saw_pickup_in_option = True
                    cum_reward += (self.gamma ** k) * r
                    k += 1
                    steps += 1
                    total_reward += r
                    s = s_next

                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

                prim_steps += 1
                if option_terminates(self.env, s, o) or done:
                    break
                if prim_steps >= self.option_step_cap:
                    break

            if saw_pickup_in_option:
                post_choice = epsilon_greedy(self.Q[int(s)], self.epsilon)
                self.post_pick_choice_counts[post_choice] += 1

            target = cum_reward
            if not done:
                target += (self.gamma ** k) * np.max(self.Q[int(s)])
            td_error = target - self.Q[int(s0), o]
            self.Q[int(s0), o] += self.alpha * td_error

        return total_reward, delivered, steps

    def train(self, n_episodes=5000, epsilon_decay=0.99995, min_epsilon=0.1, max_steps=1000, block_size=100):
        rewards = []
        block_delivery_counts = 0
        for ep in range(1, n_episodes + 1):
            r, delivered, steps = self.run_episode(max_steps_per_episode=max_steps)
            rewards.append(r)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)
            block_delivery_counts += delivered

            if ep % block_size == 0:
                print(f"[SMDP] Episode {ep}/{n_episodes}, reward={r:.1f}, eps={self.epsilon:.4f}, deliveries_last{block_size}={block_delivery_counts}, fallbacks_last{block_size}={self.fallback_count}")
                print(f"        post-pick choice counts (last block): {self.post_pick_choice_counts}")
                block_delivery_counts = 0
                self.fallback_count = 0
                self.post_pick_choice_counts[:] = 0

        return rewards

# ---------------------------
class IntraOptionQLearner:
    def __init__(self, env, n_options=4, alpha=0.1, gamma=0.9, epsilon=1.0,
                 option_step_cap=200, enable_debug=False, optimistic_init=0.1):
        self.env = env
        self.n_options = n_options
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        if not hasattr(env.observation_space, "n"):
            raise ValueError("Environment must have a discrete observation_space with .n")
        self.Q = np.ones((env.observation_space.n, n_options), dtype=np.float32) * optimistic_init
        self.option_step_cap = option_step_cap
        self.debug = enable_debug

        # diagnostics
        self.fallback_count = 0
        self.post_pick_choice_counts = np.zeros(n_options, dtype=int)

    def run_episode(self, max_steps_per_episode=2000, render=False):
        s = reset_env(self.env)
        total_reward = 0.0
        steps = 0
        done = False
        delivered = 0

        while not done and steps < max_steps_per_episode:
            o_selected = epsilon_greedy(self.Q[int(s)], self.epsilon)

            s0 = s
            cum_reward = 0.0
            k = 0
            prim_steps = 0
            saw_pickup_in_option = False

            while True:
                a = option_policy_action(self.env, s, o_selected)
                if a is None:
                    break

                s_next, r, done, info = step_env(self.env, a)
                if r == 20:
                    delivered = 1

                if a in (0,1,2,3) and s_next == s:
                    fallback = random.choice([0,1,2,3])
                    s_fb, r_fb, done_fb, info_fb = step_env(self.env, fallback)
                    self.fallback_count += 1
                    total_reward += r
                    steps += 1
                    cum_reward += (self.gamma ** k) * r
                    k += 1
                    total_reward += r_fb
                    steps += 1
                    if r_fb == 20:
                        delivered = 1
                    cum_reward += (self.gamma ** k) * r_fb
                    k += 1
                    s = s_fb
                else:
                    if a == 4:
                        saw_pickup_in_option = True
                    total_reward += r
                    steps += 1
                    cum_reward += (self.gamma ** k) * r
                    k += 1
                    s = s_next

                # intra-option updates
                for o_prime in range(self.n_options):
                    a_o_prime = option_policy_action(self.env, s, o_prime)
                    # update only if o_prime would have chosen the same primitive a at the new state
                    if a_o_prime is not None and a_o_prime == a:
                        if done:
                            td_target = r
                        else:
                            td_target = r + self.gamma * self.Q[int(s), o_prime]
                        td_error = td_target - self.Q[int(s), o_prime]
                        self.Q[int(s), o_prime] += self.alpha * td_error

                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

                prim_steps += 1
                if option_terminates(self.env, s, o_selected) or done:
                    break
                if prim_steps >= self.option_step_cap:
                    break

            if saw_pickup_in_option:
                post_choice = epsilon_greedy(self.Q[int(s)], self.epsilon)
                self.post_pick_choice_counts[post_choice] += 1

            target = cum_reward
            if not done:
                target += (self.gamma ** k) * np.max(self.Q[int(s)])
            td_error = target - self.Q[int(s0), o_selected]
            self.Q[int(s0), o_selected] += self.alpha * td_error

        return total_reward, delivered, steps

    def train(self, n_episodes=5000, epsilon_decay=0.99995, min_epsilon=0.1, max_steps=2000, block_size=100):
        rewards = []
        block_delivery_counts = 0
        for ep in range(1, n_episodes + 1):
            r, delivered, steps = self.run_episode(max_steps_per_episode=max_steps)
            rewards.append(r)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)
            block_delivery_counts += delivered

            if ep % block_size == 0:
                print(f"[IntraOption] Episode {ep}/{n_episodes}, reward={r:.1f}, eps={self.epsilon:.4f}, deliveries_last{block_size}={block_delivery_counts}, fallbacks_last{block_size}={self.fallback_count}")
                print(f"        post-pick choice counts (last block): {self.post_pick_choice_counts}")
                block_delivery_counts = 0
                self.fallback_count = 0
                self.post_pick_choice_counts[:] = 0

        return rewards

# ---------------------------
def smooth(values, window=25):
    vals = np.asarray(values, dtype=np.float32)
    n = len(vals)
    if n == 0:
        return np.array([]), np.array([])
    if n < window or window <= 1:
        return vals, np.arange(n, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    sm = np.convolve(vals, kernel, mode="valid")
    offset = (window - 1) / 2.0
    centers = np.arange(len(sm), dtype=np.float32) + offset
    return sm, centers

def plot_rewards(sm_rewards, io_rewards, label1="SMDP Q", label2="Intra-option Q", outdir="out", smooth_window=50):
    ensure_dir(outdir)
    plt.figure(figsize=(11,6))
    episodes_s = np.arange(len(sm_rewards), dtype=np.float32)
    episodes_i = np.arange(len(io_rewards), dtype=np.float32)
    plt.plot(episodes_s, sm_rewards, alpha=0.25, label=f"{label1} (raw)")
    plt.plot(episodes_i, io_rewards, alpha=0.25, label=f"{label2} (raw)")
    s_sm, s_centers = smooth(sm_rewards, window=smooth_window)
    i_sm, i_centers = smooth(io_rewards, window=smooth_window)
    if s_sm.size > 0:
        plt.plot(s_centers, s_sm, linewidth=2.2, label=f"{label1} (smoothed w={smooth_window})")
    if i_sm.size > 0:
        plt.plot(i_centers, i_sm, linewidth=2.2, label=f"{label2} (smoothed w={smooth_window})")
    plt.xlabel("Episodes (0-based index)")
    plt.ylabel("Episode Return")
    plt.title("Reward comparison (raw + smoothed)")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(outdir, f"reward_compare_{int(time.time())}.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved reward plot to {fname}")
    plt.close()

def save_qtable(Q, filename):
    ensure_dir(os.path.dirname(filename) or ".")
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"Saved Q-table to {filename}")

# -------------------------------------------------------
def plot_after_training(env, smdp_agent, io_agent, sm_rewards, io_rewards):
    window = 25
    sm_smooth, sm_centers = smooth(sm_rewards, window=window)
    io_smooth, io_centers = smooth(io_rewards, window=window)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(sm_rewards)), sm_rewards, alpha=0.2, label="SMDP Q (raw)")
    plt.plot(np.arange(len(io_rewards)), io_rewards, alpha=0.2, label="Intra-option (raw)")
    if sm_smooth.size > 0:
        plt.plot(sm_centers, sm_smooth, label="SMDP Q-learning (smoothed)", linewidth=2)
    if io_smooth.size > 0:
        plt.plot(io_centers, io_smooth, label="Intra-option Q-learning (smoothed)", linewidth=2)
    plt.xlabel("Episodes (0-based)")
    plt.ylabel("Return")
    plt.title("Reward Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(smdp_agent.Q, aspect="auto")
    plt.colorbar()
    plt.title("SMDP Q-table Heatmap")
    plt.xlabel("Option")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(io_agent.Q, aspect="auto")
    plt.colorbar()
    plt.title("Intra-option Q-table Heatmap")
    plt.xlabel("Option")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()

    best_smdp = np.argmax(smdp_agent.Q, axis=1)
    best_io = np.argmax(io_agent.Q, axis=1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    n_opts = smdp_agent.Q.shape[1]
    plt.hist(best_smdp, bins=np.arange(-0.5, n_opts + 0.5, 1), rwidth=0.8)
    plt.title("SMDP Best Option Frequency")
    plt.xlabel("Option")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(best_io, bins=np.arange(-0.5, n_opts + 0.5, 1), rwidth=0.8)
    plt.title("Intra-option Best Option Frequency")
    plt.xlabel("Option")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    max_state = env.observation_space.n - 1
    sample_states = [0, 50, 100, 200, 350]
    sample_states = [min(max_state, s) for s in sample_states]
    print("\nDecoded sample states and best options:")
    for s in sample_states:
        try:
            decoded = decode_state(env, s)
        except Exception as e:
            decoded = f"<decode error: {e}>"
        print(f"State {s:3d} decoded={decoded} | "
              f"SMDP Best={best_smdp[s]}, Intra={best_io[s]}")

# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SMDP and/or Intra-option Q-learning on Taxi-v3 with BFS navigation options.")
    parser.add_argument("--n_episodes", type=int, default=2000, help="Number of episodes to train each agent")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy")
    parser.add_argument("--epsilon_decay", type=float, default=0.99995, help="Epsilon multiplicative decay per episode")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory for plots and qtables")
    parser.add_argument("--option_step_cap", type=int, default=300, help="Max primitive steps per option")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints for blocked-navigation and fallbacks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--algo", type=str, choices=["smdp", "intra", "both"], default="both", help="Which algorithm(s) to train")
    args = parser.parse_args()

    env = gym.make("Taxi-v3")

    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        env.action_space.seed(args.seed)
    except Exception:
        pass

    n_episodes = args.n_episodes
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    outdir = args.outdir

    sm_rewards = []
    io_rewards = []
    smdp_agent = None
    io_agent = None

    if args.algo in ("smdp", "both"):
        print("Starting SMDP Q-Learning training...")
        smdp_agent = SMDP_QLearner(env, n_options=4, alpha=alpha, gamma=gamma, epsilon=epsilon,
                                  option_step_cap=args.option_step_cap, enable_debug=args.debug, optimistic_init=0.1)
        sm_rewards = smdp_agent.train(
            n_episodes=n_episodes,
            epsilon_decay=epsilon_decay,
            min_epsilon=0.1,
            max_steps=1000
        )

    if args.algo in ("intra", "both"):
        print("Starting Intra-option Q-Learning training...")
        try:
            env.close()
        except Exception:
            pass
        env = gym.make("Taxi-v3")
        try:
            env.action_space.seed(args.seed)
        except Exception:
            pass

        io_agent = IntraOptionQLearner(env, n_options=4, alpha=alpha, gamma=gamma, epsilon=epsilon,
                                       option_step_cap=args.option_step_cap, enable_debug=args.debug, optimistic_init=0.1)
        io_rewards = io_agent.train(
            n_episodes=n_episodes,
            epsilon_decay=epsilon_decay,
            min_epsilon=0.1,
            max_steps=2000
        )

    # Only plot & save if both were run; otherwise plot what we have
    plot_rewards(sm_rewards if sm_rewards else io_rewards,
                 io_rewards if io_rewards else sm_rewards,
                 outdir=outdir, smooth_window=50)

    ensure_dir(os.path.join(outdir, "qtables"))
    if smdp_agent is not None:
        save_qtable(smdp_agent.Q, os.path.join(outdir, "qtables", f"q_smdp_{int(time.time())}.pkl"))
    if io_agent is not None:
        save_qtable(io_agent.Q, os.path.join(outdir, "qtables", f"q_intra_option_{int(time.time())}.pkl"))

    if smdp_agent is not None and io_agent is not None:
        plot_after_training(env, smdp_agent, io_agent, sm_rewards, io_rewards)

    print("\nExample best options at some states (state -> best_option):")
    for s in [0, 50, 100, 200, 350]:
        s_idx = min(env.observation_space.n - 1, s)
        best_smdp = np.argmax(smdp_agent.Q[s_idx]) if smdp_agent is not None else None
        best_io = np.argmax(io_agent.Q[s_idx]) if io_agent is not None else None
        print(f"state {s_idx}: SMDP->{best_smdp}, Intra->{best_io}")

if __name__ == "__main__":
    main()
