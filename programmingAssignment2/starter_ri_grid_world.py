# starter_ri_grid_world.py
# Requirements:
# numpy, matplotlib, seaborn, tqdm, wandb (optional)
# pip install numpy matplotlib seaborn tqdm wandb

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import math
import pickle
import os

# Action constants
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# ----------------------------
# Environment import (env.py must provide create_standard_grid, create_four_room)
# ----------------------------
try:
    from env import create_standard_grid, create_four_room
except Exception as e:
    print("WARNING: Could not import env. Make sure the Grid-World-Environment repo is available.")
    print("Exception:", e)
    create_standard_grid = None
    create_four_room = None

# ----------------------------
# Utilities: action selection
# ----------------------------
def epsilon_greedy_action(Q, state_key, n_actions, epsilon):
    """Return action index according to epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    q_vals = Q[state_key]
    # q_vals may be list or numpy array
    max_q = float(np.max(q_vals))
    best_actions = [i for i, q in enumerate(q_vals) if float(q) == max_q]
    return random.choice(best_actions)

def softmax_action(Q, state_key, n_actions, tau):
    """Softmax (Boltzmann) action selection."""
    q_vals = np.array(Q[state_key], dtype=np.float64)
    if tau <= 0:
        max_q = q_vals.max()
        best = np.random.choice(np.where(q_vals == max_q)[0])
        return int(best)
    z = q_vals - np.max(q_vals)
    exps = np.exp(z / tau)
    probs = exps / np.sum(exps)
    return int(np.random.choice(len(probs), p=probs))

def state_to_key(state, env=None):
    """
    Convert a state to an integer key suitable for indexing Q-tables or env arrays.

    Handles:
      - integer state index
      - numpy array or list with single scalar (e.g., np.array([23]) or array(23))
      - (row, col) tuple or list *if env is provided* (will convert to seq)
    """
    if isinstance(state, (np.ndarray,)):
        flat = np.ravel(state)
        if flat.size == 1:
            return int(flat[0])
        # if it's (r,c) pair and env provided, convert
        if flat.size == 2 and env is not None:
            r, c = int(flat[0]), int(flat[1])
            return int(r * env.num_cols + c)
        raise ValueError(f"Unsupported numpy state shape: {state.shape}")
    if isinstance(state, (list, tuple)):
        if len(state) == 1:
            return int(state[0])
        if len(state) == 2 and env is not None:
            r, c = int(state[0]), int(state[1])
            return int(r * env.num_cols + c)
        raise ValueError(f"Unsupported state list/tuple: {state}")
    return int(state)

def is_goal_state(env, state_key):
    """Return True if state_key corresponds to one of env's goal states."""
    if hasattr(env, 'goal_states_seq'):
        flat = np.ravel(env.goal_states_seq).astype(int)
        return int(state_key) in set(flat.tolist())
    return False

# ----------------------------
# SARSA implementation 
# ----------------------------
def train_sarsa(env, num_episodes, alpha, gamma, epsilon=None, tau=None,
                max_steps=100, seed=None, record_every=1, wandb_run=None):
    """Train SARSA on given env. Caller should create a fresh env per call for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_actions = env.num_actions
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    rewards_per_episode = []
    steps_per_episode = []
    state_visit_counts = defaultdict(int)

    for ep in range(num_episodes):
        # ensure environment start/goal reset behavior occurs
        start_seq = env.reset()
        state_key = state_to_key(start_seq, env)
        state_visit_counts[state_key] += 1

        # Choose initial action
        if epsilon is not None:
            a = epsilon_greedy_action(Q, state_key, n_actions, epsilon)
        elif tau is not None:
            a = softmax_action(Q, state_key, n_actions, tau)
        else:
            a = random.randrange(n_actions)

        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            next_state, rew = env.step(state_key, a)
            # env.step in assignment returns (next_state, reward) — handle both orders if necessary
            # If type indicates the first is reward (unlikely with provided env), try to detect:
            if isinstance(next_state, (int, np.integer)) and (isinstance(rew, (float, np.floating, np.ndarray))):
                # expected ordering (next_state, reward)
                pass
            # ensure reward is scalar float
            rew = float(np.array(rew).reshape(-1)[0])
            next_key = state_to_key(next_state, env)
            state_visit_counts[next_key] += 1

            # Choose next action (on-policy)
            if epsilon is not None:
                a_next = epsilon_greedy_action(Q, next_key, n_actions, epsilon)
            elif tau is not None:
                a_next = softmax_action(Q, next_key, n_actions, tau)
            else:
                a_next = random.randrange(n_actions)

            # SARSA update
            Q[state_key][a] += alpha * (rew + gamma * Q[next_key][a_next] - Q[state_key][a])

            state_key = next_key
            a = a_next
            total_reward += rew
            steps += 1

            # terminal check
            if is_goal_state(env, state_key) or steps >= max_steps:
                done = True

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # optional logging to wandb
        if wandb_run is not None and ((ep + 1) % record_every == 0):
            wandb_run.log({"episode": ep + 1,
                           "reward": total_reward,
                           "steps": steps,
                           "algorithm": "sarsa"})

    return {
        'Q': Q,
        'rewards_per_episode': rewards_per_episode,
        'steps_per_episode': steps_per_episode,
        'state_visit_counts': state_visit_counts
    }


# ----------------------------
# Q-Learning implementation 
# ----------------------------
def train_q_learning(env, num_episodes, alpha, gamma, epsilon=None, tau=None,
                     max_steps=100, seed=None, record_every=1, wandb_run=None):
    """Train Q-learning on given env. Caller should create a fresh env per call for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_actions = env.num_actions
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    rewards_per_episode = []
    steps_per_episode = []
    state_visit_counts = defaultdict(int)

    for ep in range(num_episodes):
        start_seq = env.reset()
        state_key = state_to_key(start_seq, env)
        state_visit_counts[state_key] += 1

        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Choose action (off-policy)
            if epsilon is not None:
                a = epsilon_greedy_action(Q, state_key, n_actions, epsilon)
            elif tau is not None:
                a = softmax_action(Q, state_key, n_actions, tau)
            else:
                a = random.randrange(n_actions)

            next_state, rew = env.step(state_key, a)
            rew = float(np.array(rew).reshape(-1)[0])
            next_key = state_to_key(next_state, env)
            state_visit_counts[next_key] += 1

            # Q-learning update (off-policy)
            Q[state_key][a] += alpha * (rew + gamma * np.max(Q[next_key]) - Q[state_key][a])

            state_key = next_key
            total_reward += rew
            steps += 1

            if is_goal_state(env, state_key) or steps >= max_steps:
                done = True

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if wandb_run is not None and ((ep + 1) % record_every == 0):
            wandb_run.log({"episode": ep + 1,
                           "reward": total_reward,
                           "steps": steps,
                           "algorithm": "qlearning"})

    return {
        'Q': Q,
        'rewards_per_episode': rewards_per_episode,
        'steps_per_episode': steps_per_episode,
        'state_visit_counts': state_visit_counts
    }

# ----------------------------
# Visualization utilities
# ----------------------------
def plot_training_curves(rewards, steps, title_prefix=''):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{title_prefix} Reward per Episode')

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{title_prefix} Steps per Episode')
    plt.tight_layout()
    plt.show()

def seq_to_row_col(env, state_index):
    """
    Convert a flattened state index to (row, col) for the GridWorld.
    """
    if np.isscalar(state_index):
        row = int(state_index) // env.num_cols
        col = int(state_index) % env.num_cols
        return (row, col)
    else:
        state_index = np.array(state_index, dtype=int)
        rows = state_index // env.num_cols
        cols = state_index % env.num_cols
        return list(zip(rows.tolist(), cols.tolist()))

def heatmap_state_visits(env, state_visit_counts, title='State Visit Heatmap'):
    """Create heatmap of state visits."""
    if len(state_visit_counts) == 0:
        print("No state visit data.")
        return

    rows = getattr(env, "num_rows", 10)
    cols = getattr(env, "num_cols", 10)
    grid = np.zeros((rows, cols))

    for s, count in state_visit_counts.items():
        row, col = seq_to_row_col(env, s)
        grid[row, col] = count

    plt.figure(figsize=(6, 6))
    sns.heatmap(grid, annot=False, fmt='g', cmap='viridis')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

def q_value_heatmap_and_policy(env, Q, title='Q-values and Derived Policy'):
    """Plot heatmap of max Q-values per state with policy arrows."""
    if len(Q) == 0:
        print("No Q data.")
        return

    rows = getattr(env, "num_rows", 10)
    cols = getattr(env, "num_cols", 10)
    grid = np.zeros((rows, cols))
    policy_grid = np.full((rows, cols), -1)

    for s, q_vals in Q.items():
        row, col = seq_to_row_col(env, s)
        grid[row, col] = np.max(q_vals)
        policy_grid[row, col] = np.argmax(q_vals)

    plt.figure(figsize=(7, 7))
    sns.heatmap(grid, cmap='coolwarm')
    plt.title(title)
    plt.gca().invert_yaxis()

    # Map action index to arrow directions (dx, dy)
    action_to_delta = {
        0: (0.0, -0.4),  # up    -> dx=0, dy=-0.4
        1: (0.0, 0.4),   # down  -> dx=0, dy=0.4
        2: (-0.4, 0.0),  # left  -> dx=-0.4, dy=0
        3: (0.4, 0.0),   # right -> dx=0.4, dy=0
    }

    for r in range(rows):
        for c in range(cols):
            a = policy_grid[r, c]
            if a >= 0:
                dx, dy = action_to_delta[int(a)]
                plt.arrow(c + 0.5, r + 0.5, dx, dy,
                          head_width=0.12, head_length=0.12,
                          fc='k', ec='k')
    plt.show()


# ----------------------------
# Experiment runner (single config)
# ----------------------------
def run_single_experiment(env, algorithm='qlearning', algo_params=None,
                          num_episodes=500, seed=0, wandb_run=None):
    """
    Run a single SARSA or Q-learning experiment.
    Note: env should be a fresh environment object for each call (caller must recreate env per seed).
    """
    if algo_params is None:
        algo_params = {}
    alpha = algo_params.get('alpha', 0.1)
    gamma = algo_params.get('gamma', 0.9)
    epsilon = algo_params.get('epsilon', None)
    tau = algo_params.get('tau', None)
    max_steps = algo_params.get('max_steps', 100)

    if algorithm.lower() == 'qlearning':
        res = train_q_learning(env, num_episodes, alpha, gamma, epsilon, tau, max_steps, seed, wandb_run=wandb_run)
    elif algorithm.lower() == 'sarsa':
        res = train_sarsa(env, num_episodes, alpha, gamma, epsilon, tau, max_steps, seed, wandb_run=wandb_run)
    else:
        raise ValueError("Unknown algorithm")
    return res


# ----------------------------
# Example usage (smoke test)
# ----------------------------
if __name__ == '__main__':
    if create_standard_grid is None:
        print("Env not available. Add the Grid-World-Environment repo to import env.")
    else:
        # Create environment
        env = create_standard_grid(start_state=np.array([[0, 4]]),
                                   transition_prob=1.0,
                                   wind=False)
        print("Environment created: states:", env.num_states, "actions:", env.num_actions)

        # Shared hyperparameters
        algo_params = {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.05, 'max_steps': 100}

        # ----------------------------
        # Q-Learning Smoke Test
        # ----------------------------
        res_q = run_single_experiment(env,
                                      algorithm='qlearning',
                                      algo_params=algo_params,
                                      num_episodes=200,
                                      seed=42)
        print("Q-Learning | Avg reward (last 50 episodes):",
              np.mean(res_q['rewards_per_episode'][-50:]))
        plot_training_curves(res_q['rewards_per_episode'],
                             res_q['steps_per_episode'],
                             title_prefix='Q-Learning (smoke test)')
        heatmap_state_visits(env, res_q['state_visit_counts'],
                             title='Visits (Q-Learning)')
        q_value_heatmap_and_policy(env, res_q['Q'], title='Q-values (Q-Learning)')

        # ----------------------------
        # SARSA Smoke Test
        # ----------------------------
        res_sarsa = run_single_experiment(env,
                                          algorithm='sarsa',
                                          algo_params=algo_params,
                                          num_episodes=200,
                                          seed=42)
        print("SARSA | Avg reward (last 50 episodes):",
              np.mean(res_sarsa['rewards_per_episode'][-50:]))
        plot_training_curves(res_sarsa['rewards_per_episode'],
                             res_sarsa['steps_per_episode'],
                             title_prefix='SARSA (smoke test)')
        heatmap_state_visits(env, res_sarsa['state_visit_counts'],
                             title='Visits (SARSA)')
        q_value_heatmap_and_policy(env, res_sarsa['Q'], title='Q-values (SARSA)')

        # ----------------------------
        # Comparison Plot
        # ----------------------------
        try:
            plot_comparison(
                [res_q, res_sarsa],
                labels=['Q-Learning', 'SARSA'],
                metric='rewards_per_episode',
                title='Reward Comparison: Q-Learning vs SARSA'
            )
        except NameError:
            import matplotlib.pyplot as plt
            # Fallback simple comparison if plot_comparison() is not defined
            plt.figure(figsize=(8, 5))
            plt.plot(res_q['rewards_per_episode'], label='Q-Learning')
            plt.plot(res_sarsa['rewards_per_episode'], label='SARSA')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Reward Comparison: Q-Learning vs SARSA')
            plt.legend()
            plt.grid(True)
            plt.show()

