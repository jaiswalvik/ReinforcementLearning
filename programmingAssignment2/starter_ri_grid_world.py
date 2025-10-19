# save this as starter_rl_gridworld.py
# Requirements:
# numpy, matplotlib, seaborn, tqdm
# pip install numpy matplotlib seaborn tqdm

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import math
import pickle
import os

# ----------------------------
# Environment import
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
    max_q = max(q_vals)
    best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
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

def state_to_key(state):
    """
    Convert a state to an integer key suitable for indexing Q-tables or env arrays.
    
    GridWorld states may come as:
        - numpy arrays (e.g., [row, col])
        - lists or tuples (e.g., (row, col))
        - already an integer (state index)
    
    Returns:
        int: single integer representing the state
    """
    if isinstance(state, (np.ndarray, list, tuple)):
        # Flatten in case it's multi-dimensional and take the first element
        return int(np.ravel(state)[0])
    return int(state)

# ----------------------------
# SARSA implementation 
# ----------------------------
def train_sarsa(env, num_episodes, alpha, gamma, epsilon=None, tau=None,
                max_steps=100, seed=None, record_every=1):
    """Train SARSA on given env with integer state indices."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_actions = env.num_actions
    Q = defaultdict(lambda: [0.0] * n_actions)

    rewards_per_episode = []
    steps_per_episode = []
    state_visit_counts = defaultdict(int)

    for ep in range(num_episodes):
        state_key = state_to_key(env.start_state_seq)
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
            # Pass integer state index to env.step
            rew, next_state = env.step(state_key, a)
            next_key = state_to_key(next_state)
            state_visit_counts[next_key] += 1

            # Choose next action
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

            if rew >= 10 or steps >= max_steps:
                done = True

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

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
                     max_steps=100, seed=None):
    """Train Q-learning on given env with integer state indices."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_actions = env.num_actions
    Q = defaultdict(lambda: [0.0] * n_actions)
    rewards_per_episode = []
    steps_per_episode = []
    state_visit_counts = defaultdict(int)

    for ep in range(num_episodes):
        state_key = state_to_key(env.start_state_seq)
        state_visit_counts[state_key] += 1

        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Choose action
            if epsilon is not None:
                a = epsilon_greedy_action(Q, state_key, n_actions, epsilon)
            elif tau is not None:
                a = softmax_action(Q, state_key, n_actions, tau)
            else:
                a = random.randrange(n_actions)

            # Pass integer state index
            rew, next_state = env.step(state_key, a)
            next_key = state_to_key(next_state)
            state_visit_counts[next_key] += 1

            # Q-learning update
            Q[state_key][a] += alpha * (rew + gamma * max(Q[next_key]) - Q[state_key][a])

            state_key = next_key
            total_reward += rew
            steps += 1

            if rew >= 10 or steps >= max_steps:
                done = True

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

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

    Args:
        env: environment with num_cols
        state_index: int or list/array of ints

    Returns:
        list of tuples [(row, col), ...] if input is array
        or single tuple (row, col) if input is int
    """
    if np.isscalar(state_index):
        row = state_index // env.num_cols
        col = state_index % env.num_cols
        return (row, col)
    else:
        state_index = np.array(state_index)
        rows = state_index // env.num_cols
        cols = state_index % env.num_cols
        return list(zip(rows, cols))

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

    # Map action index to arrow directions
    action_to_delta = {
        0: (-0.4, 0.0),  # up
        1: (0.4, 0.0),   # down
        2: (0.0, -0.4),  # left
        3: (0.0, 0.4),   # right
    }

    for r in range(rows):
        for c in range(cols):
            a = policy_grid[r, c]
            if a >= 0:
                dy, dx = action_to_delta[a]
                plt.arrow(c + 0.5, r + 0.5, dx, dy,
                          head_width=0.12, head_length=0.12,
                          fc='k', ec='k')
    plt.show()

# ----------------------------
# Experiment runner (single config)
# ----------------------------
def run_single_experiment(env, algorithm='qlearning', algo_params=None,
                          num_episodes=500, seed=0):
    """Run a single SARSA or Q-learning experiment."""
    if algo_params is None:
        algo_params = {}
    alpha = algo_params.get('alpha', 0.1)
    gamma = algo_params.get('gamma', 0.9)
    epsilon = algo_params.get('epsilon', None)
    tau = algo_params.get('tau', None)
    max_steps = algo_params.get('max_steps', 100)

    if algorithm.lower() == 'qlearning':
        res = train_q_learning(env, num_episodes, alpha, gamma, epsilon, tau, max_steps, seed)
    elif algorithm.lower() == 'sarsa':
        res = train_sarsa(env, num_episodes, alpha, gamma, epsilon, tau, max_steps, seed)
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
        env = create_standard_grid(start_state=np.array([[0, 4]]),
                                   transition_prob=1.0,
                                   wind=False)
        print("Environment created: states:", env.num_states, "actions:", env.num_actions)
        algo_params = {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.05, 'max_steps': 100}
        res = run_single_experiment(env, algorithm='qlearning',
                                    algo_params=algo_params,
                                    num_episodes=200, seed=42)
        print("Avg reward (last 50 episodes):", np.mean(res['rewards_per_episode'][-50:]))
        plot_training_curves(res['rewards_per_episode'],
                             res['steps_per_episode'],
                             title_prefix='Q-Learning (smoke test)')
        heatmap_state_visits(env, res['state_visit_counts'], title='Visits (smoke test)')
        q_value_heatmap_and_policy(env, res['Q'], title='Q-values (smoke test)')


    # ----------------------------
    # TODO (your tasks)
    # ----------------------------
    # 1) Run the full set of configurations described in the assignment:
    #    - Standard grid: run Q-learning configs and SARSA configs by varying:
    #       * transition_prob {0.7, 1.0} (Q-learning)
    #       * start_state {(0,4), (3,6)}
    #       * exploration {epsilon-greedy, softmax}
    #       * wind False (Q-learning); wind True/False (SARSA)
    #    - Four-room: goal_change True/False for both algorithms
    #
    # 2) For each configuration, do hyperparameter tuning over alpha, gamma, epsilon/tau sets in assignment.
    #    For each hyperparameter tuple, run >=5 seeds and average results.
    #
    # 3) Save the best hyperparameters per configuration (based on avg reward or convergence).
    #
    # 4) Using best hyperparams, run 100 runs to produce final averaged plots:
    #    - Training curves (avg reward per episode, avg steps per episode)
    #    - State visit heatmap (averaged visits across runs)
    #    - Q-value heatmap + overlayed policy
    #
    # 5) Package your report: key code snippets, hyperparameter tables, and the plots above.