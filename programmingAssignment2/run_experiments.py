# run_experiments.py
# Requires starter_rl_gridworld.py in the same folder
# pip install numpy matplotlib seaborn tqdm

import numpy as np
import pickle
from collections import defaultdict
from tqdm import trange

from starter_ri_grid_world import (
    create_standard_grid,
    create_four_room,
    run_single_experiment,
    plot_training_curves,
    heatmap_state_visits,
    q_value_heatmap_and_policy
)

# ----------------------------
# Hyperparameter sweep for Standard and Four-Room grids
# ----------------------------
def run_full_experiments(seeds=[0,1,2,3,4], num_episodes=200, max_steps=100):
    alpha_set = [0.1, 0.3, 0.5]
    gamma_set = [0.9, 0.95, 0.99]
    epsilon_set = [0.05, 0.1, 0.2]
    tau_set = [0.1, 0.5, 1.0]

    results_summary = []

    # ----- Standard Grid Configs -----
    std_grid_configs = [
        {'start_state': np.array([[0, 4]]), 'transition_prob': 1.0, 'wind': False},
        {'start_state': np.array([[3, 6]]), 'transition_prob': 0.7, 'wind': False},
    ]

    algorithms = ['qlearning', 'sarsa']
    exploration_strategies = ['epsilon', 'softmax']

    print("\n=== Running Standard Grid Experiments ===")
    for config in std_grid_configs:
        env = create_standard_grid(start_state=config['start_state'],
                                   transition_prob=config['transition_prob'],
                                   wind=config['wind'])
        print(f"\nConfig: start={config['start_state']}, prob={config['transition_prob']}, wind={config['wind']}")

        for algo in algorithms:
            for strategy in exploration_strategies:
                print(f"Algorithm: {algo}, Exploration: {strategy}")
                best_avg_reward = -np.inf
                best_hyperparams = None

                for alpha in alpha_set:
                    for gamma in gamma_set:
                        if strategy == 'epsilon':
                            for epsilon in epsilon_set:
                                avg_rewards = []
                                for seed in seeds:
                                    res = run_single_experiment(
                                        env, algorithm=algo,
                                        algo_params={'alpha': alpha,
                                                     'gamma': gamma,
                                                     'epsilon': epsilon,
                                                     'max_steps': max_steps},
                                        num_episodes=num_episodes,
                                        seed=seed)
                                    avg_rewards.append(np.mean(res['rewards_per_episode']))
                                mean_reward = np.mean(avg_rewards)
                                if mean_reward > best_avg_reward:
                                    best_avg_reward = mean_reward
                                    best_hyperparams = {'alpha': alpha,
                                                        'gamma': gamma,
                                                        'epsilon': epsilon}
                        elif strategy == 'softmax':
                            for tau in tau_set:
                                avg_rewards = []
                                for seed in seeds:
                                    res = run_single_experiment(
                                        env, algorithm=algo,
                                        algo_params={'alpha': alpha,
                                                     'gamma': gamma,
                                                     'tau': tau,
                                                     'max_steps': max_steps},
                                        num_episodes=num_episodes,
                                        seed=seed)
                                    avg_rewards.append(np.mean(res['rewards_per_episode']))
                                mean_reward = np.mean(avg_rewards)
                                if mean_reward > best_avg_reward:
                                    best_avg_reward = mean_reward
                                    best_hyperparams = {'alpha': alpha,
                                                        'gamma': gamma,
                                                        'tau': tau}

                print(f"Best hyperparams for {algo} + {strategy}: {best_hyperparams}, avg reward={best_avg_reward:.2f}")
                results_summary.append({
                    'env_type': 'standard',
                    'config': config,
                    'algorithm': algo,
                    'exploration': strategy,
                    'best_hyperparams': best_hyperparams,
                    'best_avg_reward': best_avg_reward
                })

    # ----- Four-Room Configs -----
    four_room_configs = [
        {'goal_change': True},
        {'goal_change': False},
    ]

    print("\n=== Running Four-Room Experiments ===")
    for config in four_room_configs:
        env = create_four_room(goal_change=config['goal_change'])
        print(f"\nConfig: goal_change={config['goal_change']}")

        for algo in algorithms:
            for strategy in exploration_strategies:
                print(f"Algorithm: {algo}, Exploration: {strategy}")
                best_avg_reward = -np.inf
                best_hyperparams = None

                for alpha in alpha_set:
                    for gamma in gamma_set:
                        if strategy == 'epsilon':
                            for epsilon in epsilon_set:
                                avg_rewards = []
                                for seed in seeds:
                                    res = run_single_experiment(
                                        env, algorithm=algo,
                                        algo_params={'alpha': alpha,
                                                     'gamma': gamma,
                                                     'epsilon': epsilon,
                                                     'max_steps': max_steps},
                                        num_episodes=num_episodes,
                                        seed=seed)
                                    avg_rewards.append(np.mean(res['rewards_per_episode']))
                                mean_reward = np.mean(avg_rewards)
                                if mean_reward > best_avg_reward:
                                    best_avg_reward = mean_reward
                                    best_hyperparams = {'alpha': alpha,
                                                        'gamma': gamma,
                                                        'epsilon': epsilon}
                        elif strategy == 'softmax':
                            for tau in tau_set:
                                avg_rewards = []
                                for seed in seeds:
                                    res = run_single_experiment(
                                        env, algorithm=algo,
                                        algo_params={'alpha': alpha,
                                                     'gamma': gamma,
                                                     'tau': tau,
                                                     'max_steps': max_steps},
                                        num_episodes=num_episodes,
                                        seed=seed)
                                    avg_rewards.append(np.mean(res['rewards_per_episode']))
                                mean_reward = np.mean(avg_rewards)
                                if mean_reward > best_avg_reward:
                                    best_avg_reward = mean_reward
                                    best_hyperparams = {'alpha': alpha,
                                                        'gamma': gamma,
                                                        'tau': tau}

                print(f"Best hyperparams for {algo} + {strategy}: {best_hyperparams}, avg reward={best_avg_reward:.2f}")
                results_summary.append({
                    'env_type': 'four_room',
                    'config': config,
                    'algorithm': algo,
                    'exploration': strategy,
                    'best_hyperparams': best_hyperparams,
                    'best_avg_reward': best_avg_reward
                })

    return results_summary

# ----------------------------
# Final averaging over multiple runs
# ----------------------------
def run_final_averaged_experiments(env, algorithm, best_params, num_runs=100, num_episodes=200, max_steps=100):
    n_actions = env.num_actions
    total_rewards = np.zeros(num_episodes)
    total_steps = np.zeros(num_episodes)
    aggregated_state_visits = defaultdict(int)
    aggregated_Q = defaultdict(lambda: np.zeros(n_actions))

    for run_idx in trange(num_runs, desc=f"Running {algorithm} averaged experiments"):
        seed = run_idx
        res = run_single_experiment(env, algorithm=algorithm, algo_params={**best_params, 'max_steps': max_steps},
                                    num_episodes=num_episodes, seed=seed)
        total_rewards += np.array(res['rewards_per_episode'])
        total_steps += np.array(res['steps_per_episode'])

        for s, count in res['state_visit_counts'].items():
            aggregated_state_visits[s] += count

        for s, q_vals in res['Q'].items():
            if s not in aggregated_Q:
                aggregated_Q[s] = np.array(q_vals, dtype=np.float64)
            else:
                aggregated_Q[s] += np.array(q_vals, dtype=np.float64)

    avg_rewards = total_rewards / num_runs
    avg_steps = total_steps / num_runs
    for s in aggregated_Q:
        aggregated_Q[s] /= num_runs

    plot_training_curves(avg_rewards, avg_steps, title_prefix=f'{algorithm} (averaged)')
    heatmap_state_visits(env, aggregated_state_visits, title=f'{algorithm} Avg State Visits')
    q_value_heatmap_and_policy(env, aggregated_Q, title=f'{algorithm} Avg Q-values + Policy')

    return {
        'avg_rewards': avg_rewards,
        'avg_steps': avg_steps,
        'avg_state_visits': aggregated_state_visits,
        'avg_Q': aggregated_Q
    }

# ----------------------------
# Full end-to-end pipeline
# ----------------------------
def run_full_pipeline(seeds=[0,1,2,3,4], num_episodes=200, max_steps=100, num_runs=100):
    summary = run_full_experiments(seeds=seeds, num_episodes=num_episodes, max_steps=max_steps)

    summary_file = 'experiment_summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\nHyperparameter sweep completed. Summary saved to '{summary_file}'")

    for entry in summary:
        env_type = entry['env_type']
        config = entry['config']
        algo = entry['algorithm']
        best_params = entry['best_hyperparams']

        print(f"\nRunning final averaged experiment for {algo}, env_type={env_type}, config={config}")
        if env_type == 'standard':
            env = create_standard_grid(start_state=config['start_state'],
                                       transition_prob=config['transition_prob'],
                                       wind=config['wind'])
        elif env_type == 'four_room':
            env = create_four_room(goal_change=config['goal_change'])
        else:
            continue

        run_final_averaged_experiments(env, algo, best_params, num_runs=num_runs,
                                       num_episodes=num_episodes, max_steps=max_steps)

    print("\nEnd-to-end pipeline completed.")

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    if create_standard_grid is None or create_four_room is None:
        print("Environment not available. Add the Grid-World-Environment repo to import env.")
    else:
        run_full_pipeline(seeds=[0,1,2,3,4],
                          num_episodes=200,
                          max_steps=100,
                          num_runs=100)
