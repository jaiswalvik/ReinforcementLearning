# run_experiments.py
# Requires starter_ri_gridworld.py in the same folder
# pip install numpy matplotlib seaborn tqdm wandb

import numpy as np
import pickle
import os
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt 
import itertools
import time

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from starter_ri_grid_world import (
    create_standard_grid,
    create_four_room,
    run_single_experiment,
    plot_training_curves,
    heatmap_state_visits,
    q_value_heatmap_and_policy
)

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameter sets as per assignment
ALPHA_SET = [0.001, 0.01, 0.1, 1.0]
GAMMA_SET = [0.7, 0.8, 0.9, 1.0]
EPSILON_SET = [0.001, 0.01, 0.05, 0.1]
TAU_SET = [0.01, 0.1, 1.0, 2.0]

# Seeds for tuning and final
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Utility to compute selection metric: mean of last eval_window episodes
def selection_metric(rewards_list, eval_window=50):
    # rewards_list: list of arrays per seed (each array length num_episodes)
    arr = np.array([np.array(r) for r in rewards_list])
    # ensure episodes >= eval_window
    if arr.shape[1] < eval_window:
        # fallback to mean of all episodes
        return float(arr.mean())
    return float(np.mean(arr[:, -eval_window:]))

def save_pickle(obj, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path

# Create list of all 20 configs required by assignment
def create_all_configs():
    configs = []

    # Standard grid Q-learning
    start_states = [np.array([[0, 4]]), np.array([[3, 6]])]
    trans_probs = [1.0, 0.7]
    for start in start_states:
        for tp in trans_probs:
            for exploration in ['epsilon', 'softmax']:
                configs.append({
                    'env_type': 'standard',
                    'algorithm': 'qlearning',
                    'start_state': start,
                    'transition_prob': tp,
                    'wind': False,
                    'exploration': exploration
                })

    # Standard grid SARSA
    for start in start_states:
        for wind in [False, True]:
            for exploration in ['epsilon', 'softmax']:
                configs.append({
                    'env_type': 'standard',
                    'algorithm': 'sarsa',
                    'start_state': start,
                    'transition_prob': 1.0,
                    'wind': wind,
                    'exploration': exploration
                })

    # Four-room (no extra exploration duplication)
    for algo in ['qlearning', 'sarsa']:
        for goal_change in [True, False]:
            configs.append({
                'env_type': 'four_room',
                'algorithm': algo,
                'goal_change': goal_change,
                'exploration': 'epsilon'  # one exploration type
            })

    assert len(configs) == 20, f"Expected 20 configs, got {len(configs)}"
    return configs

# Hyperparameter tuning for a single config
def tune_hyperparams_for_config(config, seeds=DEFAULT_SEEDS, num_episodes=200, max_steps=100, eval_window=50, use_wandb=False):
    """
    Returns best_hyperparams dict and a dict of all results.
    """
    all_results = []
    best_score = -np.inf
    best_hparams = None

    # iterate over hyperparameter space
    for alpha in ALPHA_SET:
        for gamma in GAMMA_SET:
            if config['exploration'] == 'epsilon':
                for epsilon in EPSILON_SET:
                    # run seeds
                    rewards_per_seed = []
                    for seed in seeds:
                        # recreate env per seed
                        if config['env_type'] == 'standard':
                            env = create_standard_grid(start_state=config['start_state'],
                                                       transition_prob=config['transition_prob'],
                                                       wind=config['wind'])
                        elif config['env_type'] == 'four_room':
                            env = create_four_room(goal_change=config['goal_change'],
                                                   transition_prob=1.0)
                        else:
                            raise ValueError("Unknown env type")

                        algo_params = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon, 'max_steps': max_steps}
                        # optionally create a short-lived wandb run for this seed/hparam
                        wandb_run = None
                        if use_wandb and WANDB_AVAILABLE:
                            wandb_run = wandb.init(project="gridworld-td", reinit=True,
                                                   name=f"tune_{config['algorithm']}_{config['env_type']}_a{alpha}_g{gamma}_e{epsilon}_s{seed}",
                                                   config={**config, **algo_params})
                        res = run_single_experiment(env, algorithm=config['algorithm'], algo_params=algo_params,
                                                    num_episodes=num_episodes, seed=seed, wandb_run=wandb_run)
                        if wandb_run is not None:
                            wandb_run.finish()
                        rewards_per_seed.append(res['rewards_per_episode'])
                    score = selection_metric(rewards_per_seed, eval_window=eval_window)
                    entry = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon, 'score': score}
                    all_results.append(entry)
                    if score > best_score:
                        best_score = score
                        best_hparams = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
            else:  # softmax
                for tau in TAU_SET:
                    rewards_per_seed = []
                    for seed in seeds:
                        if config['env_type'] == 'standard':
                            env = create_standard_grid(start_state=config['start_state'],
                                                       transition_prob=config['transition_prob'],
                                                       wind=config['wind'])
                        elif config['env_type'] == 'four_room':
                            env = create_four_room(goal_change=config['goal_change'],
                                                   transition_prob=1.0)
                        else:
                            raise ValueError("Unknown env type")

                        algo_params = {'alpha': alpha, 'gamma': gamma, 'tau': tau, 'max_steps': max_steps}
                        wandb_run = None
                        if use_wandb and WANDB_AVAILABLE:
                            wandb_run = wandb.init(project="gridworld-td", reinit=True,
                                                   name=f"tune_{config['algorithm']}_{config['env_type']}_a{alpha}_g{gamma}_t{tau}_s{seed}",
                                                   config={**config, **algo_params})
                        res = run_single_experiment(env, algorithm=config['algorithm'], algo_params=algo_params,
                                                    num_episodes=num_episodes, seed=seed, wandb_run=wandb_run)
                        if wandb_run is not None:
                            wandb_run.finish()
                        rewards_per_seed.append(res['rewards_per_episode'])
                    score = selection_metric(rewards_per_seed, eval_window=eval_window)
                    entry = {'alpha': alpha, 'gamma': gamma, 'tau': tau, 'score': score}
                    all_results.append(entry)
                    if score > best_score:
                        best_score = score
                        best_hparams = {'alpha': alpha, 'gamma': gamma, 'tau': tau}

    return best_hparams, best_score, all_results

# Final averaging runs (100 runs) using best hyperparams
def run_final_averaged_experiments(env_factory, config, best_params, num_runs=100, num_episodes=200, max_steps=100, use_wandb=False):
    """
    env_factory: a callable that returns a fresh env (no args) per run
    config: config dict (for logging)
    best_params: dict of alpha/gamma and epsilon or tau
    """
    n_actions = None
    total_rewards = np.zeros(num_episodes)
    total_steps = np.zeros(num_episodes)
    aggregated_state_visits = defaultdict(int)
    aggregated_Q = defaultdict(lambda: np.zeros(4))

    for run_idx in trange(num_runs, desc=f"Final runs {config['algorithm']}-{config['env_type']}"):
        seed = run_idx
        env = env_factory()  # fresh env
        n_actions = env.num_actions
        algo_params = dict(best_params)
        algo_params['max_steps'] = max_steps

        wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            wandb_run = wandb.init(project="gridworld-td", reinit=True,
                                   name=f"final_{config['algorithm']}_{config['env_type']}_run{run_idx}",
                                   config={**config, **algo_params})

        res = run_single_experiment(env, algorithm=config['algorithm'],
                                    algo_params=algo_params,
                                    num_episodes=num_episodes, seed=seed, wandb_run=wandb_run)
        if wandb_run is not None:
            # log summary stats
            wandb_run.log({'avg_reward': np.mean(res['rewards_per_episode']),
                           'avg_steps': np.mean(res['steps_per_episode'])})
            wandb_run.finish()

        total_rewards += np.array(res['rewards_per_episode'])
        total_steps += np.array(res['steps_per_episode'])

        for s, count in res['state_visit_counts'].items():
            aggregated_state_visits[s] += count

        for s, q_vals in res['Q'].items():
            aggregated_Q[s] += np.array(q_vals, dtype=np.float64)

    avg_rewards = total_rewards / num_runs
    avg_steps = total_steps / num_runs
    for s in aggregated_Q:
        aggregated_Q[s] /= num_runs

    # Plotting (and saving) results
    plot_training_curves(avg_rewards, avg_steps, title_prefix=f"{config['algorithm']} {config['env_type']} final")
    # save figure images
    plt.savefig(os.path.join(RESULTS_DIR, f"{config['algorithm']}_{config['env_type']}_training_curves.png"))
    heatmap_state_visits(env, aggregated_state_visits, title=f"{config['algorithm']} Avg State Visits")
    plt.savefig(os.path.join(RESULTS_DIR, f"{config['algorithm']}_{config['env_type']}_state_visits.png"))
    q_value_heatmap_and_policy(env, aggregated_Q, title=f"{config['algorithm']} Avg Q-values + Policy")
    plt.savefig(os.path.join(RESULTS_DIR, f"{config['algorithm']}_{config['env_type']}_q_policy.png"))

    return {
        'avg_rewards': avg_rewards,
        'avg_steps': avg_steps,
        'avg_state_visits': aggregated_state_visits,
        'avg_Q': aggregated_Q
    }

# Full pipeline: tuning + final runs for all configs
def run_full_pipeline(seeds=DEFAULT_SEEDS, num_episodes_tune=200, num_episodes_final=200,
                      max_steps=100, num_runs_final=100, use_wandb=False):
    configs = create_all_configs()
    summary = []

    for idx, config in enumerate(configs):
        print(f"\n=== Config {idx+1}/{len(configs)}: {config} ===")
        # Tuning
        start_time = time.time()
        best_hparams, best_score, all_results = tune_hyperparams_for_config(config, seeds=seeds,
                                                                            num_episodes=num_episodes_tune,
                                                                            max_steps=max_steps,
                                                                            eval_window=50,
                                                                            use_wandb=use_wandb)
        tuning_time = time.time() - start_time
        print(f"Best hparams: {best_hparams}, score: {best_score:.4f} (tuning time {tuning_time:.1f}s)")

        # Save tuning results
        tune_fname = f"tuning_{config['algorithm']}_{config['env_type']}_{idx}.pkl"
        save_pickle({'config': config, 'best_hparams': best_hparams, 'best_score': best_score, 'all_results': all_results},
                    tune_fname)

        # Prepare env factory for final runs
        if config['env_type'] == 'standard':
            def make_env(start=config['start_state'], tp=config['transition_prob'], wind=config['wind']):
                return create_standard_grid(start_state=start, transition_prob=tp, wind=wind)
            env = make_env()
        else:
            def make_env(goal_change=config['goal_change']):
                return create_four_room(goal_change=goal_change)
            env = make_env()

        # Run final averaged experiments using best hyperparams
        print("Running final averaged experiments (this can take time)...")
        final_res = run_final_averaged_experiments(lambda: make_env(), config, best_hparams,
                                                   num_runs=num_runs_final,
                                                   num_episodes=num_episodes_final,
                                                   max_steps=max_steps, use_wandb=use_wandb)

        final_fname = f"final_{config['algorithm']}_{config['env_type']}_{idx}.pkl"
        serializable_res = {
            'config': config,
            'best_hparams': best_hparams,
            'avg_rewards': list(map(float, final_res['avg_rewards'])),
            'avg_steps': list(map(float, final_res['avg_steps']))
        }
        save_pickle(serializable_res, final_fname)
        summary_entry = {
            'config': config,
            'best_hparams': best_hparams,
            'best_score': best_score,
            'tune_file': tune_fname,
            'final_file': final_fname
        }
        summary.append(summary_entry)

    # Save overall summary
    summary_path = save_pickle(summary, "experiment_summary.pkl")
    print(f"\nAll experiments done. Summary saved to {summary_path}")
    return summary

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Quick mode for interactive testing: reduce sizes
    QUICK = False

    if QUICK:
        seeds = [0, 1]
        num_episodes_tune = 50
        num_episodes_final = 50
        num_runs_final = 5
    else:
        seeds = DEFAULT_SEEDS
        num_episodes_tune = 200
        num_episodes_final = 200
        num_runs_final = 100

    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        print("wandb available and will be used for logging.")
    else:
        print("wandb not available or not configured. Running without external logging.")

    run_full_pipeline(seeds=seeds,
                      num_episodes_tune=num_episodes_tune,
                      num_episodes_final=num_episodes_final,
                      max_steps=100,
                      num_runs_final=num_runs_final,
                      use_wandb=use_wandb)
