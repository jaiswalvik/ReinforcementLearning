import numpy as np
import itertools
import json
import csv
import os
import matplotlib
matplotlib.use('Agg')  # ✅ Non-blocking backend
import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm

from env import create_standard_grid, create_four_room
from starter_ri_grid_world import (
    train_q_learning,
    train_sarsa,
    plot_training_curves,
    heatmap_state_visits,
    q_value_heatmap_and_policy,
)

# ----------------------------------------------------------
# Debug Mode Toggle — Set to False for full experiments
# ----------------------------------------------------------
DEBUG = False


# ----------------------------------------------------------
# Logging Utility — Saves all console output to a file
# ----------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", buffering=1,encoding="utf-8")  # line-buffered
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


# ----------------------------------------------------------
# Experiment Configurations
# ----------------------------------------------------------
def get_configurations():
    configs = []

    # Standard Grid - Q-learning
    for transition_prob in [0.7, 1.0]:
        for start_state in [[0, 4], [3, 6]]:
            for exploration_strategy in ["epsilon_greedy", "softmax"]:
                configs.append({
                    "env_type": "standard",
                    "algo": "q_learning",
                    "transition_prob": transition_prob,
                    "start_state": np.array([start_state]),
                    "exploration": exploration_strategy,
                    "wind": False,
                })

    # Standard Grid - SARSA
    for wind in [True, False]:
        for start_state in [[0, 4], [3, 6]]:
            for exploration_strategy in ["epsilon_greedy", "softmax"]:
                configs.append({
                    "env_type": "standard",
                    "algo": "sarsa",
                    "transition_prob": 1.0,
                    "start_state": np.array([start_state]),
                    "exploration": exploration_strategy,
                    "wind": wind,
                })

    # Four Room GridWorld
    for goal_change in [True, False]:
        for algo in ["q_learning", "sarsa"]:
            configs.append({
                "env_type": "four_room",
                "algo": algo,
                "goal_change": goal_change,
            })

    # ✅ In debug mode — run a small subset
    if DEBUG:
        configs = configs[:2]

    return configs


# ----------------------------------------------------------
# Run a single experiment
# ----------------------------------------------------------
def run_experiment(config, alpha, gamma, exp_param, seeds=[0, 1, 2, 3, 4]):
    rewards_all, steps_all = [], []

    for seed in seeds:
        np.random.seed(seed)

        if config["env_type"] == "standard":
            env = create_standard_grid(
                start_state=config["start_state"],
                transition_prob=config.get("transition_prob", 1.0),
                wind=config.get("wind", False),
            )
        else:
            env = create_four_room(goal_change=config.get("goal_change", False))

        train_func = train_q_learning if config["algo"] == "q_learning" else train_sarsa
        res = train_func(
            env,
            num_episodes=50 if DEBUG else 200,
            alpha=alpha,
            gamma=gamma,
            epsilon=exp_param if config.get("exploration") == "epsilon_greedy" else None,
            tau=exp_param if config.get("exploration") == "softmax" else None,
            seed=seed,
        )

        rewards_all.append(res["rewards_per_episode"])
        steps_all.append(res["steps_per_episode"])

    avg_rewards = np.mean(rewards_all, axis=0)
    avg_steps = np.mean(steps_all, axis=0)
    return avg_rewards, avg_steps, res["Q"], res.get("state_visit_counts", None), env


# ----------------------------------------------------------
# Hyperparameter tuning across configurations
# ----------------------------------------------------------
def tune_hyperparameters(save_dir="results", num_runs=5):
    os.makedirs(save_dir, exist_ok=True)
    configs = get_configurations()

    alphas = [0.1] if DEBUG else [0.001, 0.01, 0.1, 1.0]
    gammas = [0.9] if DEBUG else [0.7, 0.8, 0.9, 1.0]
    epsilons = [0.1] if DEBUG else [0.001, 0.01, 0.05, 0.1]
    taus = [0.1] if DEBUG else [0.01, 0.1, 1, 2]

    all_results = []
    csv_path = os.path.join(save_dir, "best_hyperparams.csv")
    start_time = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Environment", "Algorithm", "Exploration", "Best α", "Best γ", "Best ε/τ", "Final Reward"])
        csvfile.flush()

        with tqdm(total=len(configs), desc="Running Configurations", ncols=100) as pbar:
            best_overall = {"reward": -np.inf, "config": None}

            for i, config in enumerate(configs, start=1):
                best_avg_reward = -np.inf
                best_params = {}

                if config.get("exploration") == "epsilon_greedy":
                    exp_values = epsilons
                elif config.get("exploration") == "softmax":
                    exp_values = taus
                else:
                    exp_values = [0.1]

                for alpha, gamma, exp_param in itertools.product(alphas, gammas, exp_values):
                    seeds = list(range(num_runs))
                    avg_rewards, avg_steps, Q, state_visits, env = run_experiment(config, alpha, gamma, exp_param, seeds)
                    mean_final_reward = np.mean(avg_rewards[-(10 if DEBUG else 50):])  # ✅ adaptive averaging

                    if mean_final_reward > best_avg_reward:
                        best_avg_reward = mean_final_reward
                        best_params = {
                            "alpha": alpha,
                            "gamma": gamma,
                            "exp_param": exp_param,
                            "final_reward": mean_final_reward
                        }
                        best_rewards, best_steps, best_Q, best_env = avg_rewards, avg_steps, Q, env

                tag = f"{config['algo']}_{config['env_type']}_{config.get('exploration', '')}_{i}"

                # Plot training curves
                plot_training_curves(best_rewards, best_steps, title_prefix=tag)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{tag}_training_curves.png"))
                plt.clf()
                plt.close('all')

                # Heatmap and policy visualization
                heatmap_state_visits(best_env, state_visits, title=f"State Visits - {tag}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{tag}_heatmap.png"))
                plt.clf()
                plt.close('all')

                q_value_heatmap_and_policy(best_env, best_Q, title=f"Q-Values - {tag}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{tag}_policy.png"))
                plt.clf()
                plt.close('all')

                writer.writerow([
                    config["env_type"], config["algo"], config.get("exploration", "default"),
                    best_params["alpha"], best_params["gamma"], best_params["exp_param"],
                    round(best_params["final_reward"], 3)
                ])
                csvfile.flush()

                all_results.append({"config": config, "best_params": best_params})

                if best_avg_reward > best_overall["reward"]:
                    best_overall = {"reward": best_avg_reward, "config": config}

                pbar.set_postfix({"BestReward": round(best_avg_reward, 2)})
                pbar.update(1)

    # Save all results to JSON
    with open(os.path.join(save_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    duration = time.time() - start_time
    print(f"\n✅ All configurations completed in {duration/60:.2f} minutes using {num_runs} runs/config.")
    if best_overall["config"] is not None:
        print(f"🏆 Best overall configuration: {best_overall['config']}")
    else:
        print("⚠️ No valid configurations found.")
    print(f"📁 Results saved to '{save_dir}/'.")


# ----------------------------------------------------------
# Main entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    # 🕒 Auto timestamped result directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 🪶 Setup logging to both terminal and file
    sys.stdout = Logger(os.path.join(save_dir, "experiment_log.txt"))

    print(f"🚀 Starting hyperparameter tuning (DEBUG={DEBUG})...\n")
    tune_hyperparameters(save_dir=save_dir, num_runs=1 if DEBUG else 5)
    print("\n✅ All experiments completed successfully.")
