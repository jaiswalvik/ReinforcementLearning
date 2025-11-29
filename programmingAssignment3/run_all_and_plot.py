#!/usr/bin/env python3
"""
run_all_and_plot.py

1) Runs all experiments serially:
   - dueling_dqn.py (mean, max) for seeds 0..4
   - reinforce.py (baseline, no baseline) for seeds 0..4
   - Envs: CartPole-v1, Acrobot-v1
   - Saves results/ files

2) After experiments finish:
   - Loads results/*.npy
   - Generates plots/algo_comparison_<env>.png
"""

import subprocess
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# ------------------------------------------------
# THIS IS YOUR PYTHON INTERPRETER (WITH GYMNASIUM)
# ------------------------------------------------
PY = r"D:\ReinforcementLearning\env\Scripts\python.exe"

# ---------- Config ----------
ENVS = ["CartPole-v1", "Acrobot-v1"]
DUELING_VARIANTS = ["mean", "max"]
SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
EPISODES = "500"  # must be string for subprocess commands

# ---------- Helper to run commands ----------
def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("\n❌ ERROR:", e)
        print("Command failed:", " ".join(cmd))
        raise

# ---------- Phase 1: Run experiments ----------
def run_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for env in ENVS:

        # ---- Dueling-DQN mean & max ----
        for var in DUELING_VARIANTS:
            for s in SEEDS:
                cmd = [
                    PY, "dueling_dqn.py",
                    "--env", env,
                    "--variant", var,
                    "--seed", str(s),
                    "--episodes", EPISODES,
                    "--save_dir", RESULTS_DIR,
                ]
                run_cmd(cmd)
                time.sleep(0.1)

        # ---- REINFORCE: no baseline + baseline ----
        for baseline in [False, True]:
            for s in SEEDS:
                cmd = [
                    PY, "reinforce.py",
                    "--env", env,
                    "--seed", str(s),
                    "--episodes", EPISODES,
                    "--save_dir", RESULTS_DIR,
                ]
                if baseline:
                    cmd.append("--baseline")
                run_cmd(cmd)
                time.sleep(0.1)

# ------------------------- Plotting Utilities -------------------------

def find_runs(env_name: str, algo: str, variant: str = None) -> List[str]:
    files = []

    # variant-specific pattern first
    if variant:
        pat = os.path.join(RESULTS_DIR, f"{env_name}_{algo}_{variant}_seed*.npy")
        files = glob.glob(pat)
        files.sort()
        if files:
            return files

    # no-variant pattern
    pat = os.path.join(RESULTS_DIR, f"{env_name}_{algo}_seed*.npy")
    files = glob.glob(pat)
    files.sort()
    if files:
        return files

    # fallback
    all_files = glob.glob(os.path.join(RESULTS_DIR, f"{env_name}*"))
    matches = []
    for f in all_files:
        name = os.path.basename(f).lower()
        if algo.lower() in name and "seed" in name:
            matches.append(f)
    matches.sort()
    return matches


def _unwrap_loaded(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1 and isinstance(arr[0], dict):
        d = arr[0]
        for k in ['returns','episode_returns','rewards','ep_returns']:
            if k in d:
                return np.asarray(d[k]).ravel()
    return np.asarray(arr).ravel()


def load_and_stack(filepaths: List[str], expected_len: int):
    runs = []
    used = []

    for f in filepaths:
        try:
            arr = _unwrap_loaded(np.load(f, allow_pickle=True))
            arr = np.asarray(arr, float).ravel()
            if arr.size == 0:
                continue

            used.append(f)

            # pad if shorter
            if len(arr) < expected_len:
                arr = np.pad(arr, (0, expected_len - len(arr)), constant_values=np.nan)
            else:
                arr = arr[:expected_len]

            runs.append(arr)
        except Exception as e:
            print("Warning:", f, "->", e)
            continue

    if len(runs) == 0:
        return np.zeros((0, expected_len))

    print(f"Loaded {len(used)} runs:")
    for f in used:
        print("  ", os.path.basename(f))

    return np.vstack(runs)


def plot_mean_std(data, label, ax):
    if data.size == 0:
        return

    valid = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid]

    mean = np.nanmean(data, axis=0)
    std  = np.nanstd(data, axis=0)
    x = np.flatnonzero(valid)

    ax.plot(x, mean, label=label, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)


def make_comparison_plots(env_name: str):
    Path(PLOTS_DIR).mkdir(exist_ok=True)

    d_mean = find_runs(env_name, "dueling", "mean")
    d_max  = find_runs(env_name, "dueling", "max")

    reinforce_all = find_runs(env_name, "reinforce")
    r_baseline = [f for f in reinforce_all if "baseline" in f.lower()]
    r_nobase   = [f for f in reinforce_all if "nobaseline" in f.lower()]

    print("\n============= FILES FOUND FOR", env_name, "=============")
    print("Dueling (mean):", d_mean)
    print("Dueling (max):", d_max)
    print("REINFORCE (baseline):", r_baseline)
    print("REINFORCE (no baseline):", r_nobase)

    # determine episode length
    max_len = 0
    for f in d_mean + d_max + r_baseline + r_nobase:
        try:
            arr = _unwrap_loaded(np.load(f, allow_pickle=True))
            max_len = max(max_len, len(arr))
        except:
            pass

    if max_len == 0:
        print("No valid files for", env_name)
        return

    # stack runs
    stacks = {
        "Dueling (mean)": load_and_stack(d_mean, max_len),
        "Dueling (max)": load_and_stack(d_max, max_len),
        "REINFORCE (baseline)": load_and_stack(r_baseline, max_len),
        "REINFORCE (no baseline)": load_and_stack(r_nobase, max_len),
    }

    # ---- Create Plot ----
    fig, ax = plt.subplots(figsize=(12,6))

    styles = {
        "Dueling (mean)": "-",
        "Dueling (max)": "--",
        "REINFORCE (baseline)": "-.",
        "REINFORCE (no baseline)": ":",
    }

    for label, arr in stacks.items():
        if arr.size == 0:
            continue
        plot_mean_std(arr, label, ax)
        ax.lines[-1].set_linestyle(styles[label])

    ax.set_title(f"{env_name} — Algorithm Comparison (Mean ± Std)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Return")
    ax.grid(alpha=0.3)
    ax.legend()

    # save
    out = Path(PLOTS_DIR) / f"algo_comparison_{env_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved plot:", out)

    # print numeric summary: mean ± std over last 50 valid episodes per run (averaged across runs)
    def summary_stats(arr):
        if arr.size == 0:
            return None
        valid_cols = ~np.all(np.isnan(arr), axis=0)
        if not valid_cols.any():
            return None
        # take last up-to-50 columns that have any valid data
        valid_idx = np.flatnonzero(valid_cols)
        last_k = min(50, valid_idx.size)
        tail = arr[:, valid_idx[-last_k:]]  # shape (runs, last_k)
        per_run_means = np.nanmean(tail, axis=1)
        return np.nanmean(per_run_means), np.nanstd(per_run_means)

    print("\n--- Summary (mean of per-run last-50 means) ---")
    for label, arr in stacks.items():
        s = summary_stats(arr)
        if s is None:
            print(f"{label}: no data")
        else:
            print(f"{label}: mean(last50) = {s[0]:.2f}  std = {s[1]:.2f}")


# ------------------------- Main -------------------------
if __name__ == "__main__":
    print("=== RUNNING ALL EXPERIMENTS SERIALLY ===")
    run_experiments()

    print("\n=== GENERATING PLOTS ===")
    for env in ENVS:
        make_comparison_plots(env)

    print("\nAll done! Plots saved to:", PLOTS_DIR)
