import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

RESULTS_DIR = "results"

def find_runs(env_name: str, algo: str, variant: str = None) -> List[str]:
    files = []
    if variant:
        pat = os.path.join(RESULTS_DIR, f"{env_name}_{algo}_{variant}_seed*.npy")
        files = glob.glob(pat)
        files.sort()
        if files:
            return files

    pat = os.path.join(RESULTS_DIR, f"{env_name}_{algo}_seed*.npy")
    files = glob.glob(pat)
    files.sort()
    if files:
        return files

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

def load_and_stack(filepaths: List[str], expected_episodes: int):
    runs = []
    used = []
    for f in filepaths:
        try:
            arr = np.load(f, allow_pickle=True)
            arr = _unwrap_loaded(arr)
            arr = np.asarray(arr, float).ravel()
            if arr.size == 0:
                continue
            used.append(f)
            if len(arr) < expected_episodes:
                arr = np.pad(arr, (0, expected_episodes - len(arr)), constant_values=np.nan)
            else:
                arr = arr[:expected_episodes]
            runs.append(arr)
        except:
            continue

    if len(runs) == 0:
        return np.zeros((0, expected_episodes))

    print(f"Loaded {len(used)} runs:")
    for f in used:
        print("   ", os.path.basename(f))

    return np.vstack(runs)

def plot_mean_std(data, label, ax):
    valid = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid]

    mean = np.nanmean(data, axis=0)
    std  = np.nanstd(data, axis=0)
    x = np.flatnonzero(valid)

    ax.plot(x, mean, label=label, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)

def make_comparison_plots(env_name: str, save_dir: str="plots"):
    Path(save_dir).mkdir(exist_ok=True)

    # --- find files ---
    d_mean = find_runs(env_name, "dueling", "mean")
    d_max  = find_runs(env_name, "dueling", "max")

    reinforce_all = find_runs(env_name, "reinforce")

    # --- group reinforce ---
    r_baseline = [f for f in reinforce_all if "baseline" in os.path.basename(f).lower()]
    r_nobase   = [f for f in reinforce_all if "nobaseline" in os.path.basename(f).lower()]

    print("\n--- FOUND FILES ---")
    print("Dueling mean:", d_mean)
    print("Dueling max :", d_max)
    print("REINFORCE baseline:", r_baseline)
    print("REINFORCE no baseline:", r_nobase)

    # determine max episode length
    all_files = d_mean + d_max + r_baseline + r_nobase
    max_len = 0
    for f in all_files:
        arr = _unwrap_loaded(np.load(f, allow_pickle=True))
        max_len = max(max_len, len(arr))

    # load stacks
    stacks = {
        "Dueling (mean)": load_and_stack(d_mean, max_len),
        "Dueling (max)": load_and_stack(d_max, max_len),
        "REINFORCE (baseline)": load_and_stack(r_baseline, max_len),
        "REINFORCE (no baseline)": load_and_stack(r_nobase, max_len),
    }

    # plotting
    fig, ax = plt.subplots(figsize=(12,6))

    style = {
        "Dueling (mean)": "-",
        "Dueling (max)": "--",
        "REINFORCE (baseline)": "-.",
        "REINFORCE (no baseline)": ":"
    }

    for label, arr in stacks.items():
        if arr.size == 0:
            print(f"Skipping {label} (no data)")
            continue
        plot_mean_std(arr, label, ax)
        ax.lines[-1].set_linestyle(style[label])

    ax.set_title(f"{env_name} — Algorithm comparison (mean ± std)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Return")
    ax.legend()
    ax.grid(alpha=0.3)

    out = Path(save_dir) / f"algo_comparison_{env_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    for env in ["CartPole-v1", "Acrobot-v1"]:
        make_comparison_plots(env)
