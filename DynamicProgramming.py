import numpy as np
import matplotlib.pyplot as plt

# --- Environment Setup ---
grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
actions = ['U', 'D', 'L', 'R']
gamma = 1.0
reward = -1
terminal_states = [(0, 0), (3, 3)]

def next_state(s, a):
    i, j = s
    if s in terminal_states:
        return s
    if a == 'U':
        return (max(i - 1, 0), j)
    if a == 'D':
        return (min(i + 1, grid_size - 1), j)
    if a == 'L':
        return (i, max(j - 1, 0))
    if a == 'R':
        return (i, min(j + 1, grid_size - 1))

# --- Initialize Random Policy and Value ---
policy = {s: np.random.choice(actions) for s in states}
for t in terminal_states:
    policy[t] = 'T'
V = np.zeros((grid_size, grid_size))

# --- Policy Evaluation (Single Sweep) ---
def policy_evaluation(policy, V):
    V_new = np.copy(V)
    for s in states:
        if s in terminal_states:
            continue
        i, j = s
        a = policy[s]
        s_next = next_state(s, a)
        i2, j2 = s_next
        V_new[i, j] = reward + gamma * V[i2, j2]
    return V_new

# --- Policy Improvement ---
def policy_improvement(policy, V):
    stable = True
    for s in states:
        if s in terminal_states:
            continue
        i, j = s
        old_action = policy[s]
        best_action = max(actions, key=lambda a: reward + gamma * V[next_state(s, a)])
        policy[s] = best_action
        if best_action != old_action:
            stable = False
    return stable, policy

# --- Main Policy Iteration Loop ---
iteration = 0
while True:
    iteration += 1
    V = policy_evaluation(policy, V)
    stable, policy = policy_improvement(policy, V)
    if stable:
        break

print(f"✅ Converged after {iteration} iterations")
print("Optimal Value Function:\n", np.round(V, 2))
print("Optimal Policy:")
for i in range(grid_size):
    print([policy[(i, j)] for j in range(grid_size)])

# --- Visualization ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(True)
ax.invert_yaxis()  # (0,0) is top-left

arrow_map = {'U': (0, -0.3), 'D': (0, 0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}

for (i, j) in states:
    if (i, j) in terminal_states:
        ax.text(j, i, 'T', ha='center', va='center', fontsize=16, color='red', fontweight='bold')
    else:
        a = policy[(i, j)]
        dx, dy = arrow_map[a]
        ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
        ax.text(j, i + 0.35, f"{V[i,j]:.1f}", ha='center', va='center', fontsize=10, color='black')

ax.set_title("Optimal Policy (arrows) and State Values")
plt.show()
