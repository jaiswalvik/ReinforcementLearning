import matplotlib.pyplot as plt
import networkx as nx
import time

# --- MDP setup ---
states = [0,1,2,3]   # 3 is goal
actions = ['a','b']
gamma = 0.9

# Deterministic transitions for simplicity
P = {
    0: {'a':1, 'b':2},
    1: {'a':2, 'b':3},
    2: {'a':3, 'b':0},
    3: {'a':3, 'b':3}  # goal
}

# Rewards
R = {s:{a:-1 for a in actions} for s in states}
R[3] = {a:0 for a in actions}

# Build a graph for visualization
G = nx.DiGraph()
for s in states:
    for a in actions:
        G.add_edge(s, P[s][a], action=a)

positions = {0:(0,1), 1:(1,2), 2:(1,0), 3:(2,1)}  # fixed positions
n = len(states)

# --- Initialize DP tables ---
V_async = {s:0 for s in states}
V_async[3] = 0
V_rt = {s:0 for s in states}
V_rt[3] = 0

# --- Setup figure ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

def draw_mdp(V, ax, title, color_base):
    ax.clear()
    # Draw nodes with color based on value (higher value = brighter)
    max_val = max([abs(v) for v in V.values()]) or 1
    for s in states:
        intensity = 0.3 + 0.7 * (1 - abs(V[s])/max_val)
        color = (color_base[0]*intensity, color_base[1]*intensity, color_base[2]*intensity)
        nx.draw_networkx_nodes(G, positions, nodelist=[s], node_color=[color], node_size=800, ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, positions, ax=ax, arrows=True)
    # Labels
    labels = {s: f"{s}\n{V[s]:.2f}" for s in states}
    nx.draw_networkx_labels(G, positions, labels, ax=ax)
    ax.set_title(title)

# --- Async DP loop ---
changed = True
iteration = 0
while changed:
    changed = False
    iteration += 1
    for s in states:
        new_val = max(R[s][a] + gamma*V_async[P[s][a]] for a in actions)
        if new_val != V_async[s]:
            V_async[s] = new_val
            changed = True
    draw_mdp(V_async, ax1, f"Async DP Iteration {iteration}", color_base=(0,0,1))  # Blue
    draw_mdp(V_rt, ax2, f"Realtime DP", color_base=(0,1,0))  # Green
    plt.pause(0.5)

# --- Realtime DP loop ---
episodes = 5
for ep in range(episodes):
    s = 0
    traj = [s]
    while s != 3:
        a = max(actions, key=lambda a: R[s][a] + gamma*V_rt[P[s][a]])
        s_next = P[s][a]
        V_rt[s] = R[s][a] + gamma*V_rt[s_next]
        s = s_next
        traj.append(s)
    draw_mdp(V_async, ax1, f"Async DP", color_base=(0,0,1))
    draw_mdp(V_rt, ax2, f"Realtime DP Episode {ep+1}", color_base=(0,1,0))
    plt.pause(0.5)

# Keep window open
plt.ioff()
plt.show()
