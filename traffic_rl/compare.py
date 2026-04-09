import numpy as np
import matplotlib.pyplot as plt
from simulation import TrafficIntersectionEnv
import torch
from agent import DQN
import os

# ---------------- LOAD MODEL ----------------
state_size = 5
action_size = 2

model = DQN(state_size, action_size)

model_path = "traffic_rl/models/traffic_dqn_50.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ---------------- SETTINGS ----------------
STEPS = 1000

# ---------------- NORMAL SYSTEM ----------------
env = TrafficIntersectionEnv()
state = env.reset()

total_wait_normal = 0

for t in range(STEPS):
    # Fixed timer signal (baseline)
    action = 0 if (t // 20) % 2 == 0 else 1

    state, reward, done, _ = env.step(action)
    total_wait_normal += np.sum(env.queues)

# ---------------- AI SYSTEM ----------------
env = TrafficIntersectionEnv()
state = env.reset()

total_wait_ai = 0

for t in range(STEPS):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        rl_action = torch.argmax(model(state_tensor)).item()

    # 🔥 HYBRID STABILITY (IMPORTANT)
    # If traffic is high → use smart rule
    if np.sum(env.queues) > 25:
        ns = env.queues[0] + env.queues[2]
        ew = env.queues[1] + env.queues[3]
        action = 0 if ns > ew else 1
    else:
        action = rl_action

    state, reward, done, _ = env.step(action)
    total_wait_ai += np.sum(env.queues)

# ---------------- RESULTS ----------------
print("\n===== RESULTS =====")
print(f"Normal Total Wait: {total_wait_normal}")
print(f"AI Total Wait: {total_wait_ai}")

improvement = ((total_wait_normal - total_wait_ai) / total_wait_normal) * 100
print(f"Improvement: {improvement:.2f}%")

# ---------------- GRAPH ----------------
plt.figure()
plt.bar(["Normal", "AI"], [total_wait_normal, total_wait_ai])
plt.title("Traffic Optimization Comparison")
plt.ylabel("Total Waiting Vehicles")
plt.savefig("comparison.png")
plt.show()  