import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from simulation import TrafficIntersectionEnv
from agent import DQN

# ---------------- INIT ----------------
env = TrafficIntersectionEnv()
state = env.reset()

state_size = 5
action_size = 2

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("traffic_rl/models/traffic_dqn_50.pth", map_location=torch.device('cpu')))
model.eval()

fig, ax = plt.subplots()

# ---------------- UPDATE FUNCTION ----------------
def update(frame):
    global state

    # Convert state → tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    # RL decision
    with torch.no_grad():
        rl_action = torch.argmax(model(state_tensor)).item()

    # 🔥 HYBRID CONTROL (stability boost)
    total_queue = np.sum(env.queues)

    if total_queue > 25:
        ns = env.queues[0] + env.queues[2]
        ew = env.queues[1] + env.queues[3]
        action = 0 if ns > ew else 1
    else:
        action = rl_action

    # Step environment
    state, reward, done, _ = env.step(action)

    lanes = env.queues
    signal = env.current_phase

    ax.clear()

    # 🔥 Dynamic colors (cleaner)
    colors = ['green' if i in ([0,2] if signal == 0 else [1,3]) else 'red' for i in range(4)]

    # Plot
    ax.bar(['N','E','S','W'], lanes, color=colors)

    # 🔥 Better title (important)
    total = int(np.sum(lanes))
    ax.set_title(f"Signal: {'NS' if signal==0 else 'EW'} | Total Cars: {total}")

    ax.set_ylim(0, env.max_queue)

    # Debug (optional)
    print(f"Queues: {lanes} | Action: {action} | Reward: {reward}")

    # Reset if done
    if done:
        state = env.reset()

# ---------------- RUN ----------------
ani = FuncAnimation(fig, update, interval=300)
plt.show()