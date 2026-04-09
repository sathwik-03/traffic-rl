# 🚦 Traffic Signal Optimization using Reinforcement Learning

## 📌 Problem
Traditional traffic signals use fixed timers, which lead to inefficient traffic flow and increased congestion.

---

## 💡 Solution
This project uses a Reinforcement Learning (RL) agent to dynamically control traffic signals based on real-time traffic conditions.

---

## ⚙️ Approach

### State
- Queue length of 4 lanes (N, E, S, W)
- Current signal phase

### Action
- 0 → North-South Green
- 1 → East-West Green

### Reward
- Minimize total waiting vehicles
- Encourage faster traffic clearance

---

## 🧠 Model
- Deep Q-Network (DQN)
- Implemented using PyTorch

---

## 📊 Results

The RL-based system is compared with a traditional fixed-timing signal.

- Fixed system uses alternating signal every few steps
- RL agent adapts based on traffic density

📌 Result:

![Comparison](comparison.png)

---

## 🎥 Visualization

The system provides real-time visualization of traffic queues and signal states.

- Green → Active lanes
- Red → Waiting lanes
- Displays total vehicles dynamically

---

## 🚀 How to Run

```bash
cd traffic_rl

# Run simulation
python3 visualize.py

# Run comparison
python3 compare.py