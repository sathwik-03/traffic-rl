# 🚦 Traffic Signal Optimization using Reinforcement Learning (DQN)

## 📌 Overview
This project presents an **AI-driven traffic signal control system** using **Reinforcement Learning (Deep Q-Network)** to optimize traffic flow at a 4-way intersection.

The system dynamically adjusts traffic signals based on real-time vehicle queue lengths, aiming to **minimize congestion and waiting time**, compared to traditional fixed-timer systems.



## 🎯 Problem Statement
Traditional traffic signals operate on fixed timers, which:
- Do not adapt to real-time traffic conditions
- Cause unnecessary waiting time
- Lead to inefficient traffic flow

👉 This project solves this using an **intelligent agent** that learns optimal signal switching.


## 🧠 Methodology

### 🔹 Environment Design
A custom simulation environment is built with:
- 4 lanes: **North, East, South, West**
- State representation:

[queue_N, queue_E, queue_S, queue_W, current_phase]

- Actions:
- `0` → North-South Green
- `1` → East-West Green

---

### 🔹 Reinforcement Learning (DQN)
- Model: Deep Q-Network (DQN)
- Input: Current traffic state
- Output: Optimal signal decision
- Reward Function:

  
Reward = - (Total Queue Length)

👉 Encourages minimizing congestion

---

### 🔹 Training Strategy
- Experience Replay
- Epsilon-Greedy exploration
- Neural Network with hidden layers
- Multiple training episodes

---

## 📊 Features
- ✅ Custom traffic simulation environment
- ✅ AI-based adaptive traffic control
- ✅ Real-time visualization using Matplotlib
- ✅ Performance comparison with baseline system
- ✅ Graphical analysis of results

---

## 🖥️ Visualization

### 🚥 Traffic Signal Behavior
- Green bars → Active lanes (vehicles moving)
- Red bars → Waiting lanes

### 📈 Comparison Output
- Bar graph comparing:
- Traditional system vs AI system
- Metric:
- Total waiting vehicles over time

---

## ⚙️ Installation

```bash
git clone https://github.com/sathwik-03/traffic-rl.git
cd traffic_rl
pip install numpy matplotlib torch


## ▶️ Usage

### 🔹 Run Traffic Simulation (AI Control)

```bash
python visualize.py
```

### 🔹 Compare AI vs Traditional System

```bash
python compare.py
```

### 🔹 Train Model (Optional)

```bash
python train.py
```



## 📈 Results

* AI system dynamically adapts signal timing
* Learns traffic patterns over time
* Performance depends on:

  * Training duration
  * Reward design
  * Traffic randomness

⚠️ Current results:

* Initial model may underperform
* Further training improves efficiency significantly


## 🛠️ Tech Stack

* Python
* NumPy
* PyTorch
* Matplotlib



## 📂 Project Structure

```
traffic_rl/
│
├── simulation.py        # Environment logic
├── agent.py            # DQN model
├── train.py            # Training script
├── visualize.py        # Real-time simulation
├── compare.py          # Performance comparison
├── baseline.py         # Traditional system
│
├── comparison.png      # Result graph
├── README.md           # Documentation
```



## 🚀 Future Improvements

* Improve reward function
* Train for more episodes
* Multi-intersection traffic system
* Integration with real-world traffic data
* Web-based interactive dashboard
* 3D visualization


## 💡 Key Learnings

* Applied Reinforcement Learning to real-world problem
* Designed custom simulation environment
* Built end-to-end AI system with visualization
* Understood reward optimization challenges



## 👤 Author

**Sathwik Akula**
B.Tech in Artificial Intelligence & Machine Learning
Woxsen University, Hyderabad






