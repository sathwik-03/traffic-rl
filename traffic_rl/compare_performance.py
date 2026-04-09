import numpy as np
import matplotlib.pyplot as plt
import torch
from simulation import TrafficIntersectionEnv
from agent import DQNAgent
from baseline import FixedTimeAgent
import random

def run_episode(agent, label, seed=42):
    # Set seed for reproducibility/fairness
    np.random.seed(seed)
    random.seed(seed)
    
    env = TrafficIntersectionEnv()
    state = env.reset()
    done = False
    total_queue = []
    
    while not done:
        if isinstance(agent, DQNAgent):
            state_tensor = torch.FloatTensor(np.array([state])).to(agent.device)
            with torch.no_grad():
                act_values = agent.model(state_tensor)
            action = torch.argmax(act_values[0]).item()
        else:
            action = agent.act(state)
            
        next_state, _, done, _ = env.step(action)
        state = next_state
        total_queue.append(np.sum(env.queues))
        
    return total_queue

def main():
    # Load RL Agent
    rl_agent = DQNAgent(state_size=5, action_size=2)
    try:
        rl_agent.load('traffic_rl/models/traffic_dqn_50.pth')
        print("Loaded RL model.")
    except:
        print("RL model not found, using random weights (expect poor performance).")

    # baseline Agent
    fixed_agent = FixedTimeAgent(switch_interval=20) # Switch every 20 steps

    print("Running comparison...")
    simulation_seed = 123
    rl_queues = run_episode(rl_agent, "RL Agent", seed=simulation_seed)
    fixed_queues = run_episode(fixed_agent, "Fixed-Time", seed=simulation_seed)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(rl_queues, label='RL Agent (DQN)', alpha=0.9)
    plt.plot(fixed_queues, label='Fixed-Time Baseline', alpha=0.9, linestyle='--')
    
    # Highlight Rush Hour
    plt.axvspan(200, 400, color='red', alpha=0.1, label='Rush Hour')
    plt.axvspan(600, 800, color='red', alpha=0.1)
    
    plt.xlabel('Time Step')
    plt.ylabel('Total Queue Length (All Lanes)')
    plt.title('Traffic Control Comparison: RL vs Fixed-Time (with Rush Hour)')
    plt.legend()
    plt.grid(True)
    plt.savefig('traffic_rl/comparison_results.png')
    print("Comparison saved to traffic_rl/comparison_results.png")

if __name__ == "__main__":
    main()
