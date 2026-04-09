import numpy as np
import matplotlib.pyplot as plt
import torch
from simulation import TrafficIntersectionEnv
from agent import DQNAgent
import time

def run_simulation(model_path='traffic_rl/models/traffic_dqn_50.pth'):
    env = TrafficIntersectionEnv()
    state_size = 5
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Running with random agent.")

    state = env.reset()
    done = False
    
    queue_history = []
    
    print("Starting Simulation...")
    print("--------------------------------")
    step = 0
    while not done:
        # Exploit only
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            act_values = agent.model(state_tensor)
        action = torch.argmax(act_values[0]).item()
        
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
        env.render()
        queue_history.append(env.queues.copy())
        
        time.sleep(0.1) # Slow down for visualization
        step += 1
        
        if step > 200: # Limit simulation steps
            break

    # Plot results
    queue_history = np.array(queue_history)
    plt.figure(figsize=(10, 6))
    plt.plot(queue_history[:, 0], label='North Queue')
    plt.plot(queue_history[:, 1], label='East Queue')
    plt.plot(queue_history[:, 2], label='South Queue')
    plt.plot(queue_history[:, 3], label='West Queue')
    plt.xlabel('Time Step')
    plt.ylabel('Queue Length')
    plt.title('Traffic Queue Lengths Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('traffic_rl/simulation_results.png')
    print("Simulation finished. Results saved to traffic_rl/simulation_results.png")

if __name__ == "__main__":
    run_simulation()
