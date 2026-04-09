import numpy as np
from simulation import TrafficIntersectionEnv
from agent import DQNAgent
import os

def main():
    env = TrafficIntersectionEnv()
    state_size = 5 # 4 queues + 1 phase
    action_size = 2 # NS Green, EW Green
    agent = DQNAgent(state_size, action_size)
    
    episodes = 50
    batch_size = 32
    output_dir = 'traffic_rl/models'
    os.makedirs(output_dir, exist_ok=True)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                
            agent.replay(batch_size)
        
        if (e + 1) % 10 == 0:
            agent.save(f"{output_dir}/traffic_dqn_{e+1}.pth")

if __name__ == "__main__":
    main()
