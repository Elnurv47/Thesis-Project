import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import glob
import os
from dqn_agent import DQNAgent
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

def visualize_training(agent, env, episodes=10):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'logs/flappy_bird_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view the visualization, run: tensorboard --logdir=logs")
    
    for e in range(episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(list(env.behavior_specs)[0])
        
        total_reward = 0
        done = False
        steps = 0
        
        while len(decision_steps.agent_id) == 0:
            env.step()
            decision_steps, terminal_steps = env.get_steps(list(env.behavior_specs)[0])
        
        agent_id = decision_steps.agent_id[0]
        state = decision_steps[agent_id].obs[0]
        
        while not done:
            action = agent.act(state)
            action_tuple = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
            env.set_actions(list(env.behavior_specs)[0], action_tuple)
            env.step()
            
            decision_steps, terminal_steps = env.get_steps(list(env.behavior_specs)[0])
            
            if agent_id in terminal_steps:
                next_obs = terminal_steps[agent_id].obs[0]
                reward = terminal_steps[agent_id].reward
                done = True
            else:
                next_obs = decision_steps[agent_id].obs[0]
                reward = decision_steps[agent_id].reward
            
            state = next_obs
            total_reward += reward
            steps += 1
            
            writer.add_scalar('Metrics/Reward_per_step', reward, steps)
            writer.add_scalar('Metrics/Cumulative_reward', total_reward, steps)
        
        writer.add_scalar('Episodes/Total_reward', total_reward, e)
        writer.add_scalar('Episodes/Steps', steps, e)
        writer.add_scalar('Episodes/Average_reward_per_step', total_reward/steps if steps > 0 else 0, e)
        
        writer.add_scalar('Agent/Epsilon', agent.epsilon, e)
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}")
    
    writer.close()
    print("\nVisualization complete! Run 'tensorboard --logdir=logs' to view the results.")

def find_latest_model():
    model_files = glob.glob('models/flappy_bird_*.pth')
    if not model_files:
        return None
    return max(model_files, key=os.path.getctime)

if __name__ == "__main__":
    try:
        print("Connecting to Unity Editor...")
        env = UnityEnvironment(file_name=None)
        print("Connected successfully!")
        
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        
        state_size = spec.observation_specs[0].shape[0]
        action_size = spec.action_spec.discrete_branches[0]
        
        print(f"State size: {state_size}, Action size: {action_size}")
        
        agent = DQNAgent(state_size, action_size)
        
        latest_model = find_latest_model()
        if latest_model:
            print(f"\nLoading latest model: {latest_model}")
            agent.load(latest_model)
            print("Model loaded successfully!")
        else:
            print("\nNo saved model found. Using untrained agent.")
        
        visualize_training(agent, env, episodes=10)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
