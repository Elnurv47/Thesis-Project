from dqn_agent import DQNAgent
import torch
import torch.nn.functional as F
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import os
import time
from mlagents_envs.exception import UnityTimeOutException
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def train_dqn(agent, env, max_steps=500000, max_steps_per_episode=2000, gamma=0.99, training_name="default"):
    try:
        env.reset()
        behavior_name = list(env.behavior_specs)[0]
        print(f"Connected to behavior: {behavior_name}")

        if not os.path.exists('models'):
            os.makedirs('models')

        best_reward = float('-inf')
        rewards_history = []
        episode_lengths = []
        total_steps = 0
        episode = 0

        time_horizon = 64
        batch_size = 512
        buffer_size = 4096
        learning_rate = 2.5e-4
        beta = 5.0e-3
        epsilon = 0.2
        lambd = 0.95
        num_epoch = 5

        while total_steps < max_steps:
            try:
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                total_reward = 0
                done = False
                steps = 0

                while len(decision_steps.agent_id) == 0:
                    try:
                        env.step()
                        decision_steps, terminal_steps = env.get_steps(behavior_name)
                        time.sleep(0.1)
                    except UnityTimeOutException:
                        print("Unity timeout, retrying...")
                        continue

                agent_id = decision_steps.agent_id[0]
                state = decision_steps[agent_id].obs[0]

                while not done and total_steps < max_steps and steps < max_steps_per_episode:
                    try:
                        action = agent.act(state)
                        action_tuple = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
                        env.set_actions(behavior_name, action_tuple)
                        env.step()

                        decision_steps, terminal_steps = env.get_steps(behavior_name)

                        if agent_id in terminal_steps:
                            next_obs = terminal_steps[agent_id].obs[0]
                            reward = terminal_steps[agent_id].reward
                            done = True
                        else:
                            next_obs = decision_steps[agent_id].obs[0]
                            reward = decision_steps[agent_id].reward

                        next_state = next_obs
                        agent.remember(state, action, reward, next_state, done)
                        agent.train()
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                        total_steps += 1

                        writer.add_scalar('Environment/Step Count', total_steps, total_steps)
                        writer.add_scalar('Environment/Decision Count', total_steps, total_steps)

                    except UnityTimeOutException:
                        print("Unity timeout during episode, retrying...")
                        continue

                if steps >= max_steps_per_episode:
                    done = True
                    print(f"Episode {episode} reached step limit of {max_steps_per_episode}")

                rewards_history.append(total_reward)
                episode_lengths.append(steps)
                episode += 1

                print(f"Episode {episode}, Total Steps: {total_steps}/{max_steps}, Reward: {total_reward}, Epsilon: {agent.epsilon}")

                writer.add_scalar('Environment/Cumulative Reward', total_reward, total_steps)
                writer.add_scalar('Environment/Episode Length', steps, total_steps)
                writer.add_scalar('Environment/Policy/Epsilon', agent.epsilon, total_steps)
                writer.add_scalar('Environment/Policy/Learning Rate', learning_rate, total_steps)
                writer.add_scalar('Environment/Reward Signals/Extrinsic', total_reward, total_steps)

                if total_reward > best_reward:
                    best_reward = total_reward
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f'models/{training_name}_best_{timestamp}.pth'
                    agent.save(model_path)
                    print(f"New best model saved with reward: {best_reward}")

                if total_steps % 100000 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = f'models/{training_name}_checkpoint_{total_steps}_{timestamp}.pth'
                    agent.save(checkpoint_path)
                    print(f"Checkpoint saved at step {total_steps}")

            except Exception as e:
                print(f"Error during episode {episode}: {str(e)}")
                continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'models/{training_name}_final_{timestamp}.pth'
        agent.save(final_model_path)
        writer.close()
        print(f"\nTraining completed!")
        print(f"Training name: {training_name}")
        print(f"Total steps: {total_steps}")
        print(f"Total episodes: {episode}")
        print(f"Best reward achieved: {best_reward}")
        print(f"Final model saved at: {final_model_path}")
        print(f"Average reward over last 100 episodes: {np.mean(rewards_history[-100:]):.2f}")
        print("To view training metrics, run: tensorboard --logdir=logs")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

try:
    print("Attempting to connect to Unity Editor...")
    env = UnityEnvironment(
        file_name=None,
        worker_id=0,
        timeout_wait=30,
        no_graphics=False
    )
    print("Successfully connected to Unity Editor!")

    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    state_size = spec.observation_specs[0].shape[0]
    action_size = spec.action_spec.discrete_branches[0]

    print(f"State size: {state_size}, Action size: {action_size}")

    agent = DQNAgent(state_size, action_size)

    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    training_name = input("Enter a name for this training run (e.g., 'dqn_v1', 'experiment_1'): ").strip()
    if not training_name:
        training_name = f"training_{timestamp}"
    
    writer = SummaryWriter(f'logs/{training_name}_{timestamp}')

    train_dqn(agent, env, max_steps=500000, max_steps_per_episode=2000, training_name=training_name)

except Exception as e:
    print(f"Error connecting to Unity Editor: {str(e)}")
    raise
