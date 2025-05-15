import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to the PPO training directories
ppo_dirs = ['PPO_CSV/main_ppo_1', 'PPO_CSV/main_ppo_2', 'PPO_CSV/main_ppo_3', 'PPO_CSV/main_ppo_4']

for ppo_dir in ppo_dirs:
    # Find the TensorBoard event file
    event_file = None
    for root, dirs, files in os.walk(ppo_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                break
        if event_file:
            break

    if event_file is None:
        print(f'No TensorBoard event file found in {ppo_dir}')
        continue

    # Load the event file
    acc = EventAccumulator(event_file)
    acc.Reload()

    # Find the tag for cumulative reward
    # List all tags
    print('Available tags:', acc.Tags()['scalars'])

    # Try to find a tag containing 'Cumulative Reward'
    cum_reward_tag = None
    for tag in acc.Tags()['scalars']:
        if 'Cumulative Reward' in tag:
            cum_reward_tag = tag
            break

    if cum_reward_tag is None:
        print('No cumulative reward tag found!')
        continue

    # Extract the scalar values
    scalars = acc.Scalars(cum_reward_tag)
    data = [(s.step, s.value) for s in scalars]
    df = pd.DataFrame(data, columns=['step', 'value'])

    # Save to CSV
    csv_path = os.path.join(ppo_dir, 'recovered_Environment_Cumulative Reward.csv')
    df.to_csv(csv_path, index=False)
    print(f'Cumulative reward saved to {csv_path}') 