import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to the TensorBoard event file in training_1
logdir = 'logs/dqn_training_1_20250512_042116'
files = os.listdir(logdir)
event_file = None
for f in files:
    if f.startswith('events.out.tfevents'):
        event_file = os.path.join(logdir, f)
        break

if event_file is None:
    print('No TensorBoard event file found in training_1')
    exit(1)

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
    exit(1)

# Extract the scalar values
scalars = acc.Scalars(cum_reward_tag)
data = [(s.step, s.value) for s in scalars]
df = pd.DataFrame(data, columns=['step', 'value'])

# Save to CSV
csv_path = os.path.join(logdir, 'recovered_Environment_Cumulative Reward.csv')
df.to_csv(csv_path, index=False)
print(f'Cumulative reward saved to {csv_path}') 