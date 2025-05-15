import matplotlib.pyplot as plt
import numpy as np

dqn_rewards = [699.96, 1185.95, 160.90, 1975.47]
ppo_rewards = [127.11, 22.19, 22.96, 22.63]

means = [np.mean(dqn_rewards), np.mean(ppo_rewards)]
stds = [np.std(dqn_rewards), np.std(ppo_rewards)]

plt.figure(figsize=(6, 5))
plt.bar(['DQN', 'PPO'], means, yerr=stds, capsize=10, color=['skyblue', 'salmon'])
plt.yscale('log')
plt.ylabel('Final Average Reward (log scale)')
plt.title('Mean Final Reward (± Std) for DQN and PPO (Log Scale)')
plt.tight_layout()
plt.savefig('dqn_ppo_barplot_log.png', dpi=300)
plt.close()