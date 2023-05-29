import gymnasium as gym
import numpy as np

# Create the MountainCar-v0 environment
env = gym.make('MountainCar-v0')

# Load the trained Q-table
q_table = np.load("mountaincarQ.npy")
print(q_table)

"""Evaluate agent's performance after Q-learning"""
episodes = 100

total_epochs = 0
total_penalties = 0

for _ in range(episodes):
    state = env.reset()[0]
    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    epochs = 0
    penalties = 0
    done = False
    truncate = False
    print(_)
    while not done and not truncate:

        t = state
        action = np.argmax(q_table[state_adj[0], state_adj[1]])
        state2, reward, done, truncate, _ = env.step(action)
        state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
        state2_adj = np.round(state2_adj, 0).astype(int)
        if reward == -1:
            penalties += 1

        state = state2_adj[0]
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")