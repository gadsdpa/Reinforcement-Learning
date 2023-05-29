import gymnasium as gym
import numpy as np

# Create the Taxi-v3 environment
env = gym.make('Taxi-v3')
q_table = np.load("taxiQ.npy")

"""Evaluate agent's performance after Q-learning"""
episodes = 100

total_epochs = 0
total_penalties = 0

for _ in range(episodes):
    state = env.reset()[0]
    epochs = 0
    penalties = 0
    done = False

    while not done:
        # env.render()
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")