import gymnasium as gym
import numpy as np
from IPython.display import clear_output

# Create the Taxi-v3 environment
env = gym.make('Taxi-v3', render_mode="ansi")
env.reset()
env.render()

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
num_episodes = 50000

# Track metrics
all_epochs = []
all_penalties = []

# Training loop
for episode in range(1, num_episodes + 1):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _,_ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    all_epochs.append(epochs)
    all_penalties.append(penalties)

    if episode % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {episode}/{num_episodes}")

print("Training finished.\n")

# Save Q-table
np.save('taxiQ.npy', q_table)