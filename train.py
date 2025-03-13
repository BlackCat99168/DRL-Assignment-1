import simple_custom_taxi_env as env_lib
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def tabular_q_learning(env_name="MiniGrid-Empty-8x8-v0", episodes=5000, alpha=0.1, gamma=0.99,
                       epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.999):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Implementing Tabular Q-Learning with Epsilon Decay
    - Uses a **Q-table** to store action values for each state.
    - Updates Q-values using the **Bellman equation**.
    - Implements **ε-greedy exploration** for action selection.
    """
    
    # TODO: Initialize an empty Q-table to store state-action values.
    Q_table = {}

    rewards_per_episode = []
    # TODO: Initialize epsilon for the exploration-exploitation tradeoff.
    epsilon = epsilon_start

    def get_state(obs):
        """✅ Extracts the state representation from the MiniGrid environment."""
        # TODO: Represent the state using the agent's position and direction.
        return (obs[0], obs[1], obs[10], obs[11], obs[12], obs[13])

    for episode in range(episodes):
        # TODO: Reset the environment at the beginning of each episode.
        env_size = random.choice([5,6,7,8,9,10])
        env = env_lib.SimpleTaxiEnv(grid_size=env_size, fuel_limit=5000)
        obs, info = env.reset()
        state = get_state(obs)
        done = False
        total_reward = 0

        while not done:
            # TODO: Initialize the state in the Q-table if it is not already present.
            if state not in Q_table:
                Q_table[state] = np.zeros(6)
            # TODO: Implement an ε-greedy policy for action selection.
            if np.random.rand() < epsilon:
              action = random.choice([0, 1, 2, 3, 4, 5])
            else:
              action = np.argmax(Q_table[state])

            # TODO: Execute the action and observe the next state and reward.
            obs, reward, done, truncated = env.step(action)
            next_state = get_state(obs)
            total_reward += reward

            # TODO: Initialize next_state in the Q-table if it is not already present.
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(6)
            # TODO: Apply the Q-learning update rule (Bellman equation).
            #print(reward)
            Q_table[state][action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state][action])

            # TODO: Update the state to the next state.
            state = next_state
            #print(Q_table)

        rewards_per_episode.append(total_reward)

        # TODO: Decay epsilon over time to gradually reduce exploration.
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 1 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")

    return Q_table, rewards_per_episode

q_table, rewards = tabular_q_learning("MiniGrid-Empty-8x8-v0", episodes=5000)

with open('q_table.pickle', 'wb') as f:
    pickle.dump(q_table, f)
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Training Progress")
plt.show()