import simple_custom_taxi_env as env_lib
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def get_state(obs):
    """✅ Extracts the state representation from the MiniGrid environment."""
    # TODO: Represent the state using the agent's position and direction.
    return (obs[0], obs[1], obs[10], obs[11], obs[12], obs[13], obs[-2], obs[-1])

def get_action(obs, q_table):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_state(obs)
    action = 0
    if state in q_table:
        if np.random.rand() < 0.00001:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_table[state])
    else:
        action = random.choice([0, 1, 2, 3, 4, 5])
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

def tabular_q_learning(env_name="MiniGrid-Empty-8x8-v0", episodes=5000, alpha=0.1, gamma=0.99,
                       epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.999, q_table=[]):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    """
    ✅ Implementing Tabular Q-Learning with Epsilon Decay
    - Uses a **Q-table** to store action values for each state.
    - Updates Q-values using the **Bellman equation**.
    - Implements **ε-greedy exploration** for action selection.
    """

    rewards_per_episode = []
    # TODO: Initialize epsilon for the exploration-exploitation tradeoff.
    epsilon = epsilon_start

    def get_state(obs):
        """✅ Extracts the state representation from the MiniGrid environment."""
        # TODO: Represent the state using the agent's position and direction.
        return (obs[0], obs[1], obs[10], obs[11], obs[12], obs[13], obs[-2], obs[-1])

    cnt = 0
    score = 0
    for episode in range(episodes):
        # TODO: Reset the environment at the beginning of each episode.
        env_size = random.choice([5,6,7,8,9,10])
        env = env_lib.SimpleTaxiEnv(grid_size=env_size, fuel_limit=5000)
        obs, info = env.reset()
        state = get_state(obs)
        done = False
        total_reward = 0

        pick_flag = 0
        length = 0
        pick_cnt = 0
        drop_cnt = 0
        while not done:
            length += 1
            # TODO: Initialize the state in the Q-table if it is not already present.
            # TODO: Implement an ε-greedy policy for action selection.
            action = get_action(obs, q_table)

            # TODO: Execute the action and observe the next state and reward.
            obs, reward, done, truncated = env.step(action)
            next_state = get_state(obs)
            total_reward += reward

            if length < 4998 and done:
                cnt += 1
            # TODO: Update the state to the next state.
            state = next_state
            #print(Q_table)

        rewards_per_episode.append(total_reward)

        # TODO: Decay epsilon over time to gradually reduce exploration.
        epsilon = max(epsilon_end, epsilon * decay_rate)
        score += total_reward
        if (episode + 1) % 1 == 0:
            avg_reward = np.mean(rewards_per_episode[-1:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            print()
    print("Success Count:", cnt)
    print("Score:", score / episodes)

    return q_table, rewards_per_episode

with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)
tabular_q_learning("MiniGrid-Empty-8x8-v0", episodes=50, q_table=q_table)