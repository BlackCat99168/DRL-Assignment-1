# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_state(obs):
    """✅ Extracts the state representation from the MiniGrid environment."""
    # TODO: Represent the state using the agent's position and direction.
    return (obs[0], obs[1], obs[10], obs[11], obs[12], obs[13], obs[-2], obs[-1])

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    with open('q_table.pickle', 'rb') as f:
        q_table = pickle.load(f)
    state = get_state(obs)
    action = 0
    if state in q_table:
        if np.random.rand() < 0.000001:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_table[state])
    else:
        action = random.choice([0, 1, 2, 3, 4, 5])
    return action # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

