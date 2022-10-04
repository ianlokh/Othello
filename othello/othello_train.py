import numpy as np
import random
import os
import copy

import gym
import tensorflow as tf

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4'  # if not hvd_utils.is_using_hvd() else str(hvd.size())


def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            for gpu in gpus_used:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


set_gpu([0])

env_name = "othello:othello-v0"
env = gym.make(env_name, render_mode="human")
env.observation_space['state']

# 4 observations
num_observations = env.observation_space['state'].shape[0]
num_actions = env.action_space.n
print(num_observations, num_actions)

# import othello
from othello import othello_agent
import importlib
import sys

importlib.reload(sys.modules.get('othello.othello_agent'))

agent_white = othello_agent.OthelloDQN(nb_observations=64, player="white")
agent_white.model_target.summary()

EPOCHS = 500

best_so_far = 0
points = 0

for epoch in range(EPOCHS):

    observation, info = env.reset()
    observation = observation["state"].reshape((1, 64))
    next_possible_actions = info["next_possible_actions"]

    done = False

    while not done:

        if info["next_player"]["name"] == "white":
            action = agent_white.choose_action(observation, next_possible_actions)

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))
            next_possible_actions = info["next_possible_actions"]

            agent_white.store_transition(observation, action, reward, done, next_observation)
        else:
            action = random.choice(list(next_possible_actions))
            action = (action[0] * 8) + action[1]

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))
            next_possible_actions = info["next_possible_actions"]

        observation = copy.deepcopy(next_observation)
        points += reward

    hist = agent_white.learn()

    if not (hist is None):
        print("Loss: ", ['%.4f' % elem for elem in hist.history['loss']],
              "Accuracy:", ['%.4f' % elem for elem in hist.history['accuracy']],
              "Learn Step Cnt:", agent_white.learn_step_counter)

    if points > best_so_far:
        best_so_far = points

    if epoch % 5 == 0:
        print(f"{epoch}: POINTS: {points} eps: {agent_white.epsilon} BSF: {best_so_far}")

# In[ ]:


# In[ ]:




