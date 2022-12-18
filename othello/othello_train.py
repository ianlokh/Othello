import sys
import os
import gc
import importlib
import numpy as np
import random
import copy
import pstats
from datetime import datetime

# import gym
import gymnasium as gym
import tensorflow as tf

# setting this to ensure that we can reproduce the results
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# setting log level for tensorflow
tf.get_logger().setLevel('ERROR')

# setting OS variables for Tensorflow
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4'  # if not hvd_utils.is_using_hvd() else str(hvd.size())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Keras RL
# from rl.agents.dqn import DQNAgent

# for performance profiling
# import cProfile as cprofile
# from memory_profiler import profile
# fp = open("report-trn.log", "w+")  # to capture memory profile logs

# import othello
from othello import othello_agent


# @profile(stream=fp)
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

# no. of observations
num_observations = env.observation_space['state'].shape[0]
num_actions = env.action_space.n
print(num_observations, num_actions)

importlib.reload(sys.modules.get('othello.othello_agent'))

agent_white = othello_agent.OthelloDQN(nb_observations=64, player="white")
# agent_white.model_target.summary()


# @profile(stream=fp)
def train():
    global is_white
    global winning_rate
    global best_winning_rate
    global reward_history

    ep_reward = []
    observation, info = env.reset()
    observation = observation["state"].reshape((1, 64))

    done = False
    while not done:
        next_possible_actions = info["next_possible_actions"]

        if info["next_player"]["name"] == "white":
            action = agent_white.choose_action(observation, next_possible_actions)

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

            agent_white.store_transition(observation, action, reward, done, next_observation)

            if done:
                print("Storing transition for last move by white. ", "Winner:", info["winner"], "Reward:", reward)

            ep_reward.append(reward)
        else:
            action = random.choice(list(next_possible_actions))
            action = (action[0] * 8) + action[1]

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

            # this is to cater for the case when the last move is by the black player, we want to store the
            # previous move by white that lead to the win/loss
            if done:
                if info["winner"] == "White":
                    print("Storing transition for last move by black. ", "Winner:", info["winner"], "Reward:", 10)
                    agent_white.reward_transition_update(10)
                elif info["winner"] == "Black":
                    print("Storing transition for last move by black. ", "Winner:", info["winner"], "Reward:", -10)
                    agent_white.reward_transition_update(-10)
                elif info["winner"] == "Tie":
                    print("Storing transition for last move by black. ", "Winner:", info["winner"], "Reward:", 2)
                    agent_white.reward_transition_update(2)

        observation = copy.deepcopy(next_observation)

    if done:
        agent_white.learn()  # train agent after each trial
        is_white.append(True if info["winner"] == "White" else False)

    # this is reward_history for white
    reward_history.append(np.sum(ep_reward))

    if (epoch % 50 == 0) and (epoch > 1):  # log winning rate in every 50 eps
        winning_rate.append((epoch, np.mean(is_white)))
        is_white = []
        print("\n***** Epoch: {:d}/{:d}, white player winning rate in latest 30 rounds: {:.2%}. *****\n".format(epoch, EPOCHS, winning_rate[-1][1]))

        if winning_rate[-1][1] >= best_winning_rate:
            agent_white.save_model(name="OthelloDQN")
            print("\n***** Save model at Epoch: {:d}/{:d}\n".format(epoch, EPOCHS))
            best_winning_rate = winning_rate[-1][1]

        # memory cleanup
        n = gc.collect()
        print("\nNumber of unreachable objects collected by GC:", n, "\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    EPOCHS = 40000
    is_white = []
    reward_history = []
    winning_rate = []
    best_winning_rate = 0

    for epoch in range(EPOCHS):
        train()

    # save winning rate file
    curr_date = datetime.now().strftime("%Y_%m_%d")
    path = "./models/{:s}/".format(curr_date)

    # IF no such folder exists, create one automatically
    if not os.path.exists(path):
        os.mkdir(path)

    # open a binary file in write mode
    with open(path + "winning_rate_{:s}".format(curr_date), "wb") as file:
        # save array to the file
        np.save(file, winning_rate)
        # close the file
        file.close

    # open the file in read binary mode
    with open(path + "winning_rate_{:s}".format(curr_date), "rb") as file:
        # read the file to numpy array
        winning_rate = np.load(file)
        # close the file
        file.close()

    print(winning_rate)