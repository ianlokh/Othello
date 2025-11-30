import os
import gc
import sys
import re
import importlib

import numpy as np
import pandas as pd
import random
import copy
import pstats

import gymnasium as gym
import tensorflow as tf

from matplotlib import pyplot as plt
from os import path
import glob

# import othello
from othello import othello_agent
from othello.othello_agent import OthelloDQN
from othello import config as cfg

# update inputs
from othello.argparser import ParserOutput

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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

parser = ParserOutput()


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

# set game environment
env_name = "othello:othello-v0"
env = gym.make(env_name, render_mode="human")

# no. of observations
num_observations = env.observation_space['state'].shape[0]
num_actions = env.action_space.n
print(num_observations, num_actions)

# reload the agent class
importlib.reload(sys.modules.get('othello.othello_agent'))


# @profile(stream=fp)
def play(agent: OthelloDQN):
    global agent_win
    global winning_rate
    global best_winning_rate
    global epoch_win_rate_log
    global env_params

    ep_reward = []
    observation, info = env.reset(options=env_params)
    observation = observation["state"].reshape((1, 64))

    done = False
    while not done:
        next_possible_actions = info["next_possible_actions"]

        if info["next_player"]["name"] == "white":
            action = agent.choose_action(observation, next_possible_actions)

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

            ep_reward.append(reward)
        else:
            # playing against a random strategy player
            action = random.choice(list(next_possible_actions))
            action = (action[0] * 8) + action[1]

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

        observation = copy.deepcopy(next_observation)

    if done:
        agent_win.append(True if info["winner"] == "White" else False)

    # this is reward_history for white
    reward_history.append(np.sum(ep_reward))

    # log the winning rate at every epoch_win_rate_log and clean up objects
    if (epoch % epoch_win_rate_log == 0) and (epoch > 1):
        winning_rate.append((epoch, np.mean(is_white)))
        is_white = []
        print("\n***** Epoch: {:d}/{:d}, white player winning rate in last {:d} rounds: {:.2%}. *****".format(epoch,
                                                                                                              EPOCHS,
                                                                                                              epoch_win_rate_log,
                                                                                                              winning_rate[
                                                                                                                  -1][
                                                                                                                  1]))
        # if better winning_rate is found then checkpoint and save model
        if winning_rate[-1][1] >= best_winning_rate:
            best_winning_rate = winning_rate[-1][1]

        # memory cleanup
        n = gc.collect()
        print("\nNumber of unreachable objects collected by GC:{:d}".format(n))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    EPOCHS = cfg.training_param.EPOCHS
    agent_win = []
    winning_rate = []
    best_winning_rate = 0
    rolling_win_rate = []
    epoch_win_rate_log = cfg.training_param.EPOCH_WIN_RATE_LOG
    self_play_update_rate = cfg.training_param.SELF_PLAY_UPDATE_LOG
    epoch_win_rate_log_msg = ""
    env_params = {
        "display_message": ""
    }

    # load models for comparison (todo)
    models = []
    # for file in glob.iglob('{:s}/**/*.h5'.format('/Users/ianlo/PycharmProjects/othello/models'), recursive=True):
    #     print(file)
    [models.append(file) for file in
     glob.iglob('{:s}/**/*'.format('/Users/ianlo/PycharmProjects/othello/models'), recursive=True) if
     file.endswith('.h5')]

    # instantiate agent object as white player in playing mode
    rl_agent = othello_agent.OthelloDQN(nb_observations=64, player="white", mode="play")

    # evaluate models against random player
    for model in models:
        winning_rate = []
        best_winning_rate = 0

        load_status, msg = rl_agent.load_model(path=os.path.dirname(os.path.abspath(model)), name="OthelloDQN",
                                               format_type="model")
        if not load_status:
            print("\nError loading agent:{:s}".format(msg))
            break
        else:
            print("\nRL Agent model loaded:{:s}".format(rl_agent.model_full_path))

        # play for EPOCHS rounds
        for epoch in range(EPOCHS):
            play(rl_agent)

        # save winning rate file
        # curr_date = datetime.now().strftime("%Y_%m_%d")
        model_name = re.search('[0-9]{8}', model).group(0)

        temp_df = pd.DataFrame(winning_rate)
        temp_df.rename({0: 'epochs', 1: 'win_rate'}, axis=1, inplace=True)
        temp_df['mean'] = temp_df['win_rate'].rolling(window=1).mean()
        temp_df.rename({'mean': '{0}-mean'.format(model_name)}, axis=1, inplace=True)

        path = "./models/evaluation/{:s}/".format(model_name)

        # IF no such folder exists, create one automatically
        if not os.path.exists(path):
            os.mkdir(path)

        # open a binary file in write mode
        with open(path + "winning_rate_{:s}".format(model_name), "wb") as file:
            # save array to the file
            np.save(file, winning_rate)
            # close the file
            file.close()

        # open the file in read binary mode
        with open(path + "winning_rate_{:s}".format(model_name), "rb") as file:
            # read the file to numpy array
            winning_rate_df = np.load(file)
            # close the file
            file.close()
            # convert to dataframe
            win_rate_df = pd.DataFrame(winning_rate_df)
            win_rate_df.rename({0: 'epochs', 1: 'win_rate'}, axis=1, inplace=True)
            win_rate_df['mean'] = win_rate_df['win_rate'].rolling(window=10).mean()

            fig, ax = plt.subplots(1, 1)
            win_rate_df.plot(x='epochs', y='win_rate', figsize=(8, 4), ax=ax)
            win_rate_df.plot(x='epochs', y='mean', figsize=(8, 4), ax=ax)
            fig.savefig(path + "winning_rate_{:s}.png".format(model_name), dpi=300)

        print(winning_rate_df)

    # # instantiate agents
    # agent_white = othello_agent.OthelloDQN(nb_observations=64, player="white")
    # # agent_white.model_target.summary()
    #
    # # if train mode is self-play then instantiate another agent for playing against
    # if parser.train_mode == 'self-play':
    #     agent_other = othello_agent.OthelloDQN(nb_observations=64, player="other")
    #
    # EPOCHS = 60000
    # is_white = []
    # reward_history = []
    # winning_rate = []
    # best_winning_rate = 0
    # epoch_win_rate_log = 50
    #
    # for epoch in range(EPOCHS):
    #     train()
    #
    # # save winning rate file
    # curr_date = datetime.now().strftime("%Y_%m_%d")
    # path = "./models/{:s}/".format(curr_date)
    #
    # # IF no such folder exists, create one automatically
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #
    # # open a binary file in write mode
    # with open(path + "winning_rate_{:s}".format(curr_date), "wb") as file:
    #     # save array to the file
    #     np.save(file, winning_rate)
    #     # close the file
    #     file.close
    #
    # # open the file in read binary mode
    # with open(path + "winning_rate_{:s}".format(curr_date), "rb") as file:
    #     # read the file to numpy array
    #     winning_rate = np.load(file)
    #     # close the file
    #     file.close()
    #     # convert to dataframe
    #     win_rate_df = pd.DataFrame(winning_rate)
    #     win_rate_df.rename({0: 'epochs', 1: 'win_rate'}, axis=1, inplace=True)
    #     win_rate_df['mean'] = win_rate_df['win_rate'].rolling(window=10).mean()
    #
    #     fig, ax = plt.subplots(1, 1)
    #     win_rate_df.plot(x='epochs', y='win_rate', figsize=(8, 4), ax=ax)
    #     win_rate_df.plot(x='epochs', y='mean', figsize=(8, 4), ax=ax)
    #     fig.savefig(path + "winning_rate_{:s}.png".format(curr_date), dpi=300)
    #
    # print(winning_rate)
