import os
import gc
import importlib
import numpy as np
import pandas as pd
import random
import copy
import pstats

from datetime import datetime
from matplotlib import pyplot as plt
from typing import Any

import gymnasium as gym

from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# Keras RL
# from rl.agents.dqn import DQNAgent

# for performance profiling
# import cProfile as cprofile
# from memory_profiler import profile
# fp = open("report-trn.log", "w+")  # to capture memory profile logs

# import othello
from othello import othello_agent
from othello import config as cfg

# command line parsing
from othello.argparser import ParserOutput

parser = ParserOutput()


env_name = "othello:othello-pygame-v0"
env = gym.make(env_name, render_mode="human")

# no. of observations
num_observations = env.observation_space['state'].shape[0]
num_actions = env.action_space.n
print(num_observations, num_actions)

importlib.reload(sys.modules.get('othello.othello_agent'))

# instantiate agents
agent_white = othello_agent.OthelloDQN(nb_observations=64, player="white")
# agent_white.model_target.summary()

# if train mode is curriculum then instantiate another agent for the self-play phase
if parser.train_mode == 'curriculum':
    agent_other = othello_agent.OthelloDQN(nb_observations=64, player="other", mode="play")


# @profile(stream=fp)
def train(curr_epoch: int):
    """
    :param curr_epoch:
    :return:
    """
    global agent_win
    global winning_rate
    global best_winning_rate
    global best_checkpoint_epoch
    # global reward_history
    global epoch_win_rate_log
    global epoch_win_rate_log_msg
    global self_play_update_rate
    global env_params

    # Determine effective opponent mode for this epoch.
    # In 'curriculum' mode the agent trains against random for WARMUP_EPOCHS then switches to self-play.
    if parser.train_mode == 'curriculum':
        effective_mode = 'random' if curr_epoch < cfg.training_param.WARMUP_EPOCHS else 'self-play'
        phase_label = " [warmup]" if effective_mode == 'random' else " [self-play]"
    else:
        effective_mode = parser.train_mode
        phase_label = ""

    # At the warmup→self-play boundary, hard-copy the stable target network weights into agent_other
    # so it starts as a strong opponent rather than an untrained random network.
    # model_target is used (not model_eval) because it is the temporally-smoothed, lower-variance
    # snapshot — the same reason DQN uses a target network for bootstrap targets.
    if parser.train_mode == 'curriculum' and curr_epoch == cfg.training_param.WARMUP_EPOCHS:
        agent_other.model_target.set_weights(agent_white.model_target.get_weights())
        print(f"\n***** Curriculum: warmup complete at epoch {curr_epoch}. Switching to self-play.")
        best_winning_rate = 0.0
        print("***** Checkpoint threshold reset to 0.0 for self-play phase.")

    env_params["display_message_line1"] = f"Epoch: {curr_epoch + 1}/{EPOCHS} | Mode: {parser.train_mode}{phase_label}"

    ep_reward: list[int] = []
    observation, info = env.reset(options=env_params)
    observation = observation["state"].reshape((1, 64))

    done = False
    while not done:
        if env.unwrapped.terminated:
            return

        next_possible_actions = info["next_possible_actions"]

        # move by white player
        if info["next_player"].name == "white":
            action = agent_white.choose_action(observation, next_possible_actions)

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

            agent_white.store_transition(observation, action, reward, done, next_observation)

            if done:
                print("Storing transition for last move by white. ", "Winner:", info["winner"], "Reward:", reward)

            ep_reward.append(reward)
        else:  # move by opponent
            if effective_mode == 'self-play':
                action = agent_other.choose_action(-observation, next_possible_actions)
            else:
                action = random.choice(list(next_possible_actions))
                action = (action[0] * 8) + action[1]

            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation["state"].reshape((1, 64))

            # this is to cater for the case when the last move is by the black player, we want to store the
            # previous move by white that lead to the win/loss
            if done:
                agent_white.reward_transition_update(reward)
                if info["winner"] == "White":
                    print("Storing transition for last move by white. ", "Winner:", info["winner"], "Reward:", reward)
                elif info["winner"] == "Black":
                    print("Storing transition for last move by white. ", "Winner:", info["winner"], "Reward:", reward)
                elif info["winner"] == "Tie":
                    print("Storing transition for last move by white. ", "Winner:", info["winner"], "Reward:", reward)

        observation = copy.deepcopy(next_observation)

    if done:
        print("Plays made by agent:", agent_white.epsilon_plays)
        agent_white.epsilon_plays = []  # reset epsilon plays

        agent_white.learn()  # train agent after each trial
        agent_win.append(True if info["winner"] == "White" else False)

    # this is reward_history for white
    # reward_history.append(np.sum(ep_reward))

    # Periodically hard-copy agent_white's stable target network into agent_other, but only when
    # agent_white is winning enough to confirm it has surpassed the current opponent.
    # - Hard copy (not soft/Polyak) because the win-rate gate already guarantees the new policy is
    #   better; blending with old weights would unnecessarily dilute that improvement.
    # - model_target (not model_eval) because it is the temporally-smoothed, lower-variance snapshot,
    #   producing a more coherent and stable opponent than the noisier online network.
    if (curr_epoch % self_play_update_rate == 0) and (effective_mode == 'self-play'):
        recent_win_rate = winning_rate[-1][1] if winning_rate else 0.0
        threshold = cfg.training_param.SELF_PLAY_UPDATE_WIN_THRESHOLD
        if recent_win_rate >= threshold:
            agent_other.model_target.set_weights(agent_white.model_target.get_weights())
            best_winning_rate = cfg.training_param.SELF_PLAY_UPDATE_WIN_THRESHOLD
            best_checkpoint_epoch = None
            print(f"\n***** Assign weights to self-play agent (win rate {recent_win_rate:.1%} >= {threshold:.1%})")
            print(f"***** Checkpoint threshold reset to {cfg.training_param.SELF_PLAY_UPDATE_WIN_THRESHOLD:.1%} for new opponent generation.")
        else:
            print(f"\n***** Skipped opponent update — win rate {recent_win_rate:.1%} below threshold {threshold:.1%}")

    # log the winning rate at every epoch_win_rate_log and clean up objects
    if (curr_epoch % epoch_win_rate_log == 0) and (curr_epoch > 1):
        win_rate = np.mean(agent_win)
        agent_win = []  # clear array to calculate the rate of win for each window

        phase_flag = 0 if effective_mode == 'random' else 1  # 0=warmup, 1=self-play/other

        # determine checkpoint before appending so the flag is stored in the tuple
        checkpoint_saved = win_rate >= best_winning_rate
        winning_rate.append((curr_epoch, win_rate, phase_flag, 1 if checkpoint_saved else 0))

        print("\n***** Epoch: {:d}/{:d}, Win rate (last {:d}): {:.1%} *****".format(
            curr_epoch, EPOCHS, epoch_win_rate_log, win_rate))

        if checkpoint_saved:
            save_step = "warmup" if effective_mode == 'random' else "self-play"
            agent_white.save_model(name="OthelloDQN", save_step=save_step)
            best_winning_rate = win_rate
            if effective_mode != 'random':
                best_checkpoint_epoch = curr_epoch
            print("\n***** Save model ({}) at Epoch: {:d}/{:d}".format(save_step, curr_epoch, EPOCHS))

        # build compact UI line 2
        if effective_mode == 'random':
            epoch_win_rate_log_msg = "Win: {:.1%} (last {:d}) | Warmup".format(win_rate, epoch_win_rate_log)
        elif best_checkpoint_epoch is None:
            epoch_win_rate_log_msg = "Win: {:.1%} (last {:d}) | No ckpt yet".format(win_rate, epoch_win_rate_log)
        else:
            epoch_win_rate_log_msg = "Win: {:.1%} (last {:d}) | Best: {:.1%} @ep{:d}".format(
                win_rate, epoch_win_rate_log, best_winning_rate, best_checkpoint_epoch)
        env_params["display_message_line2"] = epoch_win_rate_log_msg

        # memory cleanup
        n = gc.collect()
        print("\nNumber of unreachable objects collected by GC:{:d}".format(n))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    EPOCHS = cfg.training_param.EPOCHS
    agent_win = []
    winning_rate = []
    best_winning_rate = 0
    best_checkpoint_epoch = None  # epoch of the best self-play checkpoint; None until first self-play ckpt
    # reward_history = []
    epoch_win_rate_log = cfg.training_param.EPOCH_WIN_RATE_LOG
    self_play_update_rate = cfg.training_param.SELF_PLAY_UPDATE_LOG
    epoch_win_rate_log_msg = ""
    env_params = {
        "display_message_line1": "",
        "display_message_line2": ""
    }

    # train for no. of epochs
    for epoch in range(EPOCHS):
        train(epoch)
        if env.unwrapped.terminated:
            print("\nTraining terminated by user at epoch {:d}.".format(epoch + 1))
            break

    env.close()

    # save final model after training is done
    agent_white.save_model(name="OthelloDQN", save_step="final")

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
        file.close()

    # open the file in read binary mode
    with open(path + "winning_rate_{:s}".format(curr_date), "rb") as file:
        # read the file to numpy array
        winning_rate = np.load(file)
        # close the file
        file.close()
        # convert to dataframe
        win_rate_df = pd.DataFrame(winning_rate)
        win_rate_df.rename({0: 'epochs', 1: 'win_rate', 2: 'phase', 3: 'checkpoint'}, axis=1, inplace=True)
        win_rate_df['mean'] = win_rate_df['win_rate'].rolling(window=10).mean()

        fig, ax = plt.subplots(1, 1)
        ax.set_ylim([0, 1])
        win_rate_df.plot(x='epochs', y='win_rate', figsize=(8, 4), ax=ax)
        win_rate_df.plot(x='epochs', y='mean', figsize=(8, 4), ax=ax)
        ckpt_rows = win_rate_df[win_rate_df['checkpoint'] == 1.0]
        ax.scatter(ckpt_rows['epochs'], ckpt_rows['win_rate'],
                   color='green', marker='^', s=40, zorder=5, label='Checkpoint')
        ax.legend()
        fig.savefig(path + "winning_rate_{:s}.png".format(curr_date), dpi=300)

    print(winning_rate)
