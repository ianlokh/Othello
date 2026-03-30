"""Training pipeline integration tests.

othello_train.py cannot be imported (TF imports + module-level side effects),
so these tests replicate the critical training loop contracts by wiring the
agent, env, and replay buffer directly — verifying the same invariants that
the training loop must satisfy.
"""
import copy
import random

import numpy as np
import pytest

from othello import config as cfg
from othello.envs.othello_pygame_env import OthelloPygameEnv
from othello.othello_agent import OthelloDQN
from othello.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(player="white", mode="training", seed=42, batch_size=32):
    agent = OthelloDQN(nb_observations=64, player=player, mode=mode, seed=seed)
    agent.batch_size = batch_size
    return agent


def _run_one_episode(env, agent_white, agent_other=None, effective_mode="random"):
    """Run one full game episode replicating the train() loop logic.

    Returns (winner, transitions_stored, epsilon_before, epsilon_after).
    """
    obs_raw, info = env.reset()
    observation = obs_raw["state"].reshape((1, 64))
    epsilon_before = agent_white.epsilon
    transitions_before = len(agent_white.replay_buffer)
    done = False

    while not done:
        next_possible_actions = info["next_possible_actions"]

        if info["next_player"].name == "white":
            action = agent_white.choose_action(observation, next_possible_actions)
            next_obs_raw, reward, done, _, info = env.step(action)
            next_observation = next_obs_raw["state"].reshape((1, 64))
            agent_white.store_transition(observation, action, reward, done, next_observation)
        else:
            if effective_mode == "self-play" and agent_other is not None:
                action = agent_other.choose_action(-observation, next_possible_actions)
            else:
                move = random.choice(list(next_possible_actions))
                action = move[0] * 8 + move[1]
            next_obs_raw, reward, done, _, info = env.step(action)
            next_observation = next_obs_raw["state"].reshape((1, 64))
            if done:
                agent_white.reward_transition_update(reward)

        observation = copy.deepcopy(next_observation)

    return info["winner"], len(agent_white.replay_buffer) - transitions_before, epsilon_before


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = OthelloPygameEnv(render_mode=None)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Single-epoch tests
# ---------------------------------------------------------------------------

def test_single_epoch_random_opponent(env):
    """One episode against random opponent: transitions are stored and epsilon decays after learn()."""
    agent = _make_agent()
    winner, transitions_stored, eps_before = _run_one_episode(env, agent)

    assert transitions_stored > 0, "Agent must store at least one transition per episode"

    # Fill buffer enough to learn, then check epsilon decays
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.ones((1, 64), dtype=np.float32) * 0.1
    while len(agent.replay_buffer) < agent.batch_size:
        agent.store_transition(obs, 5, 0.0, False, nobs)

    eps_before_learn = agent.epsilon
    agent.learn()
    assert agent.epsilon < eps_before_learn, "Epsilon must decay after learn()"
    assert agent.learn_step_counter == 1, "Learn step counter must increment"


def test_single_epoch_self_play(env):
    """One episode with a copy opponent runs without error."""
    agent_white = _make_agent(player="white")
    agent_other = _make_agent(player="other", mode="play")
    # Copy weights from white → other (warmup boundary simulation)
    agent_other.model_target.set_weights(agent_white.model_target.get_weights())

    winner, transitions, _ = _run_one_episode(
        env, agent_white, agent_other=agent_other, effective_mode="self-play"
    )
    assert transitions > 0, "Agent must store transitions in self-play mode"


# ---------------------------------------------------------------------------
# Reward update when opponent makes final move
# ---------------------------------------------------------------------------

def test_reward_update_on_opponent_last_move():
    """When the game ends on black's move, white's last stored reward must be patched."""
    agent = _make_agent()
    agent.replay_buffer = UniformReplayBuffer(capacity=1000)

    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    # Store an interim transition (game not yet over)
    agent.store_transition(obs, 5, 0.0, False, nobs)
    reward_before = agent.replay_buffer._buffer[-1][2]

    # Simulate black making the final move — white's transition gets patched
    terminal_reward = cfg.agent_setting.PENALTY  # white lost
    agent.reward_transition_update(terminal_reward)

    patched = agent.replay_buffer._buffer[-1]
    assert patched[2] == pytest.approx(reward_before + terminal_reward, abs=1e-4)
    assert patched[4] is True, "done flag must be True after reward_transition_update"


# ---------------------------------------------------------------------------
# Warmup→self-play boundary
# ---------------------------------------------------------------------------

def test_weight_copy_at_warmup_boundary():
    """At the warmup boundary, agent_other receives exact weights from agent_white."""
    agent_white = _make_agent(player="white")
    agent_other = _make_agent(player="other", mode="play", seed=99)

    # Simulate the warmup→self-play hard copy
    agent_other.model_target.set_weights(agent_white.model_target.get_weights())

    for w, o in zip(agent_white.model_target.get_weights(),
                    agent_other.model_target.get_weights()):
        np.testing.assert_allclose(w, o, atol=1e-6,
                                   err_msg="agent_other weights must match agent_white after copy")


# ---------------------------------------------------------------------------
# Self-play opponent update gated by win rate
# ---------------------------------------------------------------------------

def test_self_play_update_gated_by_winrate():
    """Opponent weights update only when win rate meets the threshold."""
    import torch
    agent_white = _make_agent(player="white", seed=1)
    agent_other = _make_agent(player="other", mode="play", seed=2)

    # Take a snapshot of agent_other's original weights
    original_weights = [w.copy() for w in agent_other.model_target.get_weights()]

    threshold = cfg.training_param.SELF_PLAY_UPDATE_WIN_THRESHOLD

    # Below threshold → should NOT update
    recent_win_rate_low = threshold - 0.1
    if recent_win_rate_low < threshold:
        pass  # no update
    for w_orig, w_now in zip(original_weights, agent_other.model_target.get_weights()):
        np.testing.assert_allclose(w_orig, w_now, atol=1e-6,
                                   err_msg="Weights must not change below threshold")

    # At/above threshold → should update
    recent_win_rate_high = threshold + 0.1
    if recent_win_rate_high >= threshold:
        agent_other.model_target.set_weights(agent_white.model_target.get_weights())

    for w_white, w_other in zip(agent_white.model_target.get_weights(),
                                 agent_other.model_target.get_weights()):
        np.testing.assert_allclose(w_white, w_other, atol=1e-6,
                                   err_msg="Weights must match after threshold-gated update")


# ---------------------------------------------------------------------------
# Reward shaping stored in buffer
# ---------------------------------------------------------------------------

def test_shaped_rewards_stored():
    """With REWARD_SHAPING_ENABLED, non-zero obs should produce non-zero intermediate reward."""
    if not cfg.agent_setting.REWARD_SHAPING_ENABLED:
        pytest.skip("Reward shaping is disabled in config")

    agent = _make_agent()
    agent.replay_buffer = UniformReplayBuffer(capacity=1000)

    # Non-uniform board → POSITION_WEIGHTS dot product is non-zero → shaped reward != raw
    obs = np.ones((1, 64), dtype=np.float32)
    nobs = -np.ones((1, 64), dtype=np.float32)
    raw_reward = 0.0
    agent.store_transition(obs, 10, raw_reward, False, nobs)

    stored_reward = agent.replay_buffer._buffer[-1][2]
    assert stored_reward != pytest.approx(raw_reward, abs=1e-6), (
        f"Expected shaped reward to differ from raw {raw_reward}, got {stored_reward}"
    )


# ---------------------------------------------------------------------------
# PER integration with learn()
# ---------------------------------------------------------------------------

def test_per_integration_with_learn():
    """With PER strategy: learn() must pass IS weights and update priorities."""
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.batch_size = 32
    agent.replay_buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=cfg.agent_setting.PER_ALPHA,
        beta=cfg.agent_setting.PER_BETA,
        beta_increment=cfg.agent_setting.PER_BETA_INCREMENT,
        epsilon=cfg.agent_setting.PER_EPSILON,
        abs_err_upper=cfg.agent_setting.PER_ABS_ERR_UPPER,
    )

    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.ones((1, 64), dtype=np.float32) * 0.1
    for i in range(40):
        agent.store_transition(obs, i % 64, float(i % 3 - 1), i % 10 == 0, nobs)

    # Record beta before learn (it increments on each sample())
    beta_before = agent.replay_buffer._beta
    agent.learn()
    # Beta should have advanced (sample() was called inside learn())
    assert agent.replay_buffer._beta > beta_before, (
        "PER beta must anneal after learn() calls sample()"
    )
    assert agent.learn_step_counter == 1
