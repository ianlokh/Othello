"""RL best-practice property tests.

Catches common DQN bugs that cause silent training failure:
- Target network must stay frozen during a learn step
- ε-greedy exploration rate must match epsilon setting
- Q-values must stay bounded after training
- Play mode must not modify weights
"""
import numpy as np
import pytest
import torch

from othello.othello_agent import OthelloDQN, OthelloDQNModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filled_agent(seed=42, batch_size=32, n=40):
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=seed)
    agent.batch_size = batch_size
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.ones((1, 64), dtype=np.float32) * 0.1
    for i in range(n):
        agent.store_transition(obs, i % 64, float(i % 3 - 1), i % 10 == 0, nobs)
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_target_frozen_during_learn():
    """model_target parameters must not change during a single learn() call.

    The target network is only allowed to move during periodic soft updates
    (_tgt_evl_sync), never during the gradient step itself.
    """
    agent = _make_filled_agent()

    # Snapshot target network weights before learning
    snapshot = {k: v.clone() for k, v in agent.model_target.state_dict().items()}

    # Patch _tgt_evl_sync to prevent it firing (we want to isolate the gradient step)
    agent._tgt_evl_sync = lambda: None
    agent.learn()

    for key, before in snapshot.items():
        after = agent.model_target.state_dict()[key]
        assert torch.equal(before, after), (
            f"model_target parameter '{key}' changed during learn() — "
            "gradient must not flow into the target network."
        )


def test_epsilon_greedy_rate():
    """With epsilon=0.3, approximately 30% of actions should be random.

    Uses a binomial test: P(count < 250 | n=1000, p=0.3) is negligibly small.
    A hard lower bound of 200/1000 is used (> 5 sigma from expected 300).
    """
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.epsilon = 0.3

    obs = np.zeros((1, 64), dtype=np.float32)
    possible = {(3, 2), (2, 3), (4, 5), (5, 4)}

    # Track which actions come from epsilon plays (recorded in agent.epsilon_plays)
    n_trials = 1000
    random_count = 0
    for _ in range(n_trials):
        agent.epsilon_plays.clear()
        agent.choose_action(obs, possible)
        # choose_action appends to epsilon_plays only on exploitation (greedy) path
        if not agent.epsilon_plays:
            random_count += 1

    # With epsilon=0.3, ~30% of actions are random (epsilon_plays list stays empty)
    assert 200 <= random_count <= 450, (
        f"Expected ~300/1000 random actions with epsilon=0.3, got {random_count}. "
        "The ε-greedy rate is miscalibrated."
    )


def test_q_values_bounded():
    """Q-values must stay within [-100, 100] after 100 learn steps."""
    agent = _make_filled_agent(n=200, batch_size=32)

    # Extra transitions for variety
    rng = np.random.default_rng(7)
    for _ in range(200):
        obs = rng.uniform(-1, 1, (1, 64)).astype(np.float32)
        nobs = rng.uniform(-1, 1, (1, 64)).astype(np.float32)
        agent.store_transition(obs, int(rng.integers(0, 64)), float(rng.choice([-15., 0., 25.])),
                               bool(rng.random() < 0.1), nobs)

    for _ in range(100):
        agent.learn()

    agent.model_eval.eval()
    dummy = torch.zeros(1, 64)
    with torch.no_grad():
        q_vals = agent.model_eval(dummy)
    agent.model_eval.train()

    max_q = q_vals.abs().max().item()
    assert max_q < 100, f"Q-values exploded: max |Q| = {max_q:.2f}"


def test_no_training_in_play_mode():
    """An agent in play mode must not modify weights when learn() is called."""
    agent = OthelloDQN(nb_observations=64, player="other", mode="play", seed=0)

    snapshot = {k: v.clone() for k, v in agent.model_target.state_dict().items()}

    # load some fake transitions just in case
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    for _ in range(40):
        agent.store_transition(obs, 5, 1.0, False, nobs)  # no-op for "other" player

    agent.learn()  # must be a no-op

    for key, before in snapshot.items():
        after = agent.model_target.state_dict()[key]
        assert torch.equal(before, after), (
            f"Play-mode agent modified weight '{key}' — learn() must be a no-op for player='other'."
        )
