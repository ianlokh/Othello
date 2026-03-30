"""Configuration consistency tests.

Verifies that hyperparameter values are mutually consistent and won't cause
silent training failure (e.g. epsilon never reaches its minimum, beta never
reaches 1.0, shaping dominates terminal rewards).
"""
import pytest

from othello import config as cfg


def test_buffer_size_exceeds_batch():
    """Replay buffer must be at least 10× the batch size to avoid degenerate sampling."""
    assert cfg.agent_setting.REPLAY_BUFFER_SIZE > 10 * cfg.agent_setting.BATCH_SIZE, (
        f"REPLAY_BUFFER_SIZE ({cfg.agent_setting.REPLAY_BUFFER_SIZE}) must be "
        f"> 10 × BATCH_SIZE ({cfg.agent_setting.BATCH_SIZE})"
    )


def test_epsilon_reaches_min():
    """Epsilon must decay to EPSILON_MIN before training ends.

    Epsilon after N learn steps: EPSILON * EPSILON_REDUCE^N
    We need: EPSILON * EPSILON_REDUCE^EPOCHS <= EPSILON_MIN
    """
    final_epsilon = (cfg.agent_setting.EPSILON
                     * cfg.agent_setting.EPSILON_REDUCE ** cfg.training_param.EPOCHS)
    assert final_epsilon <= cfg.agent_setting.EPSILON_MIN, (
        f"Epsilon never reaches EPSILON_MIN ({cfg.agent_setting.EPSILON_MIN}). "
        f"Final value after {cfg.training_param.EPOCHS} epochs: {final_epsilon:.6f}. "
        f"Decrease EPSILON_REDUCE or increase EPOCHS."
    )


def test_per_beta_reaches_one():
    """PER beta must anneal to 1.0 within training duration.

    Beta after N sample() calls: PER_BETA + PER_BETA_INCREMENT * N
    (capped at 1.0 inside PrioritizedReplayBuffer.sample)
    """
    final_beta = (cfg.agent_setting.PER_BETA
                  + cfg.agent_setting.PER_BETA_INCREMENT * cfg.training_param.EPOCHS)
    assert final_beta >= 1.0, (
        f"PER beta only reaches {final_beta:.4f} by end of training "
        f"(needs to reach 1.0). Increase PER_BETA_INCREMENT."
    )


def test_gamma_retention():
    """Terminal reward must propagate back through a full game.

    With ~60 moves per game (white plays ~30), the terminal signal must retain
    at least 10% of its value at the start of white's trajectory.
    """
    max_white_moves = 30
    retention = cfg.agent_setting.GAMMA ** max_white_moves
    assert retention > 0.1, (
        f"GAMMA^{max_white_moves} = {retention:.4f} < 0.1 — terminal signal "
        f"won't propagate back to early moves. Increase GAMMA."
    )


def test_shaping_doesnt_dominate():
    """Reward shaping magnitude must stay well below the terminal reward.

    Max shaping per step ≈ REWARD_POSITIONAL_WEIGHT × max(|F(s,s')|).
    Max |F| ≈ GAMMA × max(Phi) − min(Phi), bounded by the position weight matrix.
    We check the scale factor is < 1.0 (shaping < terminal reward per step).
    """
    import numpy as np
    from othello.othello_agent import POSITION_WEIGHTS

    # Worst-case potential swing: all-white → all-black board
    max_phi = float(np.sum(np.abs(POSITION_WEIGHTS)))
    max_shaping_per_step = cfg.agent_setting.REWARD_POSITIONAL_WEIGHT * (
        cfg.agent_setting.GAMMA * max_phi + max_phi
    )
    assert max_shaping_per_step < cfg.agent_setting.REWARD, (
        f"Max shaping per step ({max_shaping_per_step:.2f}) >= REWARD "
        f"({cfg.agent_setting.REWARD}). Reduce REWARD_POSITIONAL_WEIGHT."
    )
