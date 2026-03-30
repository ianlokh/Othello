"""Gymnasium environment tests.

Tests the OthelloPygameEnv contract: obs/action spaces, rewards, turn logic.
All tests run headlessly with render_mode=None.
"""
import random

import numpy as np
import pytest

from othello import config as cfg
from othello.envs.othello_pygame_env import OthelloPygameEnv
from othello.constants import BLACK_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = OthelloPygameEnv(render_mode=None)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _play_full_game(env, seed=42):
    """Play a full game with deterministic move selection; return final (reward, winner)."""
    env.reset(seed=seed)
    reward = 0
    done = False
    steps = 0
    while not done and steps < 200:
        valid = list(env.next_possible_actions)
        if not valid:
            break
        # Deterministic: always pick the lowest-index move
        move = min(valid)
        action = move[0] * 8 + move[1]
        _, reward, done, _, _ = env.step(action)
        steps += 1
    return reward, env.winner


# ---------------------------------------------------------------------------
# Reset contract
# ---------------------------------------------------------------------------

def test_env_reset_returns_obs_and_info(env):
    """reset() must return (obs, info) where obs['state'] has shape (64,)."""
    obs, info = env.reset()
    assert "state" in obs
    assert obs["state"].shape == (64,)
    assert "next_possible_actions" in info
    assert len(info["next_possible_actions"]) > 0


def test_env_initial_valid_moves(env):
    """Initial valid moves should match the standard Othello opening for black."""
    _, info = env.reset()
    expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert info["next_possible_actions"] == expected


# ---------------------------------------------------------------------------
# Step contract
# ---------------------------------------------------------------------------

def test_env_step_basics(env):
    """Step returns correct types and alternates player on a normal move."""
    env.reset()
    valid_move = next(iter(env.next_possible_actions))
    action = valid_move[0] * 8 + valid_move[1]
    initial_player = env.current_index

    obs, reward, done, truncated, info = env.step(action)

    assert "state" in obs
    assert obs["state"].shape == (64,)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert truncated is False
    # Player should have alternated (unless the opponent has no moves)
    if info["next_possible_actions"]:
        assert env.current_index != initial_player, "Player should alternate after a normal step"


def test_env_step_invalid_action(env):
    """Stepping with an invalid action should raise an AssertionError."""
    env.reset()
    # Action 0 = (0,0) is never valid on the opening board
    with pytest.raises(AssertionError):
        env.step(0)


# ---------------------------------------------------------------------------
# Terminal rewards  (generalised: sign matches winner, not hardcoded values)
# ---------------------------------------------------------------------------

def test_env_terminal_rewards(env):
    """Terminal reward sign must match the winner: White→>0, Black→<0, Tie→>0."""
    # Play a full game with two different seeds to maximise chance of hitting
    # different outcomes. Assert the invariant for each outcome encountered.
    outcomes_seen = set()
    for seed in range(20):
        reward, winner = _play_full_game(env, seed=seed)
        if winner is None:
            continue
        outcomes_seen.add(winner)
        if winner == "White":
            assert reward > 0, f"White win should give reward > 0, got {reward}"
        elif winner == "Black":
            assert reward < 0, f"Black win should give reward < 0, got {reward}"
        elif winner == "Tie":
            assert reward > 0, f"Tie should give reward > 0, got {reward}"

    assert len(outcomes_seen) > 0, "No games reached a terminal state"


# ---------------------------------------------------------------------------
# Full game loop (also covers skip-turn: if skip-turn breaks, game hangs/crashes)
# ---------------------------------------------------------------------------

def test_env_full_game_terminates(env):
    """A full random game must terminate with done=True within 64 steps.

    This also exercises skip-turn logic — if it breaks, the loop exceeds 64.
    """
    env.reset(seed=7)
    done = False
    steps = 0
    while not done:
        valid = list(env.next_possible_actions)
        if not valid:
            break
        move = random.choice(valid)
        action = move[0] * 8 + move[1]
        _, _, done, _, _ = env.step(action)
        steps += 1
    assert steps <= 64, f"Game took {steps} steps — should end within 64"
    assert done or not env.next_possible_actions, "Game should be over"


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def test_env_reward_shaping_applied(env):
    """With shaping enabled, a non-terminal transition on a non-zero board gets
    a non-zero shaped reward component."""
    from othello.othello_agent import OthelloDQN, POSITION_WEIGHTS
    from othello.replay_buffer import UniformReplayBuffer

    if not cfg.agent_setting.REWARD_SHAPING_ENABLED:
        pytest.skip("Reward shaping is disabled in config")

    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.replay_buffer = UniformReplayBuffer(capacity=1000)

    # Use all-ones board (maximises POSITION_WEIGHTS dot product → large Phi)
    # so the shaping term is guaranteed non-zero
    obs = np.ones((1, 64), dtype=np.float32)
    nobs = -np.ones((1, 64), dtype=np.float32)   # opposite sign → large delta
    raw_reward = 0.0
    agent.store_transition(obs, 10, raw_reward, False, nobs)

    stored = agent.replay_buffer._buffer[-1]
    stored_reward = stored[2]
    assert stored_reward != pytest.approx(raw_reward, abs=1e-6), (
        f"Expected shaped reward to differ from raw {raw_reward}, got {stored_reward}"
    )


# ---------------------------------------------------------------------------
# UI/logic boundary
# ---------------------------------------------------------------------------

def test_game_advance_matches_env_step(env):
    """Env board state after one step must match a standalone Board after the same move."""
    from othello_main_pygame import Board

    env.reset(seed=42)
    board = Board()

    move = (2, 3)  # standard black opening move
    action = move[0] * 8 + move[1]

    board.place_token(move[0], move[1], BLACK_ID)
    obs, _, _, _, _ = env.step(action)

    env_board_flat = obs["state"]
    standalone_flat = board.to_numpy().flatten()
    np.testing.assert_array_equal(env_board_flat, standalone_flat)
