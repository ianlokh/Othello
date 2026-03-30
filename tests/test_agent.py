"""Agent unit tests — ported from test_pytorch_agent.py to pytest format,
plus new regression tests for bugs identified in the Phase 1 audit.
"""
import numpy as np
import pytest
import torch

from othello import othello_agent
from othello.othello_agent import OthelloDQN, OthelloDQNModel
from othello import config as cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_agent():
    return OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)


@pytest.fixture
def filled_agent():
    """White agent whose buffer is full enough to call learn()."""
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.batch_size = 32
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.ones((1, 64), dtype=np.float32) * 0.1
    for i in range(40):
        agent.store_transition(obs, i % 64, float(i % 3 - 1), i % 20 == 0, nobs)
    return agent


# ---------------------------------------------------------------------------
# Ported tests (regression guard for the existing 10 checks)
# ---------------------------------------------------------------------------

def test_instantiation(white_agent):
    assert hasattr(white_agent, "model_eval")
    assert hasattr(white_agent, "model_target")
    assert white_agent.model_eval.optimizer is not None
    assert white_agent.model_eval.scheduler is not None
    assert white_agent.model_eval.criterion is not None
    assert not white_agent.model_target.training, "model_target must be in eval() mode"
    assert white_agent.model_eval.training, "model_eval must be in train() mode"


def test_play_mode_agent():
    agent = OthelloDQN(nb_observations=64, player="other", mode="play", seed=0)
    assert not hasattr(agent, "model_eval"), "model_eval must not exist for player='other'"
    assert hasattr(agent, "model_target")
    assert agent.epsilon == 0.0
    assert not agent.model_target.training


def test_choose_action_exploration(white_agent):
    obs = np.zeros((1, 64), dtype=np.float32)
    possible = {(3, 2), (2, 3), (4, 5), (5, 4)}
    valid_indices = {r * 8 + c for r, c in possible}
    white_agent.epsilon = 1.0
    for _ in range(20):
        assert white_agent.choose_action(obs, possible) in valid_indices


def test_choose_action_exploitation(white_agent):
    obs = np.zeros((1, 64), dtype=np.float32)
    possible = {(3, 2), (2, 3), (4, 5), (5, 4)}
    valid_indices = {r * 8 + c for r, c in possible}
    white_agent.epsilon = 0.0
    action = white_agent.choose_action(obs, possible)
    assert 0 <= action < 64
    assert action in valid_indices


def test_store_transition(white_agent):
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.ones((1, 64), dtype=np.float32) * 0.1
    white_agent.store_transition(obs, 20, 0.0, False, nobs)
    assert len(white_agent.replay_buffer) == 1


def test_learn_increments_counter(filled_agent):
    before = filled_agent.learn_step_counter
    filled_agent.learn()
    assert filled_agent.learn_step_counter == before + 1


def test_learn_decays_epsilon(filled_agent):
    initial_eps = filled_agent.epsilon
    filled_agent.learn()
    assert filled_agent.epsilon < initial_eps


def test_soft_update():
    src = OthelloDQNModel(64, 64)
    tgt = OthelloDQNModel(64, 64)
    template = src.state_dict()
    src_sd = {k: torch.ones_like(v) if v.dtype.is_floating_point else v.clone()
              for k, v in template.items()}
    tgt_sd = {k: torch.zeros_like(v) if v.dtype.is_floating_point else v.clone()
              for k, v in template.items()}
    src.load_state_dict(src_sd)
    tgt.load_state_dict(tgt_sd)
    OthelloDQN._soft_update(tgt, src, alpha=0.5)
    for key, val in tgt.state_dict().items():
        if val.dtype.is_floating_point:
            assert torch.allclose(val, torch.full_like(val, 0.5), atol=1e-5), \
                f"Soft-update mismatch at '{key}'"


def test_save_load_round_trip(filled_agent):
    filled_agent.learn()
    filled_agent.save_model(name="OthelloDQN_test", save_step="unit_test")
    save_dir = "./models/unit_test"
    agent2 = OthelloDQN(nb_observations=64, player="white", mode="play", seed=1)
    ok, _ = agent2.load_model(path=save_dir, name="OthelloDQN_test")
    assert ok
    sd1 = filled_agent.model_target.state_dict()
    sd2 = agent2.model_target.state_dict()
    for key in sd1:
        assert torch.allclose(sd1[key].float(), sd2[key].float(), atol=1e-6), \
            f"State dict mismatch at '{key}'"
    assert not agent2.model_target.training


def test_reload_model(filled_agent):
    """reload_model() must successfully reload from the path stored during save_model().

    This exercises the path used by the Pygame UI when the user resets the game.
    """
    filled_agent.learn()
    filled_agent.save_model(name="OthelloDQN_reload_test", save_step="unit_test")

    ok, msg = filled_agent.reload_model()
    assert ok, f"reload_model() failed: {msg}"

    # Weights must be identical to what was saved (model_target round-trip)
    sd_before = filled_agent.model_target.state_dict()
    for key in sd_before:
        assert sd_before[key] is not None, f"Missing key after reload: {key}"
    assert not filled_agent.model_target.training, "model_target must remain in eval() after reload"


def test_reward_transition_update():
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=5)
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    agent.store_transition(obs, 10, 0.0, False, nobs)
    agent.reward_transition_update(25.0)   # must not raise


def test_reward_transition_update_patches_values():
    """reward_transition_update must add terminal_reward to the stored reward and set done=True.

    BEFORE FIX: test_reward_transition_update only checked no-crash, not actual patching.
    This test verifies the actual patch: reward is added, done is flipped.
    """
    from othello.replay_buffer import UniformReplayBuffer
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=5)
    # Force uniform buffer so we can inspect via _buffer[-1]
    agent.replay_buffer = UniformReplayBuffer(capacity=1000)
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    # All-zero observations → shaping adds 0.0 → stored reward == 0.0
    agent.store_transition(obs, 10, 0.0, False, nobs)
    stored_before = agent.replay_buffer._buffer[-1]
    assert stored_before[2] == pytest.approx(0.0, abs=1e-4), \
        f"Expected stored reward ≈ 0.0 before patch, got {stored_before[2]}"
    assert stored_before[4] is False, "done should be False before patch"

    agent.reward_transition_update(25.0)

    stored_after = agent.replay_buffer._buffer[-1]
    assert stored_after[2] == pytest.approx(25.0, abs=1e-4), \
        f"Expected stored reward ≈ 25.0 after patch (+25.0 terminal), got {stored_after[2]}"
    assert stored_after[4] is True, "done should be True after patch"


def test_weight_sync_between_agents(white_agent):
    agent_other = OthelloDQN(nb_observations=64, player="other", mode="play", seed=99)
    agent_other.model_target.load_state_dict(white_agent.model_target.state_dict())
    agent_other.model_target.eval()
    for key in white_agent.model_target.state_dict():
        assert torch.allclose(
            white_agent.model_target.state_dict()[key].float(),
            agent_other.model_target.state_dict()[key].float(),
            atol=1e-6,
        )
    assert not agent_other.model_target.training


# ---------------------------------------------------------------------------
# New tests — audit issues
# ---------------------------------------------------------------------------

def test_loss_only_on_taken_action_gradient():
    """With the gather fix the last Linear layer must have gradients ONLY at the
    row corresponding to the taken action.

    Setup: every transition in the buffer takes action=7, so the batch is all-7.
    After fix: last_layer.weight.grad[j] == 0 for all j != 7.
    FAILS before gather fix: BatchNorm train/eval mismatch leaks non-zero
    gradients into every output row.
    """
    FIXED_ACTION = 7
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=0)
    agent.batch_size = 32

    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    for _ in range(40):
        agent.store_transition(obs, FIXED_ACTION, 1.0, False, nobs)

    agent.model_eval.optimizer.zero_grad()
    agent.learn()

    # Last layer is the final nn.Linear(64, 64) → weight shape (64, 64)
    last_linear = list(agent.model_eval.net.children())[-1]
    assert isinstance(last_linear, torch.nn.Linear), "Last layer must be nn.Linear"
    grad = last_linear.weight.grad   # shape (action_dim=64, hidden=64)
    assert grad is not None, "Last layer has no gradient at all"

    for action_idx in range(64):
        row_norm = grad[action_idx].abs().sum().item()
        if action_idx == FIXED_ACTION:
            assert row_norm > 0, (
                f"Expected non-zero gradient at output row {FIXED_ACTION} "
                f"(the taken action), got 0."
            )
        else:
            assert row_norm == 0.0, (
                f"Spurious gradient at output row {action_idx} "
                f"(action was never taken): L1-norm = {row_norm:.4e}.\n"
                "This indicates the loss is computed over all 64 Q-values "
                "instead of only the taken action. Fix: use torch.gather."
            )


def test_target_sync_skips_step_zero():
    """_tgt_evl_sync must NOT fire when learn_step_counter == 0.

    FAILS before fix: the guard is `counter % replace_target_iter == 0`
    which is True at step 0 before any gradient update has happened.
    """
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.batch_size = 32

    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    for _ in range(40):
        agent.store_transition(obs, 5, 0.0, False, nobs)

    assert agent.learn_step_counter == 0, "Counter must start at 0"

    sync_calls = []
    original_sync = agent._tgt_evl_sync
    agent._tgt_evl_sync = lambda: (sync_calls.append(1), original_sync())[1]

    agent.learn()

    assert len(sync_calls) == 0, (
        "_tgt_evl_sync fired at learn_step_counter == 0, before any gradient "
        "update has been applied. Fix: add `learn_step_counter > 0` to the guard."
    )


def test_learn_returns_loss():
    """learn() must return a finite float (needed for Check 3 and monitoring).

    FAILS before fix: learn() has no return statement → returns None.
    """
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=42)
    agent.batch_size = 32
    obs = np.zeros((1, 64), dtype=np.float32)
    nobs = np.zeros((1, 64), dtype=np.float32)
    for _ in range(40):
        agent.store_transition(obs, 5, 1.0, False, nobs)

    result = agent.learn()
    assert result is not None, "learn() returned None — add 'return loss.item()'"
    assert isinstance(result, float), f"learn() should return float, got {type(result)}"
    assert np.isfinite(result), f"learn() returned non-finite loss: {result}"


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------

def test_predict_on_batch_no_grad(white_agent):
    """predict_on_batch() must not accumulate gradients."""
    obs = np.zeros((1, 1, 64), dtype=np.float32)
    white_agent.model_eval.predict_on_batch(obs)
    for param in white_agent.model_eval.parameters():
        assert param.grad is None or param.grad.abs().sum() == 0, \
            "predict_on_batch() should not produce gradients"


def test_train_on_batch_updates_weights(white_agent):
    """A single train_on_batch() call must change model weights."""
    before = {k: v.clone() for k, v in white_agent.model_eval.state_dict().items()}
    x = np.random.randn(4, 1, 64).astype(np.float32)
    y = np.random.randn(4, 1, 64).astype(np.float32)
    white_agent.model_eval.train_on_batch(x, y)
    after = white_agent.model_eval.state_dict()
    changed = any(
        not torch.equal(before[k].float(), after[k].float())
        for k in before if after[k].dtype.is_floating_point
    )
    assert changed, "Weights did not change after train_on_batch()"


def test_train_on_batch_with_sample_weights(white_agent):
    """IS-weighted loss should differ from unweighted loss."""
    x = np.random.randn(4, 1, 64).astype(np.float32)
    y = np.random.randn(4, 1, 64).astype(np.float32)
    loss_unweighted, _ = white_agent.model_eval.train_on_batch(x, y)

    # Reset weights so the comparison is fair
    white_agent.model_eval.load_state_dict(white_agent.model_target.state_dict())
    white_agent.model_eval.build_model()
    weights = np.array([0.1, 0.1, 10.0, 10.0], dtype=np.float32)
    loss_weighted, _ = white_agent.model_eval.train_on_batch(x, y, sample_weight=weights)
    # With highly skewed weights the two losses should differ
    assert loss_unweighted != pytest.approx(loss_weighted, abs=1e-6), \
        "Weighted and unweighted losses should differ with skewed IS weights"


def test_lr_scheduler_decays():
    """LR must decrease after sufficient scheduler steps."""
    model = OthelloDQNModel(64, 64).build_model()
    initial_lr = model.optimizer.param_groups[0]["lr"]
    x = np.random.randn(2, 64).astype(np.float32)
    y = np.random.randn(2, 64).astype(np.float32)
    # StepLR decays every 10000 steps
    for _ in range(10001):
        model.train_on_batch(x, y)
    current_lr = model.optimizer.param_groups[0]["lr"]
    assert current_lr < initial_lr, \
        f"LR should decay after 10k+ steps: initial={initial_lr}, current={current_lr}"


def test_model_persistence_round_trips():
    """Both get/set_weights() and save()/load_state_dict() must preserve parameters."""
    import tempfile
    model = OthelloDQNModel(64, 64).build_model()

    # get_weights / set_weights round trip
    original_weights = model.get_weights()
    model2 = OthelloDQNModel(64, 64).build_model()
    model2.set_weights(original_weights)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "get/set_weights round trip failed"

    # save / load_state_dict round trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        model3 = OthelloDQNModel(64, 64).build_model()
        model3.load_state_dict(torch.load(f.name, weights_only=True))
    for k in model.state_dict():
        assert torch.allclose(
            model.state_dict()[k].float(), model3.state_dict()[k].float(), atol=1e-6
        ), f"save/load round trip failed at '{k}'"


