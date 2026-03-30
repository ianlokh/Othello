"""Replay buffer unit tests.

Tests both UniformReplayBuffer and PrioritizedReplayBuffer.
Common behaviour is parametrized across both types; strategy-specific tests are separate.
"""
import numpy as np
import pytest

from othello.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(i=0):
    """Return a synthetic (obs, action, reward, next_obs, done) tuple."""
    obs = np.full((1, 64), float(i), dtype=np.float32)
    return (obs, i % 64, float(i), obs * 0.5, i % 10 == 0)


def _fill(buf, n):
    for i in range(n):
        buf.add(_make_transition(i))


@pytest.fixture(params=["uniform", "per"], ids=["uniform", "per"])
def buffer(request):
    """Parametrized fixture: fresh buffer of each type, capacity=100."""
    if request.param == "uniform":
        return UniformReplayBuffer(capacity=100)
    return PrioritizedReplayBuffer(capacity=100)


# ---------------------------------------------------------------------------
# Parametrized tests (both buffer types)
# ---------------------------------------------------------------------------

def test_buffer_add_and_len(buffer):
    """Adding N transitions gives len() == N."""
    _fill(buffer, 10)
    assert len(buffer) == 10


def test_buffer_capacity_overflow(buffer):
    """Adding more than capacity keeps len() at capacity."""
    _fill(buffer, 150)
    assert len(buffer) == 100


def test_buffer_update_last_patches(buffer):
    """update_last() adds terminal reward and sets done=True on the last transition.

    Verified by sampling repeatedly until the patched transition is encountered,
    avoiding any internal implementation inspection.
    """
    # Add a transition with reward=0.0, done=False
    buf_transition = (np.zeros((1, 64), dtype=np.float32), 5, 0.0,
                      np.zeros((1, 64), dtype=np.float32), False)
    _fill(buffer, 19)        # fill most of the buffer first
    buffer.add(buf_transition)  # this is the one we will patch
    buffer.update_last(25.0, done=True)

    # Sample repeatedly to find the patched transition
    found_patched = False
    for _ in range(500):
        transitions, _, _ = buffer.sample(10)
        for t in transitions:
            if abs(t[2] - 25.0) < 1e-4 and t[4] is True:
                found_patched = True
                break
        if found_patched:
            break

    assert found_patched, (
        "Patched transition (reward=25.0, done=True) was not found in 500 sample draws"
    )


# ---------------------------------------------------------------------------
# Uniform-specific tests
# ---------------------------------------------------------------------------

def test_uniform_sample():
    """sample() returns batch_size transitions with weights=1.0 and indices=None."""
    buf = UniformReplayBuffer(capacity=100)
    _fill(buf, 20)
    transitions, weights, indices = buf.sample(5)
    assert len(transitions) == 5
    assert indices is None
    assert np.allclose(weights, 1.0)


# ---------------------------------------------------------------------------
# PER-specific tests
# ---------------------------------------------------------------------------

def test_per_sample_returns_indices_and_weights():
    """PER sample() must return non-None indices and IS weight array."""
    buf = PrioritizedReplayBuffer(capacity=100)
    _fill(buf, 20)
    transitions, is_weights, indices = buf.sample(5)
    assert len(transitions) == 5
    assert indices is not None
    assert len(indices) == 5
    assert is_weights.shape == (5,)


def test_per_is_weights_bounded():
    """IS weights must lie in [0.0, 1.0] after normalization."""
    buf = PrioritizedReplayBuffer(capacity=100)
    _fill(buf, 50)
    _, is_weights, _ = buf.sample(10)
    assert np.all(is_weights >= 0.0)
    assert np.all(is_weights <= 1.0 + 1e-6)


def test_per_high_priority_sampled_more():
    """A transition with 1000× higher priority should dominate sampling."""
    buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
    _fill(buf, 20)

    # Elevate one specific tree leaf to a very high priority
    high_idx = buf._tree.capacity - 1 + 5   # tree index for data slot 5
    buf._tree.update(high_idx, 1000.0)

    count = 0
    n_samples = 2000
    for _ in range(n_samples):
        _, _, indices = buf.sample(1)
        if indices[0] == high_idx:
            count += 1

    assert count > n_samples * 0.1, (
        f"High-priority transition sampled only {count}/{n_samples} times"
    )


def test_per_beta_anneals():
    """Beta must increase by beta_increment after each sample() call."""
    buf = PrioritizedReplayBuffer(capacity=100, beta=0.4, beta_increment=0.01)
    _fill(buf, 20)
    initial_beta = buf._beta
    buf.sample(5)
    assert buf._beta > initial_beta
    assert buf._beta == pytest.approx(initial_beta + 0.01, abs=1e-6)
