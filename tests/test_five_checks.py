"""Five mandatory verification checks from CLAUDE.md.

Uses a synthetic (no-Pygame) training loop so the tests run headlessly.
All five tests must pass before any RL task is marked complete.
"""
import numpy as np
import pytest
import torch

from othello.othello_agent import OthelloDQN, OthelloDQNModel

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
SMALL_BATCH = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fill_buffer(agent, n, seed):
    """Push *n* random synthetic transitions into *agent*'s replay buffer."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, (1, 64)).astype(np.float32)
    for _ in range(n):
        nobs = rng.uniform(-1, 1, (1, 64)).astype(np.float32)
        action = int(rng.integers(0, 64))
        reward = float(rng.choice([-15.0, 0.0, 25.0]))
        done = bool(rng.random() < 0.1)
        agent.store_transition(obs, action, reward, done, nobs)
        obs = nobs


def _make_agent(seed=42, batch_size=SMALL_BATCH):
    """Return a white agent with a full replay buffer, ready to learn()."""
    agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=seed)
    agent.batch_size = batch_size
    _fill_buffer(agent, n=batch_size + 50, seed=seed)
    return agent


# ---------------------------------------------------------------------------
# Check 1 — Shape Check
# ---------------------------------------------------------------------------

def test_check1_shape():
    """Forward-pass shapes must match Othello's observation and action space."""
    model = OthelloDQNModel(nb_observations=64, action_dim=64).to(DEVICE)
    model.eval()

    # 2-D input: (batch, 64)
    x2d = torch.zeros(8, 64, device=DEVICE)
    with torch.no_grad():
        out2d = model(x2d)
    assert out2d.shape == (8, 64), f"2D input: expected (8, 64), got {out2d.shape}"

    # 3-D legacy input: (batch, 1, 64) — observations stored with extra dim
    x3d = torch.zeros(8, 1, 64, device=DEVICE)
    with torch.no_grad():
        out3d = model(x3d)
    assert out3d.shape == (8, 64), f"3D input: expected (8, 64), got {out3d.shape}"

    print(f"\n  [Check 1] 2D ({x2d.shape}) → {out2d.shape}  ✓")
    print(f"  [Check 1] 3D ({x3d.shape}) → {out3d.shape}  ✓")


# ---------------------------------------------------------------------------
# Check 2 — Gradient Flow Check
# ---------------------------------------------------------------------------

def test_check2_gradient_flow():
    """Every named parameter must have a non-None, non-zero gradient after one learn()."""
    agent = _make_agent(seed=42)
    agent.learn()

    for name, param in agent.model_eval.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"
        assert param.grad.abs().sum() > 0, f"Zero gradient for '{name}'"

    n_params = sum(p.numel() for p in agent.model_eval.parameters())
    print(f"\n  [Check 2] All parameters ({n_params:,} total) received gradients  ✓")


# ---------------------------------------------------------------------------
# Check 3 — Numerical Sanity Check
# ---------------------------------------------------------------------------

def test_check3_numerical_sanity():
    """50 learn() steps: loss finite every step; Q-values do not explode.

    This test requires learn() to return loss.item().
    FAILS before fix: learn() returns None.
    """
    agent = _make_agent(seed=42, batch_size=SMALL_BATCH)
    _fill_buffer(agent, n=300, seed=99)   # extra transitions for variety

    losses = []
    for _ in range(50):
        loss_val = agent.learn()
        assert loss_val is not None, (
            "learn() must return loss.item() — currently returns None. "
            "Add 'return loss.item()' at the end of learn()."
        )
        assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        losses.append(loss_val)

    first10 = float(np.mean(losses[:10]))
    last10 = float(np.mean(losses[-10:]))

    # Downward trend: last-10 should not be more than 2× the first-10 mean
    assert last10 <= first10 * 2.0, (
        f"Loss is trending sharply upward — possible exploding gradients. "
        f"First-10 mean: {first10:.4f}, last-10 mean: {last10:.4f}"
    )

    # Q-values must stay in a sane range
    agent.model_eval.eval()
    dummy = torch.zeros(1, 64, device=DEVICE)
    with torch.no_grad():
        q_vals = agent.model_eval(dummy)
    agent.model_eval.train()
    assert q_vals.abs().max().item() < 1e4, (
        f"Q-values exploded: max abs = {q_vals.abs().max().item():.2e}"
    )

    print(f"\n  [Check 3] Loss step 1: {losses[0]:.4f}, step 50: {losses[-1]:.4f}")
    print(f"  [Check 3] First-10 mean: {first10:.4f}, last-10 mean: {last10:.4f}  ✓")
    print(f"  [Check 3] Max |Q-value|: {q_vals.abs().max().item():.4f}  ✓")


# ---------------------------------------------------------------------------
# Check 4 — Reproducibility Check
# ---------------------------------------------------------------------------

def test_check4_reproducibility():
    """Same seed → identical model weights after 10 learn() steps.

    FAILS on MPS before fix: torch.mps.manual_seed() is not called,
    so the Metal RNG is not deterministically seeded.
    """
    def _run(seed):
        agent = OthelloDQN(nb_observations=64, player="white", mode="training", seed=seed)
        agent.batch_size = SMALL_BATCH
        _fill_buffer(agent, n=SMALL_BATCH + 50, seed=seed)
        for _ in range(10):
            agent.learn()
        return {k: v.clone().cpu() for k, v in agent.model_eval.state_dict().items()}

    w1 = _run(42)
    w2 = _run(42)

    for key in w1:
        max_diff = (w1[key].float() - w2[key].float()).abs().max().item()
        assert torch.allclose(w1[key].float(), w2[key].float(), atol=1e-5), (
            f"Non-deterministic at '{key}': max diff = {max_diff:.2e}. "
            "Fix: add torch.mps.manual_seed(seed) in set_env_seeds()."
        )

    print(f"\n  [Check 4] All state_dict keys identical across two seed-42 runs  ✓")


# ---------------------------------------------------------------------------
# Check 5 — Device Placement Check
# ---------------------------------------------------------------------------

def test_check5_device_placement():
    """Model parameters and training tensors must reside on the target device."""
    agent = _make_agent(seed=42)

    eval_device = next(agent.model_eval.parameters()).device
    tgt_device = next(agent.model_target.parameters()).device

    assert eval_device.type == DEVICE.type, (
        f"model_eval on {eval_device}, expected {DEVICE}"
    )
    assert tgt_device.type == DEVICE.type, (
        f"model_target on {tgt_device}, expected {DEVICE}"
    )

    # Run one learn() step and verify no MPS→CPU silent fallback raises an error
    agent.learn()

    print(f"\n  [Check 5] model_eval on {eval_device}  ✓")
    print(f"  [Check 5] model_target on {tgt_device}  ✓")
