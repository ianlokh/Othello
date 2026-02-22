"""Replay buffer implementations for DQN training.

Provides a common ``ReplayBuffer`` interface with two concrete strategies:

* ``UniformReplayBuffer`` — simple random sampling (wraps a deque).
* ``PrioritizedReplayBuffer`` — proportional prioritization via a SumTree,
  with importance-sampling (IS) weight correction.

Usage::

    from othello.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

    buf = UniformReplayBuffer(capacity=200_000)
    buf.add((obs, action, reward, next_obs, done))
    transitions, is_weights, indices = buf.sample(batch_size)
    buf.update_priorities(indices, td_errors)   # no-op for uniform
    buf.update_last(terminal_reward, done=True)  # patch last transition
"""

import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ReplayBuffer(ABC):
    """Abstract interface for experience replay buffers.

    Every buffer stores transitions as tuples
    ``(observation, action, reward, next_observation, done)``.
    """

    @abstractmethod
    def add(self, transition):
        """Store a single transition tuple."""

    @abstractmethod
    def sample(self, batch_size):
        """Sample a batch of transitions.

        Returns
        -------
        transitions : list[tuple]
            List of ``(obs, action, reward, next_obs, done)`` tuples.
        is_weights : np.ndarray, shape (batch_size,)
            Importance-sampling weights (all 1.0 for uniform replay).
        indices : np.ndarray | None
            Buffer-internal indices for ``update_priorities``.
            ``None`` when priorities are not used (uniform replay).
        """

    @abstractmethod
    def update_priorities(self, indices, td_errors):
        """Update priorities for previously sampled transitions.

        No-op for uniform replay.
        """

    @abstractmethod
    def update_last(self, terminal_reward, done):
        """Patch the most recently added transition.

        Adds *terminal_reward* to the existing reward and sets the *done*
        flag.  Used when the game ends on the opponent's move and the
        agent's last stored transition needs to reflect the terminal outcome.
        """

    @abstractmethod
    def __len__(self):
        """Return the number of transitions currently stored."""


# ---------------------------------------------------------------------------
# Uniform replay buffer
# ---------------------------------------------------------------------------

class UniformReplayBuffer(ReplayBuffer):
    """Simple replay buffer with uniform random sampling."""

    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def add(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self._buffer, batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return samples, weights, None

    def update_priorities(self, indices, td_errors):
        pass  # uniform sampling ignores priorities

    def update_last(self, terminal_reward, done):
        if len(self._buffer) == 0:
            return
        last = self._buffer[-1]
        patched = (last[0], last[1], last[2] + terminal_reward, last[3], done)
        self._buffer[-1] = patched

    def __len__(self):
        return len(self._buffer)


# ---------------------------------------------------------------------------
# SumTree (internal data structure for PER)
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf holds a priority value.

    Supports O(log N) proportional sampling and priority updates.
    Adapted from https://github.com/jaara/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.size < self.capacity:
            self.size += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        """Sample a leaf proportional to stored priorities."""
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left]:
                parent_idx = left
            else:
                value -= self.tree[left]
                parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


# ---------------------------------------------------------------------------
# Prioritized replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional Prioritized Experience Replay (Schaul et al. 2016).

    Transitions with higher TD error are sampled more frequently.
    Importance-sampling weights correct for the resulting bias.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    alpha : float
        Prioritization exponent.  0 = uniform, 1 = full prioritization.
    beta : float
        Initial IS weight exponent.  Anneals toward 1.0 over training.
    beta_increment : float
        Amount to increase *beta* on each ``sample()`` call.
    epsilon : float
        Small constant added to TD errors to avoid zero priority.
    abs_err_upper : float
        Maximum clipped TD error (prevents extreme priorities).
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001,
                 epsilon=0.01, abs_err_upper=10.0):
        self._tree = SumTree(capacity)
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = beta_increment
        self._epsilon = epsilon
        self._abs_err_upper = abs_err_upper
        self._last_data_pointer = None

    # -- ReplayBuffer interface ------------------------------------------------

    def add(self, transition):
        max_p = np.max(self._tree.tree[-self._tree.capacity:])
        if max_p == 0:
            max_p = self._abs_err_upper
        self._last_data_pointer = self._tree.data_pointer
        self._tree.add(max_p, transition)

    def sample(self, batch_size):
        indices = np.empty(batch_size, dtype=np.int32)
        transitions = []
        is_weights = np.empty(batch_size, dtype=np.float32)

        segment = self._tree.total_priority / batch_size
        self._beta = min(1.0, self._beta + self._beta_increment)

        # minimum non-zero priority for IS weight normalization
        leaf_priorities = self._tree.tree[-self._tree.capacity:]
        valid = leaf_priorities[leaf_priorities > 0]
        min_prob = (valid.min() / self._tree.total_priority) if len(valid) > 0 else 1e-5

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            v = np.random.uniform(low, high)
            idx, priority, data = self._tree.get_leaf(v)
            prob = priority / self._tree.total_priority
            is_weights[i] = (prob / min_prob) ** (-self._beta)
            indices[i] = idx
            transitions.append(data)

        # normalize IS weights to [0, 1]
        is_weights /= is_weights.max()
        return transitions, is_weights, indices

    def update_priorities(self, indices, td_errors):
        abs_errors = np.abs(td_errors) + self._epsilon
        clipped = np.minimum(abs_errors, self._abs_err_upper)
        priorities = clipped ** self._alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))

    def update_last(self, terminal_reward, done):
        if self._last_data_pointer is None:
            return
        idx = self._last_data_pointer
        old = self._tree.data[idx]
        if old is None or (isinstance(old, (int, float)) and old == 0):
            return
        patched = (old[0], old[1], old[2] + terminal_reward, old[3], done)
        self._tree.data[idx] = patched

        # boost priority so this terminal transition gets replayed soon
        tree_idx = idx + self._tree.capacity - 1
        max_p = np.max(self._tree.tree[-self._tree.capacity:])
        self._tree.update(tree_idx, max(max_p, self._abs_err_upper))

    def __len__(self):
        return self._tree.size
