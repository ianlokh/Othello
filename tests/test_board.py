"""Board logic unit tests.

Tests the pure game-logic Board class from othello_main_pygame.
Zero pygame dependency in the Board class itself — these tests run headlessly.
"""
import numpy as np
import pytest

from othello_main_pygame import Board
from othello.constants import BLACK_ID, WHITE_ID, GRID_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def board():
    return Board()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_board_layout(board):
    """The four centre cells must be placed in the standard Othello opening."""
    mid = GRID_SIZE // 2  # 4
    assert board.grid[mid - 1][mid - 1] == WHITE_ID
    assert board.grid[mid][mid] == WHITE_ID
    assert board.grid[mid - 1][mid] == BLACK_ID
    assert board.grid[mid][mid - 1] == BLACK_ID
    # All other cells must be empty
    occupied = {(mid - 1, mid - 1), (mid, mid), (mid - 1, mid), (mid, mid - 1)}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) not in occupied:
                assert board.grid[r][c] == 0, f"Cell ({r},{c}) should be empty"


def test_initial_valid_moves_black(board):
    """Black's opening moves must be exactly the four standard positions."""
    expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert board.get_valid_moves(BLACK_ID) == expected


def test_initial_scores(board):
    """Initial board has 2 black and 2 white tokens."""
    assert board.calculate_scores() == (2, 2)


# ---------------------------------------------------------------------------
# Token placement and flipping
# ---------------------------------------------------------------------------

def test_place_token(board):
    """Placing black at (2,3) must flip (3,3) and return the flipped list."""
    flipped = board.place_token(2, 3, BLACK_ID)
    assert (3, 3) in flipped, "Expected (3,3) to be in flipped list"
    assert board.grid[2][3] == BLACK_ID, "Placed cell should be black"
    assert board.grid[3][3] == BLACK_ID, "Flipped cell (3,3) should now be black"


def test_flip_multiple_directions():
    """A single placement must be able to flip tokens in two distinct directions.

    Setup (verified by manual trace):
      After moves B(2,3), W(4,2), B(5,4), W(3,5), B(4,1), W(3,2)
      Black at (2,2) flips (3,3) [diagonal] and (3,2) [vertical] — two directions.
    """
    board = Board()
    board.place_token(2, 3, BLACK_ID)
    board.place_token(4, 2, WHITE_ID)
    board.place_token(5, 4, BLACK_ID)
    board.place_token(3, 5, WHITE_ID)
    board.place_token(4, 1, BLACK_ID)
    board.place_token(3, 2, WHITE_ID)

    assert (2, 2) in board.get_valid_moves(BLACK_ID), "Setup error: (2,2) should be a valid black move"
    flipped = board.place_token(2, 2, BLACK_ID)
    assert (3, 3) in flipped, "Expected (3,3) in flipped (diagonal direction)"
    assert (3, 2) in flipped, "Expected (3,2) in flipped (vertical direction)"
    assert len(flipped) == 2, f"Expected exactly 2 flips, got {len(flipped)}: {flipped}"


# ---------------------------------------------------------------------------
# Valid moves edge cases
# ---------------------------------------------------------------------------

def test_no_valid_moves_returns_empty_set():
    """A board where one player has no valid moves returns an empty set."""
    board = Board()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            board.grid[r][c] = WHITE_ID
    assert board.get_valid_moves(BLACK_ID) == set()


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def test_is_terminal_false_at_start(board):
    assert not board.is_terminal()


@pytest.mark.parametrize("setup_id", ["full_board", "no_moves_for_either"])
def test_is_terminal_true_cases(setup_id):
    """Parametrized: both a full board and a no-moves state must be terminal."""
    board = Board()
    if setup_id == "full_board":
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                board.grid[r][c] = WHITE_ID
    elif setup_id == "no_moves_for_either":
        # All-black board except one isolated white corner — neither can flip
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                board.grid[r][c] = BLACK_ID
        board.grid[0][0] = WHITE_ID
    assert board.is_terminal()


# ---------------------------------------------------------------------------
# Numpy conversion
# ---------------------------------------------------------------------------

def test_to_numpy_shape_and_dtype(board):
    arr = board.to_numpy()
    assert arr.shape == (8, 8)
    assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_restores_initial_state(board):
    """After mutations, reset() must restore the standard opening position."""
    board.place_token(2, 3, BLACK_ID)
    board.place_token(2, 4, WHITE_ID)
    board.reset()
    assert board.calculate_scores() == (2, 2)
    assert board.get_valid_moves(BLACK_ID) == {(2, 3), (3, 2), (4, 5), (5, 4)}


# ---------------------------------------------------------------------------
# Full game sequence (deterministic, exact scores)
# ---------------------------------------------------------------------------

def test_full_game_sequence():
    """Known 3-move sequence must produce exact scores (5 black, 2 white).

    Trace (all moves verified valid):
      Black at (2,3) → flips (3,3)      → scores (4, 1)
      White at (2,4) → flips (3,4)      → scores (3, 3)
      Black at (3,5) → flips (3,4)      → scores (5, 2)
    """
    board = Board()
    board.place_token(2, 3, BLACK_ID)
    board.place_token(2, 4, WHITE_ID)
    board.place_token(3, 5, BLACK_ID)
    assert board.calculate_scores() == (5, 2)
