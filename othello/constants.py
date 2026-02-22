"""Shared constants for the Othello game.

Game-rule constants, rendering dimensions, and colour definitions used by
both the interactive Pygame UI (``othello_main_pygame``) and the Gymnasium
training environment (``othello.envs.othello_pygame_env``).
"""

# ---------------------------------------------------------------------------
# Game-rule constants
# ---------------------------------------------------------------------------

GRID_SIZE = 8

DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)

# Player IDs
BLACK_ID = -1
WHITE_ID = 1

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

FPS = 60
CELL_SIZE = 50
BOARD_PX = CELL_SIZE * GRID_SIZE  # 400
BORDER_WIDTH = 10

# ---------------------------------------------------------------------------
# Colours (RGB / RGBA)
# ---------------------------------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREEN = (53, 136, 86)
DARK_GREY = (40, 40, 40)
LIGHT_GREY = (211, 211, 211)
HINT_COLOR = (30, 144, 255, 80)  # translucent blue
