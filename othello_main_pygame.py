"""Othello — Pygame edition.

A two-player Othello (Reversi) game with support for Human-vs-Human and
Human-vs-RL-Agent modes.  Uses pygame-ce for rendering and pygame_gui for
UI widgets (buttons, dialogs).
"""

import os
import sys

import numpy as np
import pygame
import pygame.gfxdraw
import pygame_gui
from numpy.random import PCG64

from othello import othello_agent
from othello.constants import (
    FPS,
    GRID_SIZE,
    BOARD_PX,
    BORDER_WIDTH,
    DIRECTIONS,
    WHITE,
    BLACK,
    DARK_GREEN,
    DARK_GREY,
    LIGHT_GREY,
    HINT_COLOR,
    BLACK_ID,
    WHITE_ID,
)

# ---------------------------------------------------------------------------
# UI-only constants
# ---------------------------------------------------------------------------
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 700

SPLASH_IMG_PATH = os.path.join("img", "splash", "othello-splash3.jpg")


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    """Lightweight container for player metadata."""

    def __init__(self, player_id, color, label):
        self.id = player_id
        self.color = color
        self.label = label
        self.score = 0


# ---------------------------------------------------------------------------
# Board  (pure game logic — no pygame dependency)
# ---------------------------------------------------------------------------

class Board:
    """8x8 Othello board with all game-rule logic."""

    def __init__(self):
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self._place_starting_tokens()

    # -- public API ---------------------------------------------------------

    def reset(self):
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self._place_starting_tokens()

    def place_token(self, row, col, player_id):
        """Place *player_id* at *(row, col)* and flip captured tokens.

        Returns a list of (row, col) positions that were flipped.
        """
        self.grid[row][col] = player_id
        flipped = []
        for dr, dc in DIRECTIONS:
            if self._check_direction(row, col, dr, dc, player_id):
                flipped.extend(self._flip_direction(row, col, dr, dc, player_id))
        return flipped

    def get_valid_moves(self, player_id):
        """Return a set of (row, col) tuples where *player_id* can play."""
        opponent = -player_id
        moves = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    if self._check_direction(r, c, dr, dc, player_id):
                        moves.add((r, c))
                        break
        return moves

    def calculate_scores(self):
        """Return *(black_score, white_score)*."""
        black = white = 0
        for row in self.grid:
            for cell in row:
                if cell == BLACK_ID:
                    black += 1
                elif cell == WHITE_ID:
                    white += 1
        return black, white

    # def is_terminal(self):
    #     """True when neither player has a legal move."""
    #     return (not self.get_valid_moves(BLACK_ID)
    #             and not self.get_valid_moves(WHITE_ID))

    def to_numpy(self):
        """Return the board as a numpy array (for the RL agent)."""
        return np.array(self.grid, dtype=np.float64)

    # -- private helpers ----------------------------------------------------

    def _place_starting_tokens(self):
        mid = GRID_SIZE // 2
        self.grid[mid - 1][mid - 1] = WHITE_ID
        self.grid[mid][mid] = WHITE_ID
        self.grid[mid - 1][mid] = BLACK_ID
        self.grid[mid][mid - 1] = BLACK_ID

    def _check_direction(self, row, col, dr, dc, player_id):
        """Check whether placing at (row, col) captures in direction (dr, dc)."""
        r, c = row + dr, col + dc
        if not self._in_bounds(r, c) or self.grid[r][c] != -player_id:
            return False
        r += dr
        c += dc
        while self._in_bounds(r, c):
            if self.grid[r][c] == player_id:
                return True
            if self.grid[r][c] == 0:
                return False
            r += dr
            c += dc
        return False

    def _flip_direction(self, row, col, dr, dc, player_id):
        """Flip opponent tokens in one direction; return list of flipped positions."""
        flipped = []
        r, c = row + dr, col + dc
        while self._in_bounds(r, c) and self.grid[r][c] == -player_id:
            self.grid[r][c] = player_id
            flipped.append((r, c))
            r += dr
            c += dc
        return flipped

    @staticmethod
    def _in_bounds(r, c):
        return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


# ---------------------------------------------------------------------------
# BoardRenderer  (shared rendering — no window ownership)
# ---------------------------------------------------------------------------

class BoardRenderer:
    """Renders the Othello board, tokens, hints, and HUD onto any surface.

    Used by both ``GameScreen`` (interactive game) and ``OthelloPygameEnv``
    (Gymnasium training environment) so that rendering logic is defined once.
    The renderer does **not** own a window or clock — callers are responsible
    for presenting the surface they pass in.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board_x = 0
        self.board_y = 0
        self._cell_px = 0.0
        self._static_board = None
        self._build_layout()

    # -- layout ---------------------------------------------------------------

    def resize(self, width, height):
        """Recompute layout for a new window size."""
        self.width = width
        self.height = height
        self._build_layout()

    def _build_layout(self):
        self.board_x = (self.width - BOARD_PX) // 2
        self.board_y = (self.height - BOARD_PX) // 2
        self._cell_px = (BOARD_PX - 2 * BORDER_WIDTH) / GRID_SIZE
        self._static_board = None  # force rebuild

    # -- coordinate helpers ---------------------------------------------------

    def cell_center(self, row, col):
        """Return the pixel centre (x, y) of board cell (row, col)."""
        x = int(self.board_x + BORDER_WIDTH + col * self._cell_px + self._cell_px / 2)
        y = int(self.board_y + BORDER_WIDTH + row * self._cell_px + self._cell_px / 2)
        return x, y

    def mouse_to_cell(self, mx, my):
        """Convert mouse position to (row, col) or *None* if outside the board."""
        inner_x = mx - self.board_x - BORDER_WIDTH
        inner_y = my - self.board_y - BORDER_WIDTH
        if inner_x < 0 or inner_y < 0:
            return None
        col = int(inner_x // self._cell_px)
        row = int(inner_y // self._cell_px)
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None

    # -- drawing methods ------------------------------------------------------

    def draw_board(self, surface):
        """Blit the cached static board (border + green inner + grid lines)."""
        if self._static_board is None:
            self._build_static_board()
        surface.blit(self._static_board, (self.board_x, self.board_y))

    def draw_tokens(self, surface, board):
        """Draw all tokens from *board*.grid onto *surface*."""
        token_radius = int(self._cell_px * 0.42)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pid = board.grid[r][c]
                if pid != 0:
                    color = BLACK if pid == BLACK_ID else WHITE
                    cx, cy = self.cell_center(r, c)
                    pygame.gfxdraw.aacircle(surface, cx, cy, token_radius, color)
                    pygame.gfxdraw.filled_circle(surface, cx, cy, token_radius, color)

    def draw_hints(self, surface, valid_moves):
        """Draw translucent hint dots for *valid_moves* onto *surface*."""
        if not valid_moves:
            return
        hint_radius = int(self._cell_px * 0.15)
        hint_surf = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        for r, c in valid_moves:
            cx, cy = self.cell_center(r, c)
            hx = cx - self.board_x
            hy = cy - self.board_y
            pygame.gfxdraw.aacircle(hint_surf, hx, hy, hint_radius, HINT_COLOR)
            pygame.gfxdraw.filled_circle(hint_surf, hx, hy, hint_radius, HINT_COLOR)
        surface.blit(hint_surf, (self.board_x, self.board_y))

    def draw_hud(self, surface, score_text, instruction, message=None):
        """Draw score text above the board, instruction below, optional message below that."""
        font = pygame.font.SysFont("Courier", 22, bold=True)
        small_font = pygame.font.SysFont("Courier", 18)

        score_surf = font.render(score_text, True, DARK_GREY)
        score_rect = score_surf.get_rect(center=(self.width // 2, self.board_y - 30))
        surface.blit(score_surf, score_rect)

        inst_surf = font.render(instruction, True, DARK_GREY)
        inst_rect = inst_surf.get_rect(center=(self.width // 2, self.board_y + BOARD_PX + 25))
        surface.blit(inst_surf, inst_rect)

        if message:
            msg_surf = small_font.render(message, True, (120, 120, 120))
            msg_rect = msg_surf.get_rect(center=(self.width // 2, self.board_y + BOARD_PX + 55))
            surface.blit(msg_surf, msg_rect)

    # -- private helpers ------------------------------------------------------

    def _build_static_board(self):
        """Pre-render the green board + grid lines onto a cached surface."""
        surf = pygame.Surface((BOARD_PX, BOARD_PX))
        surf.fill(WHITE)
        inner = pygame.Rect(
            BORDER_WIDTH, BORDER_WIDTH,
            BOARD_PX - 2 * BORDER_WIDTH, BOARD_PX - 2 * BORDER_WIDTH,
        )
        pygame.draw.rect(surf, DARK_GREEN, inner)
        for i in range(GRID_SIZE + 1):
            offset = BORDER_WIDTH + i * self._cell_px
            pygame.draw.line(surf, BLACK, (offset, BORDER_WIDTH), (offset, BOARD_PX - BORDER_WIDTH))
            pygame.draw.line(surf, BLACK, (BORDER_WIDTH, offset), (BOARD_PX - BORDER_WIDTH, offset))
        self._static_board = surf


# ---------------------------------------------------------------------------
# SplashScreen
# ---------------------------------------------------------------------------

class SplashScreen:
    """Title screen with background image and mode-selection buttons."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.manager = pygame_gui.UIManager((width, height))

        # Load background image
        try:
            raw = pygame.image.load(SPLASH_IMG_PATH).convert()
            self.bg_image = pygame.transform.smoothscale(raw, (width, height))
        except (pygame.error, FileNotFoundError):
            self.bg_image = None

        self._file_dialog = None
        self._create_buttons()

    # -- UI construction ----------------------------------------------------

    def _create_buttons(self):
        btn_w, btn_h, gap = 220, 50, 20
        x = (self.width - btn_w) // 2
        # place buttons in the lower third
        base_y = int(self.height * 0.62)

        self.btn_human = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, base_y, btn_w, btn_h),
            text="Play vs Human",
            manager=self.manager,
        )
        self.btn_agent = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, base_y + btn_h + gap, btn_w, btn_h),
            text="Play vs Agent",
            manager=self.manager,
        )
        self.btn_quit = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(x, base_y + 2 * (btn_h + gap), btn_w, btn_h),
            text="Quit",
            manager=self.manager,
        )

    def _open_file_dialog(self):
        """Open a pygame_gui file dialog for selecting the model directory."""
        rect = pygame.Rect(0, 0, 500, 400)
        rect.center = (self.width // 2, self.height // 2)
        self._file_dialog = pygame_gui.windows.UIFileDialog(
            rect=rect,
            manager=self.manager,
            window_title="Select folder containing the RL model",
            initial_file_path=os.getcwd(),
            allow_picking_directories=True,
        )

    # -- screen interface ---------------------------------------------------

    def handle_event(self, event):
        # Handle file dialog result
        if self._file_dialog is not None:
            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                path = event.text
                self._file_dialog = None
                return ("play_agent", path)
            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._file_dialog:
                    self._file_dialog = None
                    return None
            return None  # swallow other events while dialog is open

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_human:
                return "play_human"
            if event.ui_element == self.btn_agent:
                self._open_file_dialog()
                return None
            if event.ui_element == self.btn_quit:
                return "quit"
        return None

    def update(self, dt):
        self.manager.update(dt)

    def draw(self, surface):
        if self.bg_image is not None:
            surface.blit(self.bg_image, (0, 0))
        else:
            surface.fill(DARK_GREY)
            title_font = pygame.font.SysFont("Courier", 64, bold=True)
            title_surf = title_font.render("OTHELLO", True, WHITE)
            surface.blit(title_surf, title_surf.get_rect(center=(self.width // 2, self.height // 3)))
        self.manager.draw_ui(surface)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.manager.clear_and_reset()
        self.manager.set_window_resolution((width, height))
        self._file_dialog = None
        if self.bg_image is not None:
            raw = pygame.image.load(SPLASH_IMG_PATH).convert()
            self.bg_image = pygame.transform.smoothscale(raw, (width, height))
        self._create_buttons()


# ---------------------------------------------------------------------------
# GameScreen
# ---------------------------------------------------------------------------

class GameScreen:
    """Main game screen — board rendering, player interaction, agent play."""

    def __init__(self, width, height, mode="human", rl_agent=None):
        self.width = width
        self.height = height
        self.mode = mode  # "human" or "agent"
        self.rl_agent = rl_agent
        self.manager = pygame_gui.UIManager((width, height))

        # Shared renderer
        self.renderer = BoardRenderer(width, height)

        # Game objects
        self.board = Board()
        self.players = [
            Player(BLACK_ID, BLACK, "Player 1 (Black)"),
            Player(WHITE_ID, WHITE, "Player 2 (White)"),
        ]
        if mode == "agent":
            self.players[1].label = "RL Agent (White)"

        # State
        self.current_index = 0  # index into self.players; 0 = black, 1 = white
        self.valid_moves = self.board.get_valid_moves(self.current_player.id)
        self.game_over = False
        self.skip_turn = False

        # UI elements
        self._create_ui()

        # Dialog state
        self._dialog = None
        self._dialog_type = None  # "back_confirm" | "game_over"
        self._btn_replay = None
        self._btn_done = None
        self._btn_cancel = None

    # -- properties ---------------------------------------------------------

    @property
    def current_player(self):
        return self.players[self.current_index]

    @property
    def opponent(self):
        return self.players[1 - self.current_index]

    # -- ui -----------------------------------------------------------------

    def _create_ui(self):
        btn_w, btn_h = 180, 40
        self.btn_back = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, self.height - btn_h - 10, btn_w, btn_h),
            text="Back to Menu",
            manager=self.manager,
        )

    # -- screen interface ---------------------------------------------------

    def handle_event(self, event):
        # --- dialog interactions -------------------------------------------
        if self._dialog is not None:
            if self._dialog_type == "back_confirm":
                if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                    self._close_dialog()
                    return "back_to_splash"
                if event.type == pygame_gui.UI_WINDOW_CLOSE:
                    self._close_dialog()
                    return None
            elif self._dialog_type == "game_over":
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self._btn_replay:
                        self._close_dialog()
                        self.reset()
                        return None
                    if event.ui_element == self._btn_done:
                        self._close_dialog()
                        return "back_to_splash"
                    if event.ui_element == self._btn_cancel:
                        self._close_dialog()
                        return None
            return None  # swallow all other events while a dialog is open

        # --- button presses ------------------------------------------------
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_back:
                self._show_back_confirm()
                return None

        # --- keyboard ------------------------------------------------------
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if event.key == pygame.K_q:
                self._end_game()
                return None

        # --- mouse click on board ------------------------------------------
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.game_over:
                self._handle_human_click(event.pos)

        return None

    def update(self, dt):
        self.manager.update(dt)

    def draw(self, surface):
        surface.fill(LIGHT_GREY)

        # -- board, tokens, hints (via shared renderer) ---------------------
        self.renderer.draw_board(surface)
        self.renderer.draw_tokens(surface, self.board)
        if not self.game_over and self.valid_moves:
            self.renderer.draw_hints(surface, self.valid_moves)

        # -- HUD -----------------------------------------------------------
        score_text = (f"{self.players[0].label}: {self.players[0].score}    "
                      f"{self.players[1].label}: {self.players[1].score}")
        if self.game_over:
            instruction = "Game Over"
        elif self.skip_turn:
            instruction = (f"{self.opponent.label} has no moves — "
                           f"{self.current_player.label} plays again")
        else:
            instruction = f"{self.current_player.label} to play"
        self.renderer.draw_hud(
            surface, score_text, instruction,
            message="Press Q to end game  |  ESC to quit",
        )

        # -- pygame_gui overlay ---------------------------------------------
        self.manager.draw_ui(surface)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.renderer.resize(width, height)
        self.manager.clear_and_reset()
        self.manager.set_window_resolution((width, height))
        self._dialog = None
        self._dialog_type = None
        self._btn_replay = None
        self._btn_done = None
        self._btn_cancel = None
        self._create_ui()

    # -- game logic ---------------------------------------------------------

    def reset(self):
        self.board.reset()
        self.players[0].score = 0
        self.players[1].score = 0
        self.current_index = 0
        self.valid_moves = self.board.get_valid_moves(self.current_player.id)
        self.game_over = False
        self.skip_turn = False
        self._dialog = None
        self._dialog_type = None
        self._btn_replay = None
        self._btn_done = None
        self._btn_cancel = None

        # Reload agent model with a fresh seed for variety
        if self.mode == "agent" and self.rl_agent is not None:
            prev_path = self.rl_agent.model_full_path
            seed = int(PCG64().random_raw() * 100 / max(PCG64().random_raw(), 1))
            self.rl_agent = othello_agent.OthelloDQN(
                nb_observations=64, player="white", mode="play", seed=seed,
            )
            self.rl_agent.reload_model(path=prev_path)

    def _handle_human_click(self, pos):
        """Process a mouse click from the human player."""
        # In agent mode, only black (human) may click
        if self.mode == "agent" and self.current_player.id != BLACK_ID:
            return

        cell = self.renderer.mouse_to_cell(*pos)
        if cell is None:
            return
        row, col = cell
        if (row, col) not in self.valid_moves:
            return

        self.board.place_token(row, col, self.current_player.id)
        self._advance_turn()

    def _advance_turn(self):
        """Switch to the next player, handle skip-turn and game-over, trigger agent."""
        # Update scores
        black_score, white_score = self.board.calculate_scores()
        self.players[0].score = black_score
        self.players[1].score = white_score

        # Try switching to opponent
        opp_index = 1 - self.current_index
        opp_moves = self.board.get_valid_moves(self.players[opp_index].id)

        if opp_moves:
            # Normal turn change
            self.current_index = opp_index
            self.valid_moves = opp_moves
            self.skip_turn = False
        else:
            # Opponent cannot move — check if current player can
            cur_moves = self.board.get_valid_moves(self.current_player.id)
            if cur_moves:
                # Skip opponent's turn, current player goes again
                self.valid_moves = cur_moves
                self.skip_turn = True
            else:
                # Neither player can move → game over
                self.game_over = True
                self.valid_moves = set()
                self._end_game()
                return

        # If it is now the agent's turn, play automatically
        if self.mode == "agent" and self.current_player.id == WHITE_ID and not self.game_over:
            self._play_agent_turn()

    def _play_agent_turn(self):
        """Let the RL agent choose and execute a move."""
        if self.rl_agent is None or not self.valid_moves:
            return

        observation = self.board.to_numpy().flatten().reshape((1, 64))
        action = self.rl_agent.choose_action(observation, self.valid_moves)
        row = action // 8
        col = action % 8

        if (row, col) not in self.valid_moves:
            # Fallback: pick first valid move if agent returns invalid action
            row, col = next(iter(self.valid_moves))

        self.board.place_token(row, col, self.current_player.id)
        self._advance_turn()

    def _end_game(self):
        """Show the game-over dialog."""
        self.game_over = True
        black_score, white_score = self.board.calculate_scores()
        self.players[0].score = black_score
        self.players[1].score = white_score

        if black_score > white_score:
            msg = f"{self.players[0].label} wins!  {black_score} - {white_score}"
        elif white_score > black_score:
            msg = f"{self.players[1].label} wins!  {white_score} - {black_score}"
        else:
            msg = f"It's a draw!  {black_score} - {white_score}"

        self._show_game_over(msg)

    # -- dialogs ------------------------------------------------------------

    def _show_back_confirm(self):
        rect = pygame.Rect(0, 0, 420, 200)
        rect.center = (self.width // 2, self.height // 2)
        self._dialog = pygame_gui.windows.UIConfirmationDialog(
            rect=rect,
            manager=self.manager,
            window_title="Confirm",
            action_long_desc="End current game and return to the menu?",
            action_short_name="Yes",
            blocking=True,
        )
        self._dialog_type = "back_confirm"

    def _show_game_over(self, message):
        dialog_w, dialog_h = 440, 210
        rect = pygame.Rect(0, 0, dialog_w, dialog_h)
        rect.center = (self.width // 2, self.height // 2)

        self._dialog = pygame_gui.elements.UIPanel(
            relative_rect=rect,
            starting_height=10,
            manager=self.manager,
        )
        self._dialog_type = "game_over"

        # Title and message labels
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, dialog_w - 20, 30),
            text="Game Over",
            manager=self.manager,
            container=self._dialog,
        )
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 52, dialog_w - 20, 36),
            text=message,
            manager=self.manager,
            container=self._dialog,
        )

        # Three buttons: Replay | Done | Cancel
        btn_w, btn_h = 110, 38
        gap = 10
        total_w = 3 * btn_w + 2 * gap
        bx = (dialog_w - total_w) // 2
        by = dialog_h - btn_h - 20

        self._btn_replay = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(bx, by, btn_w, btn_h),
            text="Replay",
            manager=self.manager,
            container=self._dialog,
        )
        self._btn_done = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(bx + btn_w + gap, by, btn_w, btn_h),
            text="Done",
            manager=self.manager,
            container=self._dialog,
        )
        self._btn_cancel = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(bx + 2 * (btn_w + gap), by, btn_w, btn_h),
            text="Cancel",
            manager=self.manager,
            container=self._dialog,
        )

    def _close_dialog(self):
        if self._dialog is not None:
            self._dialog.kill()  # killing the panel also kills its children
            self._dialog = None
        self._btn_replay = None
        self._btn_done = None
        self._btn_cancel = None
        self._dialog_type = None


# ---------------------------------------------------------------------------
# OthelloGame  (top-level controller)
# ---------------------------------------------------------------------------

class OthelloGame:
    """Owns the pygame window and delegates to the active screen."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Othello")
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.surface = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()
        self.running = True

        self.splash = SplashScreen(self.width, self.height)
        self.game_screen = None
        self.screen = self.splash

    def run(self):
        """Main game loop: events → update → draw."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            # --- events ----------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.surface = pygame.display.set_mode(
                        (self.width, self.height), pygame.RESIZABLE,
                    )
                    self.screen.resize(self.width, self.height)

                result = self.screen.handle_event(event)
                self._route(result)

                self.screen.manager.process_events(event)

            # --- update & draw ---------------------------------------------
            self.screen.update(dt)
            self.screen.draw(self.surface)
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    # -- routing ------------------------------------------------------------

    def _route(self, result):
        if result is None:
            return
        if result == "quit":
            self.running = False
        elif result == "play_human":
            self._start_game("human")
        elif isinstance(result, tuple) and result[0] == "play_agent":
            self._load_agent_and_start(result[1])
        elif result == "back_to_splash":
            self.screen = self.splash
            self.splash.resize(self.width, self.height)
            self.game_screen = None

    def _start_game(self, mode, rl_agent=None):
        self.game_screen = GameScreen(
            self.width, self.height, mode=mode, rl_agent=rl_agent,
        )
        self.screen = self.game_screen

    def _load_agent_and_start(self, model_dir):
        """Instantiate the RL agent from *model_dir* and start the game."""
        if not model_dir:
            return

        seed = int(PCG64().random_raw() * 100 / max(PCG64().random_raw(), 1))
        agent = othello_agent.OthelloDQN(
            nb_observations=64, player="white", mode="play", seed=seed,
        )
        load_ok, msg = agent.load_model(
            path=model_dir, name="OthelloDQN", format_type="model",
        )
        if not load_ok:
            print(f"Failed to load agent: {msg}")
            return  # stay on splash

        print(msg)
        self._start_game("agent", rl_agent=agent)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    game = OthelloGame()
    game.run()
