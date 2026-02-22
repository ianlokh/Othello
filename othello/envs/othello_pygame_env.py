"""Othello Gymnasium environment with Pygame rendering.

Replaces the Turtle/tkinter-based OthelloEnv with a Pygame-based
implementation.  Reuses the Board, Player, BoardRenderer classes
from othello_main_pygame and shared constants from othello.constants
for consistent object modelling and rendering across the interactive
game and the RL training pipeline.
"""

import random

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from othello import config as cfg
from othello.constants import (
    FPS,
    GRID_SIZE,
    BOARD_PX,
    BLACK_ID,
    WHITE_ID,
    BLACK,
    WHITE,
    DARK_GREY,
    LIGHT_GREY,
)
from othello_main_pygame import (
    Board,
    Player,
    BoardRenderer,
)

# Window defaults for the training renderer (taller than the interactive UI
# to accommodate the training HUD with pause/terminate buttons).
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

class OthelloPygameEnv(gym.Env):
    """8x8 Othello environment rendered with Pygame.

    Observation
        ``{"state": Box(-1, 1, shape=(64,))}`` — flattened board grid
        where -1 = black, 0 = empty, 1 = white.

    Action
        ``Discrete(64)`` — index = row * 8 + col.

    Reward
        * ``cfg.agent_setting.REWARD`` when White wins.
        * ``cfg.agent_setting.PENALTY`` when Black wins.
        * ``2`` on a draw.
        * ``0`` otherwise.

    Render modes
        * ``"human"``    — opens a Pygame window and draws every frame.
        * ``"rgb_array"`` — returns an (H, W, 3) uint8 numpy array.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        # Game objects (mirroring GameScreen) ----------------------------------
        self.board = None
        self.players = [
            Player(BLACK_ID, BLACK, "Player 1 (Black)"),
            Player(WHITE_ID, WHITE, "Player 2 (White)"),
        ]
        # Add name attribute used by the training script
        self.players[0].name = "black"
        self.players[1].name = "white"

        # State (mirroring GameScreen.current_index / valid_moves) -------------
        self.current_index = 0  # 0 = black, 1 = white
        self.next_possible_actions = set()
        self.game_over = False
        self.winner = None
        self.message_str_line1 = ""
        self.message_str_line2 = ""

        # Training UI state ----------------------------------------------------
        self.paused = False
        self.terminated = False
        self._pause_btn_rect = None
        self._term_btn_rect = None
        self._hud_font = None
        self._hud_small_font = None

        # Spaces ---------------------------------------------------------------
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float64)}
        )

        # Rendering (shared BoardRenderer) ------------------------------------
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._win_width = WINDOW_WIDTH
        self._win_height = WINDOW_HEIGHT
        self.renderer = BoardRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)

    # -- properties (mirroring GameScreen) ------------------------------------

    @property
    def current_player(self):
        return self.players[self.current_index]

    @property
    def opponent(self):
        return self.players[1 - self.current_index]

    @property
    def next_player(self):
        """The player who will act next (used by info dict)."""
        if self.game_over:
            return None
        return self.players[self.current_index]

    # -- coordinate helpers ---------------------------------------------------

    @staticmethod
    def _action_to_pos(action):
        """Convert a flat action index (0-63) to (row, col)."""
        row = action // GRID_SIZE
        col = action % GRID_SIZE
        return row, col

    @staticmethod
    def _pos_to_action(row, col):
        """Convert (row, col) to a flat action index."""
        return row * GRID_SIZE + col

    # -- observation / info helpers -------------------------------------------

    def _get_obs(self):
        return {"state": self.board.to_numpy().flatten()}

    def _get_info(self):
        return {
            "next_player": self.next_player,
            "next_possible_actions": self.next_possible_actions,
            "winner": self.winner,
        }

    # -- Gymnasium API --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = Board()
        self.current_index = 0  # black starts
        self.players[0].score = 0
        self.players[1].score = 0
        self.next_possible_actions = self.board.get_valid_moves(BLACK_ID)
        self.game_over = False
        self.winner = None

        if options and "display_message_line1" in options:
            self.message_str_line1 = options["display_message_line1"]
        if options and "display_message_line2" in options:
            self.message_str_line2 = options["display_message_line2"]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0

        assert self.action_space.contains(action), "Invalid Action"
        row, col = self._action_to_pos(action)
        assert (row, col) in self.next_possible_actions, "Invalid Next Action"

        # Place token using Board (same as GameScreen._handle_human_click)
        self.board.place_token(row, col, self.current_player.id)

        # Advance turn (mirroring GameScreen._advance_turn) --------------------
        black_score, white_score = self.board.calculate_scores()
        self.players[0].score = black_score
        self.players[1].score = white_score

        opp_index = 1 - self.current_index
        opp_moves = self.board.get_valid_moves(self.players[opp_index].id)

        if opp_moves:
            # Normal turn change
            self.current_index = opp_index
            self.next_possible_actions = opp_moves
        else:
            # Opponent cannot move — check if current player can
            cur_moves = self.board.get_valid_moves(self.current_player.id)
            if cur_moves:
                # Skip opponent's turn, current player goes again
                self.next_possible_actions = cur_moves
            else:
                # Neither player can move → game over
                self.next_possible_actions = set()
                self.game_over = True

        # Terminal reward ------------------------------------------------------
        if self.game_over:
            conclusion = "\nGame Over! "
            if black_score == white_score:
                reward += cfg.agent_setting.TIE
                self.winner = "Tie"
                conclusion += "No winner, ends up a Tie"
            elif black_score > white_score:
                self.winner = "Black"
                reward += cfg.agent_setting.PENALTY
                conclusion += "Winner is Black."
            else:
                self.winner = "White"
                reward += cfg.agent_setting.REWARD
                conclusion += "Winner is White."
            print(conclusion)

        # Render ---------------------------------------------------------------
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, self.game_over, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    # -- utility --------------------------------------------------------------

    def get_random_action(self):
        """Return a random valid (row, col) tuple (used during training)."""
        if self.next_possible_actions:
            return random.choice(list(self.next_possible_actions))
        return ()

    # -- Pygame rendering (delegates to shared BoardRenderer) -----------------

    def _ensure_fonts(self):
        """Lazily initialise cached fonts (requires pygame.font to be ready)."""
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont("Courier", 22, bold=True)
            self._hud_small_font = pygame.font.SysFont("Courier", 18)

    def _render_frame(self):
        """Draw the current board state to a Pygame surface.

        When ``self.paused`` is True the method loops internally so the
        window stays responsive while the training loop is blocked.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode(
                (self._win_width, self._win_height), pygame.RESIZABLE,
            )
            pygame.display.set_caption("Othello — Training")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self._ensure_fonts()

        while True:
            # Process pending events (handle resize, buttons, window close)
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.terminated = True
                        self.paused = False
                    elif event.type == pygame.VIDEORESIZE:
                        self._win_width, self._win_height = event.w, event.h
                        self.window = pygame.display.set_mode(
                            (self._win_width, self._win_height), pygame.RESIZABLE,
                        )
                        self.renderer.resize(self._win_width, self._win_height)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_button_click(event.pos)

            canvas = pygame.Surface((self._win_width, self._win_height))
            canvas.fill(LIGHT_GREY)

            # Board, tokens, hints (via shared BoardRenderer)
            self.renderer.draw_board(canvas)
            self.renderer.draw_tokens(canvas, self.board)
            if not self.game_over and self.next_possible_actions:
                self.renderer.draw_hints(canvas, self.next_possible_actions)

            # Training-specific HUD
            self._draw_training_hud(canvas)

            # Present
            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

            # When paused, keep looping so the UI stays responsive.
            # When un-paused (or terminated), return to the caller.
            if not self.paused:
                break

    # -- Training HUD -------------------------------------------------------

    def _draw_training_hud(self, canvas):
        """Render player scores, epoch/winning-rate info, and control buttons."""
        font = self._hud_font
        small_font = self._hud_small_font
        center_x = self._win_width // 2
        board_y = self.renderer.board_y
        board_bottom = board_y + BOARD_PX

        # --- Player scores above the board (stacked, centred) ----------------
        p1_surf = font.render(
            f"{self.players[0].label}: {self.players[0].score}", True, DARK_GREY,
        )
        canvas.blit(p1_surf, p1_surf.get_rect(center=(center_x, board_y - 55)))

        p2_surf = font.render(
            f"{self.players[1].label}: {self.players[1].score}", True, DARK_GREY,
        )
        canvas.blit(p2_surf, p2_surf.get_rect(center=(center_x, board_y - 25)))

        # --- Instruction below the board -------------------------------------
        if self.game_over:
            instruction = f"Game Over — {self.winner}"
        else:
            instruction = f"{self.current_player.label} to play"
        inst_surf = font.render(instruction, True, DARK_GREY)
        canvas.blit(inst_surf, inst_surf.get_rect(center=(center_x, board_bottom + 30)))

        # --- Epoch count -----------------------------------------------------
        if self.message_str_line1:
            msg1_surf = small_font.render(self.message_str_line1, True, (120, 120, 120))
            canvas.blit(msg1_surf, msg1_surf.get_rect(center=(center_x, board_bottom + 62)))

        # --- Winning rate (below epoch) --------------------------------------
        if self.message_str_line2:
            msg2_surf = small_font.render(self.message_str_line2, True, (120, 120, 120))
            canvas.blit(msg2_surf, msg2_surf.get_rect(center=(center_x, board_bottom + 88)))

        # --- Pause / Terminate buttons ---------------------------------------
        btn_w, btn_h = 130, 36
        btn_gap = 24
        total_w = 2 * btn_w + btn_gap
        btn_x = center_x - total_w // 2
        btn_y = board_bottom + 120

        # Pause / Resume
        self._pause_btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        pause_bg = (50, 150, 50) if self.paused else (80, 80, 160)
        pygame.draw.rect(canvas, pause_bg, self._pause_btn_rect, border_radius=5)
        pause_label = "Resume" if self.paused else "Pause"
        pause_surf = small_font.render(pause_label, True, WHITE)
        canvas.blit(pause_surf, pause_surf.get_rect(center=self._pause_btn_rect.center))

        # Terminate
        self._term_btn_rect = pygame.Rect(btn_x + btn_w + btn_gap, btn_y, btn_w, btn_h)
        pygame.draw.rect(canvas, (180, 50, 50), self._term_btn_rect, border_radius=5)
        term_surf = small_font.render("Terminate", True, WHITE)
        canvas.blit(term_surf, term_surf.get_rect(center=self._term_btn_rect.center))

    def _handle_button_click(self, pos):
        """Toggle pause or trigger termination when a button is clicked."""
        if self._pause_btn_rect and self._pause_btn_rect.collidepoint(pos):
            self.paused = not self.paused
        elif self._term_btn_rect and self._term_btn_rect.collidepoint(pos):
            self.terminated = True
            self.paused = False
