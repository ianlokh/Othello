import random
import numpy as np
import sys, os

import pickle
import pygame
import pygame.gfxdraw
import pygame_gui  # Import pygame_gui
from pygments.styles.solarized import LIGHT_COLORS

# import gym
import gymnasium as gym
from gymnasium import spaces

from othello import config as cfg
from othello import othello_agent

# Constants
DARK_GREY = (40, 40, 40)
WHITE = (255, 255, 255)
DARK_GREEN = (53, 136, 86)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Add this line
BORDER_WIDTH = 10
GRID_SIZE = 8  # 8x8 grid
CELL_SIZE = 50  # Size of each grid cell
BOARD_SIZE = CELL_SIZE * GRID_SIZE  # Total size of the game board

directions = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))  # eight directions


def yes_no_dialog_gui(manager, message):
    """Create a yes/no dialog with properly appearing buttons."""
    screen_rect = pygame.display.get_surface().get_rect()
    dialog_rect = pygame.Rect(0, 0, 400, 200)
    dialog_rect.center = screen_rect.center

    # Create the dialog with default settings first
    dialog = pygame_gui.windows.UIConfirmationDialog(
        rect=dialog_rect,
        manager=manager,
        window_title="Confirm",
        action_long_desc=message,
        action_short_name="Yes",
        blocking=True
    )

    # Force UI update to ensure the dialog is fully initialized
    manager.process_events(pygame.event.Event(pygame.USEREVENT, {}))
    manager.update(0.1)  # Small time delta to process UI changes

    # Get references to buttons (must exist after update)
    confirm_button = dialog.confirm_button
    cancel_button = dialog.cancel_button

    if confirm_button is None or cancel_button is None:
        raise RuntimeError("Dialog buttons failed to initialize!")

    # Set button sizes
    button_width = 100
    button_height = 30
    confirm_button.set_dimensions((button_width, button_height))
    cancel_button.set_dimensions((button_width, button_height))

    # Position buttons side by side at the bottom
    button_spacing = 20
    bottom_margin = 20
    button_top =  button_height + bottom_margin

    confirm_button.set_relative_position((-110, -button_top))
    cancel_button.set_relative_position((-190, -button_top))

    # Change "Cancel" to "No"
    cancel_button.set_text("No")

    # Rebuild buttons to apply changes
    confirm_button.rebuild()
    cancel_button.rebuild()

    # Force a final UI update
    manager.update(0.01)

    return dialog


# Player model class
class Player:
    def __init__(self, player_id, color, color_hex, label):
        self.rl_agent = None
        self.id = player_id
        self.color = color
        self.color_hex = color_hex
        self.label = label
        self.score = 0
        # a set of the possible positions for the player in turn
        self.player_valid_pos = set()

# Othello board model class
class OthelloBoard:
    def __init__(self):
        self.valid_positions = None
        self.board_y = None
        self.board_x = None
        self.game_board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.initialize_starting_positions()
        self.static_board_surface = None

    def initialize_starting_positions(self):
        self.game_board[3][3] = 1
        self.game_board[4][4] = 1
        self.game_board[3][4] = -1
        self.game_board[4][3] = -1

    def create_static_board(self, window_width, window_height):
        """
        Create a surface for the static board and draw the static elements on it.
        """
        self.static_board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        self.static_board_surface.fill(WHITE)
        inner_rect = pygame.Rect(
            BORDER_WIDTH, BORDER_WIDTH,
            BOARD_SIZE - (BORDER_WIDTH * 2), BOARD_SIZE - (BORDER_WIDTH * 2)
        )
        pygame.draw.rect(self.static_board_surface, DARK_GREEN, inner_rect)

        adjusted_cell_size = (BOARD_SIZE - (BORDER_WIDTH * 2)) / GRID_SIZE

        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.static_board_surface, BLACK,
                             (BORDER_WIDTH + i * adjusted_cell_size, BORDER_WIDTH),
                             (BORDER_WIDTH + i * adjusted_cell_size, BOARD_SIZE - BORDER_WIDTH), 1)
            pygame.draw.line(self.static_board_surface, BLACK,
                             (BORDER_WIDTH, BORDER_WIDTH + i * adjusted_cell_size),
                             (BOARD_SIZE - BORDER_WIDTH, BORDER_WIDTH + i * adjusted_cell_size), 1)

        # Calculate the board's position based on the new window dimensions
        # self.board_x = (window_width - BOARD_SIZE) // 2
        self.board_x = 20
        self.board_y = (window_height - BOARD_SIZE) // 2

    def draw(self, screen, window_width, window_height):
        """
        Draw the static board and the tokens on the screen.
        """
        if self.static_board_surface is None:
            self.create_static_board(window_width, window_height)

        # # Draw the static board
        screen.blit(self.static_board_surface, (self.board_x, self.board_y))

        # Draw the tokens
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.game_board[x][y] == -1:
                    self.draw_token(screen, x, y, BLACK, window_width, window_height)
                elif self.game_board[x][y] == 1:
                    self.draw_token(screen, x, y, WHITE, window_width, window_height)

       # Draw valid positions for the current player
        if self.valid_positions is not None:
            for pos in self.valid_positions:
                self.draw_cross(screen, pos[0], pos[1], RED, window_width, window_height)

    def draw_token(self, screen, x, y, color, window_width, window_height):
        """
        Draw a token (circle) on the game board at the specified grid cell.
        """
        # board_x = (window_width - BOARD_SIZE) // 2
        # board_y = (window_height - BOARD_SIZE) // 2
        adjusted_cell_size = (BOARD_SIZE - (BORDER_WIDTH * 2)) / GRID_SIZE
        center_x = int(self.board_x + BORDER_WIDTH + (x * adjusted_cell_size) + (adjusted_cell_size / 2))
        center_y = int(self.board_y + BORDER_WIDTH + (y * adjusted_cell_size) + (adjusted_cell_size / 2))
        pygame.gfxdraw.aacircle(screen, center_x, center_y, 20, color)
        pygame.gfxdraw.filled_circle(screen, center_x, center_y, 20, color)

    def draw_thick_line(self, screen, color, start_pos, end_pos, width):
        """
        Draw a thick anti-aliased line using pygame.gfxdraw.
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = max(abs(dx), abs(dy))
        if length == 0:
            return

        # Calculate the perpendicular vector
        perpendicular_x = -dy / length
        perpendicular_y = dx / length

        # Calculate the four corners of the thick line
        x1 = start_pos[0] + perpendicular_x * width / 2
        y1 = start_pos[1] + perpendicular_y * width / 2
        x2 = start_pos[0] - perpendicular_x * width / 2
        y2 = start_pos[1] - perpendicular_y * width / 2
        x3 = end_pos[0] - perpendicular_x * width / 2
        y3 = end_pos[1] - perpendicular_y * width / 2
        x4 = end_pos[0] + perpendicular_x * width / 2
        y4 = end_pos[1] + perpendicular_y * width / 2

        # Draw the thick line as a polygon
        pygame.gfxdraw.aapolygon(screen, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color)
        pygame.gfxdraw.filled_polygon(screen, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color)

    def draw_cross(self, screen, x, y, color, window_width, window_height, line_width=2):
        """
        Draw an anti-aliased cross on the game board at the specified grid cell.
        """
        adjusted_cell_size = (BOARD_SIZE - (BORDER_WIDTH * 2)) / GRID_SIZE
        center_x = int(self.board_x + BORDER_WIDTH + (x * adjusted_cell_size) + (adjusted_cell_size / 2))
        center_y = int(self.board_y + BORDER_WIDTH + (y * adjusted_cell_size) + (adjusted_cell_size / 2))
        length = 20
        half_length = length // 2

        # Draw anti-aliased lines for the cross with a thicker width
        self.draw_thick_line(screen, color,
                             (center_x - half_length, center_y - half_length),
                             (center_x + half_length, center_y + half_length),
                             line_width)
        self.draw_thick_line(screen, color,
                             (center_x + half_length, center_y - half_length),
                             (center_x - half_length, center_y + half_length),
                             line_width)

    def draw_text(self, screen, window_width, window_height, text, area):
        """
        Draw text in the specified area (either "above" or "below" the game board).
        The text is centered horizontally relative to the game board.
        """
        # Use a system font like Arial
        font = pygame.font.SysFont("Courier", 22)

        # Split the text into lines
        lines = text.split('\n')

        # Calculate the total height of the text block
        line_height = font.get_height()
        total_height = len(lines) * line_height

        # Calculate the starting y position based on the specified area
        if area == "below":
            # Position text 30 pixels below the game board
            text_y = self.board_y + BOARD_SIZE + 30
        elif area == "above":
            # Position text 5 pixels above the game board
            text_y = self.board_y - 5 - total_height
        else:
            raise ValueError("Invalid area specified. Use 'above' or 'below'.")

        # Center the text horizontally relative to the game board
        text_x = self.board_x + (BOARD_SIZE // 2)  # Center of the game board

        # Render each line of text
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, WHITE)
            text_rect = text_surface.get_rect(center=(text_x, text_y + i * line_height))
            screen.blit(text_surface, text_rect)

    def get_board_index(self, mouse_x, mouse_y, board_x, board_y, adjusted_cell_size):
        x_index = int((mouse_x - (self.board_x + BORDER_WIDTH)) // adjusted_cell_size)
        y_index = int((mouse_y - (self.board_y + BORDER_WIDTH)) // adjusted_cell_size)
        x_index = max(0, min(x_index, GRID_SIZE - 1))
        y_index = max(0, min(y_index, GRID_SIZE - 1))
        return x_index, y_index

    def flatten(self):
        return [cell for row in self.game_board for cell in row]

# Controller class
class MainGameScreen:

    def __init__(self, width, height):

        # yes/no dialog
        self.dialog = None
        self.dialog_open = False

        self.width = width
        self.height = height
        self.button_width = 200
        self.button_height = 50

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.manager = pygame_gui.UIManager((self.width, self.height))
        self.background_color = (211, 211, 211)  # Light Grey
        self.back_button = self.create_button("Back to Splash", 0, 0)
        self.load_agent_button = self.create_button("Load Agent",
                                                    self.width - self.button_width - 10,
                                                    10)

        self.board = OthelloBoard()
        self.players = [Player(-1, BLACK, "#000000", "Player 1 (Black)"),
                        Player(1, WHITE, "#FFFFFF", "Player 2 (White)")]
        # black starts first
        self.current_player_index = 0
        # track winner for each round
        self.winner = None

        self.running = True
        self.clicked_cell = None
        self.board.valid_positions = self.get_valid_positions()
        self.game_mode = None  # "human" or "computer"
        self.rl_agent = None  # Add this line to store the AI agent

    def get_board(self):
        """
        Returns the current game board.
        """
        return self.board

    def reset(self):
        self.board = OthelloBoard()
        self.players = [Player(-1, BLACK, "#000000", "Player 1 (Black)"),
                        Player(1, WHITE, "#FFFFFF", "Player 2 (White)")]
        self.current_player_index = 0
        self.running = True
        self.clicked_cell = None
        self.board.valid_positions = self.get_valid_positions()

    def create_button(self, text, x, y):
        if x==0 and y==0:
            # Center the button at the bottom of the screen
            x = (self.width - self.button_width) // 2
            y = self.height - self.button_height - 10

        button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((x, y), (self.button_width, self.button_height)),
            text=text,
            manager=self.manager
        )

        return button

    def handle_events(self, event):
        # If dialog is open, only process dialog-related events
        if self.dialog_open:
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.dialog.confirm_button:
                    self.dialog_open = False
                    self.dialog = None
                    return "back_to_splash"
                elif event.ui_element == self.dialog.cancel_button:
                    self.dialog_open = False
                    self.dialog = None
                    return "continue_game"
            return None  # Ignore all other events while dialog is open

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.back_button:
                # return "back_to_splash"

                # Show modal dialog
                self.dialog = yes_no_dialog_gui(
                    self.manager,
                    "Are you sure you want to end the current game and return to the splash screen?"
                )
                self.dialog_open = True
                return None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return "quit"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            window_width = self.screen.get_width()
            window_height = self.screen.get_height()
            mouse_x, mouse_y = event.pos
            board_x = (window_width - BOARD_SIZE) // 2
            board_y = (window_height - BOARD_SIZE) // 2
            adjusted_cell_size = (BOARD_SIZE - (BORDER_WIDTH * 2)) / GRID_SIZE
            self.clicked_cell = self.board.get_board_index(mouse_x, mouse_y, board_x, board_y, adjusted_cell_size)
            print(self.clicked_cell)
            self.handle_move(self.clicked_cell)  # Handle the move after clicking

        return None

    def handle_move(self, cell):
        x, y = cell
        current_player = self.players[self.current_player_index]

        # Check if the move is valid
        if self.is_valid_move(x, y):
            # Update the board with the new move
            self.board.game_board[x][y] = current_player.id
            self.make_move(x, y)
            self.calculate_scores()  # Update scores after each move

            # Switch to the next player
            self.current_player_index = 1 - self.current_player_index
            self.board.valid_positions = self.get_valid_positions()  # Update valid positions after move

            # Redraw the board
            self.draw()

    def make_move(self, x, y):
        self.board.game_board[x][y] = self.players[self.current_player_index].id
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                self.flip_tokens(x, y, dx, dy)

    def check_direction(self, x, y, dx, dy):
        x += dx
        y += dy
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or self.board.game_board[x][y] != self.players[1 - self.current_player_index].id:
            return False
        while 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if self.board.game_board[x][y] == self.players[self.current_player_index].id:
                return True
            if self.board.game_board[x][y] == 0:
                return False
            x += dx
            y += dy
        return False

    def flip_tokens(self, x, y, dx, dy):
        x += dx
        y += dy
        while self.board.game_board[x][y] == self.players[1 - self.current_player_index].id:
            self.board.game_board[x][y] = self.players[self.current_player_index].id
            x += dx
            y += dy

    def is_valid_move(self, x, y):
        if self.board.game_board[x][y] != 0:
            return False
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                return True
        return False

    def get_valid_positions(self):
        """
        Calculate all valid positions for the current player.
        """
        valid_positions = set()
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.board.game_board[x][y] == 0:  # Check empty cells
                    for dx, dy in directions:
                        if self.check_direction(x, y, dx, dy):
                            valid_positions.add((x, y))
                            break
        return valid_positions

    def calculate_scores(self):
        """
        Calculate the scores for both players based on the current state of the board.
        """
        self.players[0].score = 0
        self.players[1].score = 0
        for row in self.board.game_board:
            for cell in row:
                if cell == -1:
                    self.players[0].score += 1
                elif cell == 1:
                    self.players[1].score += 1

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.manager.clear_and_reset()  # Clear the old UI elements
        self.manager.set_window_resolution((self.width, self.height))
        self.back_button = self.create_button("Back to Splash", 0, 0)

    def draw(self):
        self.screen.fill(self.background_color)  # Clear the screen with the background color
        # Draw the Othello board
        self.board.draw(self.screen, self.width, self.height)
        self.manager.draw_ui(self.screen)

        pygame.display.update()


'''
Othello game env
'''

class OthelloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super(OthelloEnv, self).__init__()

        # define the game board
        self.game_controller = MainGameScreen(800, 700)
        self.game_board = self.game_controller.get_board()

        # define action space
        self.action_space = spaces.Discrete(8 * 8)  # 8x8 possible positions
        # a set of the possible coordinates (x, y) for the next player
        self.next_possible_actions = set()  # a set of the possible coordinates (row, col) for the next player

        # define observation space
        self.observation_shape = (8, 8)  # 8 row by 8 col grid
        self.observation_space = spaces.Dict(
            {
                # the observation is a very large discrete space, and I do not want to use it
                "state": spaces.Box(low=0, high=64, shape=(64,))
                # "state": spaces.Discrete(8 * 8)
            }
        )

        self.running = True

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        helper function to get environment observations
        :return: a set of "state" and 1d array of positions on game board
        """
        return {"state": self.game_board.flatten()}  # self.game_board

    def _get_info(self):
        """
        helper function to get environment information
        :return: a set of next player, set of next possible actions and current state winner
        """
        return {"next_player": self.game_controller.current_player_index,
                "next_possible_actions": self.next_possible_actions,
                "winner": self.game_controller.winner}

    def _action_to_pos(self, action):
        """
        helper function to convert action into game board positions
        :param action: integer 0 - 63 corresponding to each position on the game board
        :return: a set of next player, set of next possible actions and current state winner
        """
        assert self.action_space.contains(action), "Invalid Action"
        y_ind = action % 8
        x_ind = (action // 8) % 8
        return x_ind, y_ind

    def _pos_to_action(self, x_ind, y_ind):
        """
        helper function to convert game board position into integer
        :param x_ind: x position on game board
        :param y_ind: y position on game board
        :return: a set of next player, set of next possible actions and current state winner
        """
        action = (x_ind * 8) + y_ind
        assert self.action_space.contains(action), "Invalid Action"
        return action

    def reset(self, seed=None, options=None):
        """
        Resets the game, along with the default players and initial board positions
        :param seed: set seed including super class
        :param options: not used
        :return: environment observations, environment info
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.game_controller = MainGameScreen(800, 700)
        self.running = True

        # variable for player turn - black always starts first
        self.game_controller.players = [Player(-1, BLACK, "#000000", "Player 1 (Black)"),
                                        Player(1, WHITE, "#FFFFFF", "Player 2 (White)")]
        # black starts first
        self.game_controller.current_player_index = 0
        # track winner for each round
        self.game_controller.winner = None
        self.next_possible_actions = self.game_controller.get_valid_positions()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # @profile(stream=fp)
    def step(self, action):
        """
        Plays one move of the game. Method override from gym class to capture the environment changes for each step.
        :param action: integer 0 - 63 corresponding to each play position on the game board
        :return: observation, reward, done, FALSE, info
        """
        self.game_controller.clicked_cell = self._action_to_pos(action)
        self.game_controller.handle_move(self.game_controller.clicked_cell)



        if done:
            conclusion = "\nGame Over! "
            if _score_black == _score_white:  # Tie
                reward += 2
                self.winner = "Tie"
                conclusion += "No winner, ends up a Tie"
            elif _score_black > _score_white:
                self.winner = "Black"
                reward += cfg.agent_setting.PENALTY  # if player == black_player else cfg.agent_setting.REWARD
                conclusion += "Winner is Black."
            else:
                self.winner = "White"
                reward += cfg.agent_setting.REWARD  # if player == white_player else cfg.agent_setting.PENALTY
                conclusion += "Winner is White."

            print(conclusion)

        # return game board as observations
        observation = self._get_obs()
        # return game information
        info = self._get_info()

        # performance profiling
        # self.prof.disable()

        # additional parameter truncated is always FALSE
        return observation, reward, done, FALSE, info