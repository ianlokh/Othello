import pickle
import pygame
import pygame.gfxdraw
import pygame_gui  # Import pygame_gui
import sys, os
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

class OthelloBoard:
    def __init__(self):
        self.board_y = None
        self.board_x = None
        self.game_board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.initialize_starting_positions()
        self.static_board_surface = None

    def initialize_starting_positions(self):
        self.game_board[3][3] = -1
        self.game_board[4][4] = -1
        self.game_board[3][4] = 1
        self.game_board[4][3] = 1

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

        # board_x = (window_width - BOARD_SIZE) // 2
        # board_y = (window_height - BOARD_SIZE) // 2
        # 
        # # Draw the static board
        # screen.blit(self.static_board_surface, (board_x, board_y))
        screen.blit(self.static_board_surface, (self.board_x, self.board_y))

        # Draw the tokens
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.game_board[x][y] == -1:
                    self.draw_token(screen, x, y, BLACK, window_width, window_height)
                elif self.game_board[x][y] == 1:
                    self.draw_token(screen, x, y, WHITE, window_width, window_height)

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
        perp_x = -dy / length
        perp_y = dx / length

        # Calculate the four corners of the thick line
        x1 = start_pos[0] + perp_x * width / 2
        y1 = start_pos[1] + perp_y * width / 2
        x2 = start_pos[0] - perp_x * width / 2
        y2 = start_pos[1] - perp_y * width / 2
        x3 = end_pos[0] - perp_x * width / 2
        y3 = end_pos[1] - perp_y * width / 2
        x4 = end_pos[0] + perp_x * width / 2
        y4 = end_pos[1] + perp_y * width / 2

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

class SplashScreen:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.manager = pygame_gui.UIManager((window_width, window_height))
        self.human_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((window_width // 2 - 100, window_height // 2 - 50), (200, 50)),
            text='Play with Human',
            manager=self.manager
        )
        self.computer_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((window_width // 2 - 100, window_height // 2 + 50), (200, 50)),
            text='Play with Computer',
            manager=self.manager
        )
        self.file_dialog = None

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.human_button:
                    return "human"
                elif event.ui_element == self.computer_button:
                    self.file_dialog = pygame_gui.windows.UIFileDialog(
                        rect=pygame.Rect((self.window_width // 2 - 200, self.window_height // 2 - 150), (400, 300)),
                        manager=self.manager,
                        allow_picking_directories=False,
                        allow_existing_files_only=True,
                        allowed_suffixes={"h5"}
                    )
        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            return event.text
        return None

    def draw(self, screen):
        screen.fill(DARK_GREY)
        self.manager.draw_ui(screen)

class OthelloGame:
    def __init__(self):
        self.board = OthelloBoard()
        self.players = [Player(-1, BLACK, "#000000", "Player 1 (Black)"),
                        Player(1, WHITE, "#FFFFFF", "Player 2 (White)")]
        self.current_player_index = 0
        self.running = True
        self.clicked_cell = None
        self.valid_positions = set()  # Store valid positions for the current player
        self.game_mode = None  # "human" or "computer"
        self.rl_agent = None  # Add this line to store the AI agent
        self.manager = None  # pygame_gui manager
        self.splash_screen = None

    def handle_events(self, window_width, window_height):
        for event in pygame.event.get():

            if self.splash_screen:
                result = self.splash_screen.handle_events(event)
                if result == "human":
                    self.game_mode = "human"
                    self.splash_screen = None
                elif result and result.endswith(".h5"):
                    self.load_rl_agent(result)
                    self.game_mode = "computer"
                    self.splash_screen = None

                # Ensure the pygame_gui manager is updated
                if self.splash_screen and self.splash_screen.manager:
                    self.splash_screen.manager.process_events(event)

            else:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_Q:
                        self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    window_width, window_height = event.size
                    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                    self.board.screen = screen
                    self.board.create_static_board(window_width, window_height)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    board_x = (window_width - BOARD_SIZE) // 2
                    board_y = (window_height - BOARD_SIZE) // 2
                    adjusted_cell_size = (BOARD_SIZE - (BORDER_WIDTH * 2)) / GRID_SIZE
                    self.clicked_cell = self.get_board_index(mouse_x, mouse_y, board_x, board_y, adjusted_cell_size)

            if self.manager:
                self.manager.process_events(event)

    def load_rl_agent(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.rl_agent = pickle.load(f)
                print(f"RL agent loaded from {file_path}")
                self.players[1].rl_agent = self.rl_agent
        except Exception as e:
            print(f"Failed to load RL agent: {e}")

    def get_board_index(self, mouse_x, mouse_y, board_x, board_y, adjusted_cell_size):
        x_index = int((mouse_x - (self.board.board_x + BORDER_WIDTH)) // adjusted_cell_size)
        y_index = int((mouse_y - (self.board.board_y + BORDER_WIDTH)) // adjusted_cell_size)
        x_index = max(0, min(x_index, GRID_SIZE - 1))
        y_index = max(0, min(y_index, GRID_SIZE - 1))
        return x_index, y_index

    def update(self):
        if self.clicked_cell:
            x_index, y_index = self.clicked_cell
            if self.is_valid_move(x_index, y_index):
                self.make_move(x_index, y_index)
                self.current_player_index = 1 - self.current_player_index
                self.valid_positions = self.get_valid_positions()  # Update valid positions after move
                self.calculate_scores()  # Update scores after each move

            self.clicked_cell = None

    def is_valid_move(self, x, y):
        if self.board.game_board[x][y] != 0:
            return False
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                return True
        return False

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

    def make_move(self, x, y):
        self.board.game_board[x][y] = self.players[self.current_player_index].id
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                self.flip_tokens(x, y, dx, dy)

    def flip_tokens(self, x, y, dx, dy):
        x += dx
        y += dy
        while self.board.game_board[x][y] == self.players[1 - self.current_player_index].id:
            self.board.game_board[x][y] = self.players[self.current_player_index].id
            x += dx
            y += dy

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
            text_y = self.board.board_y + BOARD_SIZE + 30
        elif area == "above":
            # Position text 5 pixels above the game board
            text_y = self.board.board_y - 5 - total_height
        else:
            raise ValueError("Invalid area specified. Use 'above' or 'below'.")

        # Center the text horizontally relative to the game board
        text_x = self.board.board_x + (BOARD_SIZE // 2)  # Center of the game board

        # Render each line of text
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, WHITE)
            text_rect = text_surface.get_rect(center=(text_x, text_y + i * line_height))
            screen.blit(text_surface, text_rect)


    def draw(self, screen, window_width, window_height):
        self.board.draw(screen, window_width, window_height)
        if self.clicked_cell:
            x_index, y_index = self.clicked_cell
            self.board.draw_token(screen, x_index, y_index, WHITE, window_width, window_height)
            self.board.draw_cross(screen, x_index + 1, y_index + 1, WHITE, window_width, window_height)
            font = pygame.font.Font(None, 36)
            text = font.render(f"Clicked Cell: ({x_index}, {y_index})", True, WHITE)
            screen.blit(text, (20, 20))

       # Draw valid positions for the current player
        for pos in self.valid_positions:
            self.board.draw_cross(screen, pos[0], pos[1], RED, window_width, window_height)

        # Draw text in the specified areas
        self.draw_text(screen, window_width, window_height, f"Current Player: {self.players[self.current_player_index].label}", "below")
        # Display scores above the game board
        score_text = f"Player 1 (Black): {self.players[0].score}\nPlayer 2 (White): {self.players[1].score}"
        self.draw_text(screen, window_width, window_height, score_text, "above")

def main():
    pygame.init()
    # screen_info = pygame.display.Info()
    # screen_width = screen_info.current_w
    # screen_height = screen_info.current_h
    # window_width = int(screen_width * 0.5)
    # window_height = int(screen_height * 0.75)
    # window_x = (screen_width - window_width) // 2
    # window_y = (screen_height - window_height) // 2
    # os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    window_width = 800
    window_height = 700
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("Pygame Game Board")
    screen.fill(DARK_GREY)
    font = pygame.font.Font(None, 36)
    game = OthelloGame()

    # Initialize the splash screen
    game.splash_screen = SplashScreen(window_width, window_height)

    while game.running:
        time_delta = pygame.time.Clock().tick(60) / 1000.0

        # Handle events
        game.handle_events(window_width, window_height)

        # Draw the splash screen if it's active
        if game.splash_screen:
            game.splash_screen.draw(screen)
            game.splash_screen.manager.update(time_delta)
        else:
            # Update and draw the game board
            game.update()
            screen.fill(DARK_GREY)
            game.draw(screen, window_width, window_height)

        # Update the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
