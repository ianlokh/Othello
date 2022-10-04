import cProfile as profile
import random
import turtle
from tkinter import *

import gym
import numpy as np
from gym import spaces

# Making the coordinate arrays
grid_pos = [-150, -100, -50, 0, 50, 100, 150]  # 7 lines = 8 grids
black_player = {"id": -1, "name": "black", "colour": "#000000", "label": "Player 1 (Black)", "score": 0}
white_player = {"id": 1, "name": "white", "colour": "#FFFFFF", "label": "Player 2 (White)", "score": 0}
directions = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))  # eight directions

'''
Othello game env
'''


class OthelloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        '''
        Initialises the key game variables

        player_pos_list - list of player positions based on grid size
        game_board - n x n board matrix
        curr_player - current player can be white_player or black_player
        next_player - next player can be white_player or black_player
        token - Turtle graphics object to draw tokens
        instruction - Turtle graphics object for instruction text
        score - Turtle graphics object for score text
        window - Turtle graphics object to display game window and for GUI events
        '''

        # global instance of turtles
        self.outer = None
        self.inner = None
        self.grid = None
        self.token = None
        self.cross = None
        self.instruction = None
        self.score = None
        self.window = None

        self.player_pos_list = None
        self.game_board = None
        self.winner = None

        # variable for player turn
        self.curr_player = None
        self.next_player = None

        # a set of the possible positions for the player in turn
        self.player_valid_pos = set()

        # if True, show a red plus sign in the grid where the player is allowed to put a piece
        self.show_next_possible_actions_hint = True

        # define action space
        self.action_space = spaces.Discrete(8 * 8)  # 8x8 possible positions
        # a set of the possible coordinates (x, y) for the next player
        self.next_possible_actions = set()  # a set of the possible coordinates (row, col) for the next player

        # define observation space
        self.observation_shape = (8, 8)  # it is a 8 row by 8 col grid
        self.observation_space = spaces.Dict(
            {
                # the observation is a very large discrete space, and I do not want to use it
                "state": spaces.Box(low=0, high=64, shape=(64,))
                # "state": spaces.Discrete(8 * 8)
            }
        )

        self.STEP_LIMIT = 1000  # safe guard to ensure agent doesn't get stuck in a loop
        self.sleep = 0  # slow the rendering for human

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None

        # # profiling
        # self.prof = profile.Profile()

    def _get_obs(self):
        return {"state": self.game_board.flatten()}  # self.game_board

    def _get_info(self):
        return {"next_player": self.next_player, "next_possible_actions": self.next_possible_actions,
                "winner": self.winner}

    def _action_to_pos(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        y_ind = action % 8
        x_ind = (action // 8) % 8
        return x_ind, y_ind

    def _pos_to_action(self, x_ind, y_ind):
        action = (x_ind * 8) + y_ind
        assert self.action_space.contains(action), "Invalid Action"
        return action

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            self.outer = turtle.Turtle()
            self.outer.speed(0)
            self.outer.ht()

            self.inner = turtle.Turtle()
            self.inner.speed(0)
            self.inner.ht()

            self.grid = turtle.Turtle()
            self.grid.speed(0)
            self.grid.ht()

            # global instance of token turtle
            self.token = turtle.Turtle()
            self.token.speed(0)
            self.token.ht()

            # global instance of cross turtle
            self.cross = turtle.Turtle()
            self.cross.speed(0)
            self.cross.ht()

            # global instance of instruction turtle
            self.instruction = turtle.Turtle()
            self.instruction.speed(0)
            self.instruction.ht()

            # global instance of score turtle
            self.score = turtle.Turtle()
            self.score.speed(0)
            self.score.ht()

            # Window Setup - needs to be here so that we can initialise the window, draw and initialise the game board,
            # capture the mouse clicks using window.mainloop()
            self.window = turtle.Screen()
            self.window.bgcolor("#444444")
            self.window.colormode(255)

            # draw the game board once only
            self.draw_board(grid_pos, self.outer, self.inner, self.grid)
        else:
            # clear existing turtles
            self.token.clear()
            self.cross.clear()
            self.instruction.clear()
            self.score.clear()

        # initialise the board positions
        self.init_board(self.player_pos_list)
        # return the game_board grid
        return self.game_board

    def reset(self, seed=None, options=None):
        '''
        Resets the game, along with the default snake size and spawning food.
        '''
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # initialise datastructure
        self.player_pos_list = self.generate_player_pos_list(grid_pos)
        self.game_board = np.zeros((len(self.player_pos_list), len(self.player_pos_list)))

        if self.render_mode == "human":
            self._render_frame()

        # variable for player turn - black always starts first
        self.curr_player = None
        self.next_player = black_player
        self.next_possible_actions = self.get_valid_board_pos(black_player)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # We will draws a dot based on the turtle's position as the center of the circle. Because of this, we need a new array
    # called playerposlist. The values in this will be 25 away from each value in gridpos, because our grid boxes are 50x50.
    # So, we'll start at -225, then -175, -125, etc.
    # Note that because the gridpos are actually position of the lines, therefore the number of boxes will be 1 + no of
    # lines thus the playerposlist should be = 1+len(gridpos).  When we populate this array of positions, we need to then
    # make sure that the center (i.e. where gridpos[i] == 0) translates into 2 positions of the playerposlist[i] and
    # playerposlist[i+1]
    @staticmethod
    def generate_player_pos_list(_grid_pos):
        lst = [None] * (len(_grid_pos) + 1)
        for i in range(len(_grid_pos)):
            if _grid_pos[i] < 0:
                lst[i] = _grid_pos[i] - 25
            elif _grid_pos[i] == 0:
                lst[i] = _grid_pos[i] - 25
                lst[i + 1] = _grid_pos[i] + 25  # to populate the additional position
            elif grid_pos[i] > 0:
                lst[i + 1] = _grid_pos[i] + 25  # henceforth the indexing on lst would be i+1
        return lst

    @staticmethod
    def draw_board(_grid_pos, outer, inner, grid):
        border = 10

        grid_pos_x = max(_grid_pos) + 50 + border
        grid_pos_y = max(_grid_pos) + 50 + border
        grid_length = grid_pos_x + grid_pos_x

        # Making the outer border of the game
        # outer.color("#000000", "#FFFFFF")
        outer.color((255, 255, 255))
        outer.up()
        outer.goto(grid_pos_x, grid_pos_y)
        outer.down()
        outer.begin_fill()
        for z in range(4):
            outer.rt(90)
            outer.fd(grid_length)
        outer.end_fill()

        # Making the inner border of the game
        # inner.color("#000000", "#358856")
        inner.color((53, 136, 86))
        inner.up()
        inner.goto(grid_pos_x - border, grid_pos_y - border)
        inner.down()
        inner.begin_fill()
        for z in range(4):
            inner.rt(90)
            inner.fd(grid_length - (border * 2))
        inner.end_fill()

        # Making the grid
        grid.color((0, 0, 0))
        for p in range(len(grid_pos)):
            grid.up()
            grid.goto(-grid_pos_x + border, grid_pos[p])
            grid.down()
            grid.fd(grid_length - (border * 2))
            grid.lt(90)
            grid.up()
            grid.goto(grid_pos[p], -grid_pos_y + border)
            grid.down()
            grid.fd(grid_length - (border * 2))
            grid.rt(90)

    # draw token
    def draw_token(self, x_ind, y_ind, colour, _pos_list):
        self.token.speed(0)
        self.token.up()
        self.token.goto(_pos_list[x_ind], _pos_list[y_ind])
        self.token.dot(40, colour)

    # draw cross
    def draw_cross(self, x_ind, y_ind, colour, width, length, _pos_list):
        self.cross.speed(0)
        self.cross.width(width)
        self.cross.color(colour)
        self.cross.penup()
        self.cross.goto(_pos_list[x_ind], _pos_list[y_ind])
        self.cross.pendown()
        self.cross.right(45)
        self.cross.forward(length)
        self.cross.backward(length * 2)
        self.cross.forward(length)
        self.cross.left(90)
        self.cross.forward(length)
        self.cross.backward(length * 2)

    # initialise gameboard
    def init_board(self, _pos_list):

        # set the game_board matrix
        self.game_board[3, 3] = black_player['id']
        self.game_board[4, 4] = black_player['id']
        self.game_board[3, 4] = white_player['id']
        self.game_board[4, 3] = white_player['id']

        if self.render_mode == "human":
            # turn turtle animation on or off and set a delay for update drawings.
            self.window.delay(0)
            self.window.tracer(0, 0)

            self.draw_token(3, 3, black_player['colour'], _pos_list)
            self.draw_token(4, 4, black_player['colour'], _pos_list)
            self.draw_token(3, 4, white_player['colour'], _pos_list)
            self.draw_token(4, 3, white_player['colour'], _pos_list)

            # write for next player - first player always black
            self.instruction.clear()
            self.instruction.penup()
            self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
            self.instruction.write(black_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

            # draw valid positions on board
            self.show_valid_board_pos(black_player)

            # Perform a TurtleScreen update. To be used when tracer is turned off.
            self.window.update()

    # get the next player
    def get_player(self):
        if self.curr_player == black_player:
            self.curr_player = white_player
        elif self.curr_player == white_player:
            self.curr_player = black_player
        elif self.curr_player is None:
            self.curr_player = black_player
        return self.curr_player

    # Function that returns all adjacent elements
    @staticmethod
    def get_adjacent(arr, i, j):
        def is_valid_pos(_i, _j, _n, _m):
            if _i < 0 or _j < 0 or _i > _n - 1 or _j > _m - 1:
                return 0
            return 1

        # Size of given 2d array
        n = arr.shape[0]
        m = arr.shape[1]

        # Initialising a vector array where adjacent element will be stored
        v = []

        # Checking for all the possible adjacent positions
        # bottom left
        if is_valid_pos(i - 1, j - 1, n, m):
            v.append(arr[i - 1][j - 1])
        # left
        if is_valid_pos(i - 1, j, n, m):
            v.append(arr[i - 1][j])
        # top left
        if is_valid_pos(i - 1, j + 1, n, m):
            v.append(arr[i - 1][j + 1])
        # top
        if is_valid_pos(i, j - 1, n, m):
            v.append(arr[i][j - 1])
        # top right
        if is_valid_pos(i, j + 1, n, m):
            v.append(arr[i][j + 1])
        # right
        if is_valid_pos(i + 1, j - 1, n, m):
            v.append(arr[i + 1][j - 1])
        # bottom right
        if is_valid_pos(i + 1, j, n, m):
            v.append(arr[i + 1][j])
        # bottom
        if is_valid_pos(i + 1, j + 1, n, m):
            v.append(arr[i + 1][j + 1])

        # Returning the vector
        return v

    @staticmethod
    def calculate_score(_game_board):
        # calculate score the score
        score_white = 0
        score_black = 0

        for i in range(len(_game_board)):
            for j in range(len(_game_board)):
                value = _game_board[i, j]
                if value == white_player['id']:
                    score_white += 1
                elif value == black_player['id']:
                    score_black += 1

        return score_white, score_black

    def check_board_pos(self, x_ind, y_ind):
        # get all the adjacent cells
        adj = self.get_adjacent(self.game_board, x_ind, y_ind)
        adj_sum = 0
        for i in range(len(adj)):
            adj_sum += abs(adj[i])

        # position must be either 0 or near an already placed token
        if self.game_board[x_ind, y_ind] == 0 and adj_sum > 0:
            valid_pos = True
        else:
            valid_pos = False

        return valid_pos

    def get_valid_board_pos(self, player):
        assert player in (black_player, white_player), "illegal player input"
        flip_seq = []
        flip_tokens = False
        valid_positions = []
        for x in range(8):
            for y in range(8):
                if self.game_board[x, y] in (0, player['id']): continue
                for i in range(len(directions)):
                    _x = x + directions[i][0]
                    _y = y + directions[i][1]
                    if (_x < 0 or _x > 7) or (_y < 0 or _y > 7) or self.game_board[_x, _y] != 0: continue
                    for j in range(len(directions)):
                        flip_tokens, flip_seq = self.eval_cell(_x + directions[j][0], _y + directions[j][1], j, player,
                                                               flip_seq, flip_tokens)
                        if flip_tokens and len(flip_seq) > 0:
                            valid_positions.append((_x, _y))
        return set(valid_positions)

    def show_valid_board_pos(self, player):
        # get all valid positions for the player
        self.player_valid_pos = self.get_valid_board_pos(player)
        # draw possible positions on board
        self.cross.clear()
        for pos in self.player_valid_pos:
            self.cross.setheading(0)
            self.draw_cross(pos[0], pos[1], "NavyBlue", 3, 10, self.player_pos_list)

    # check if the position has any tokens that can be flipped
    def eval_cell(self, x, y, _direction, _player, _flip_seq, _flip_tokens):
        try:
            cell_value = self.game_board[x, y]

            # if the cell is a 0 or out of bounds then end the recursion and return the current flip state and token
            # list as what is recorded thus far
            if (cell_value == 0) or (y < 0 or y >= 8) or (x < 0 or x >= 8):
                # return _flip_tokens, _flip_seq
                return False, []

            # if the cell is not the player's cell then mark for flipping
            if _player['id'] != cell_value:
                _flip_seq.append([x, y])
                _flip_tokens = False
            elif _player['id'] == cell_value:  # if the cell is the player's token then end the recursion
                _flip_tokens = True
                return _flip_tokens, _flip_seq

            # recursion if there are still more cells to evaluate. The evaluation and input to x and y depends on the
            # direction
            if _direction == 0 and cell_value != 0 and (y < self.game_board.shape[1]):
                _flip_tokens, _flip_seq = self.eval_cell(x, y + 1, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 1 and cell_value != 0 and (y < self.game_board.shape[1] and x < self.game_board.shape[0]):
                _flip_tokens, _flip_seq = self.eval_cell(x + 1, y + 1, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 2 and cell_value != 0 and (x < self.game_board.shape[0]):
                _flip_tokens, _flip_seq = self.eval_cell(x + 1, y, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 3 and cell_value != 0 and (y >= 0 and x < self.game_board.shape[0]):
                _flip_tokens, _flip_seq = self.eval_cell(x + 1, y - 1, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 4 and cell_value != 0 and (y >= 0):
                _flip_tokens, _flip_seq = self.eval_cell(x, y - 1, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 5 and cell_value != 0 and (y >= 0 and x >= 0):
                _flip_tokens, _flip_seq = self.eval_cell(x - 1, y - 1, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 6 and cell_value != 0 and (x >= 0):
                _flip_tokens, _flip_seq = self.eval_cell(x - 1, y, _direction, _player, _flip_seq, _flip_tokens)

            if _direction == 7 and cell_value != 0 and (y < self.game_board.shape[1] and x >= 0):
                _flip_tokens, _flip_seq = self.eval_cell(x - 1, y + 1, _direction, _player, _flip_seq, _flip_tokens)

            # return at the end of the recursion
            return _flip_tokens, _flip_seq
        except (IndexError, ValueError):
            return False, []

    # add position to game board
    def add_to_board(self, x_ind, y_ind, player):
        # count the number of flipped tokens
        flipped_cnt = 0

        # place the player on the game board
        self.game_board[x_ind, y_ind] = player['id']

        # draw the player's token on the screen
        self.draw_token(x_ind, y_ind, player['colour'], self.player_pos_list)

        # validate the play and identify any captured positions
        for direction in range(8):
            flip_seq = []
            flip_tokens = False

            if direction == 0:
                flip_tokens, flip_seq = self.eval_cell(x_ind, y_ind + 1, direction, player, flip_seq, flip_tokens)

            if direction == 1:
                flip_tokens, flip_seq = self.eval_cell(x_ind + 1, y_ind + 1, direction, player, flip_seq, flip_tokens)

            if direction == 2:
                flip_tokens, flip_seq = self.eval_cell(x_ind + 1, y_ind, direction, player, flip_seq, flip_tokens)

            if direction == 3:
                flip_tokens, flip_seq = self.eval_cell(x_ind + 1, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 4:
                flip_tokens, flip_seq = self.eval_cell(x_ind, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 5:
                flip_tokens, flip_seq = self.eval_cell(x_ind - 1, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 6:
                flip_tokens, flip_seq = self.eval_cell(x_ind - 1, y_ind, direction, player, flip_seq, flip_tokens)

            if direction == 7:
                flip_tokens, flip_seq = self.eval_cell(x_ind - 1, y_ind + 1, direction, player, flip_seq, flip_tokens)

            # if there is a valid capture and the list of captured positions is > 0
            if flip_tokens and len(flip_seq) > 0:
                # flip all captured positions
                flipped_cnt = len(flip_seq)
                for i in range(len(flip_seq)):
                    self.game_board[flip_seq[i][0], flip_seq[i][1]] = player['id']
                    self.draw_token(flip_seq[i][0], flip_seq[i][1], player['colour'], self.player_pos_list)
                # print(self.game_board)

        return flipped_cnt

    # place the token based on the mouse click position x, y this function will then execute all the logic of the game
    # change from play_token to step
    def step(self, action):

        # self.prof.enable()

        # turn turtle animation on or off and set a delay for update drawings.
        self.window.delay(0)
        self.window.tracer(0, 0)

        # initialise step variables
        reward = 0
        done = False
        # get the current player - which is the next player based on the current state of the board
        # and sets the internal variable curr_player
        player = self.get_player()

        # get x, y index positions from action
        assert self.action_space.contains(action), "Invalid Action"
        x_ind, y_ind = self._action_to_pos(action)

        # add the token to the board and get the number of tokens flipped as the reward
        # change the reward to tune the training of the agents
        assert (x_ind, y_ind) in self.next_possible_actions, "Invalid Next Action"
        self.add_to_board(x_ind, y_ind, player)

        # set the next_player variable
        next_player = white_player if player == black_player else black_player

        # get list of possible actions (positions) for the next player
        possible_actions = self.get_valid_board_pos(next_player)
        # if there is no possible action for next player, then skip the next player and turn now reverts to current
        # player
        if not (len(possible_actions) > 0):
            # back to current player's turn
            curr_possible_actions = self.get_valid_board_pos(player)
            # if even current player cannot place any position then game ends
            if not (len(curr_possible_actions) > 0):
                self.next_possible_actions = set()
                self.next_player = None
                done = True
            else:  # if current player can place position then continue with current player as next player
                reward += 0
                self.next_possible_actions = curr_possible_actions
                self.curr_player = next_player  # this is so that the get_player() is getting the correct player turn
                self.next_player = player  # this is for the info object
        else:  # else if there are possible actions for the next player then capture it
            self.next_possible_actions = possible_actions
            self.next_player = next_player

        # calculate players scores
        _score_white, _score_black = self.calculate_score(self.game_board)
        # assign scores to the player objects
        white_player['score'] = _score_white
        black_player['score'] = _score_black

        self.score.clear()
        self.score.penup()
        # self.score.goto(0, -(self.window.window_height() / 2) + 700)
        self.score.goto(0, (self.window.window_height() / 2) - 100)
        self.score.write(white_player['label'] + " score:" + str(white_player['score']), align="center",
                         font=("Courier", 24, "bold"))

        self.score.penup()
        # self.score.goto(0, -(self.window.window_height() / 2) + 670)
        self.score.goto(0, (self.window.window_height() / 2) - 130)
        self.score.write(black_player['label'] + " score:" + str(black_player['score']), align="center",
                         font=("Courier", 24, "bold"))

        # check if there are still positions to play else end the game
        if done:
            self.instruction.clear()
        else:
            # write instructions for next player
            self.instruction.clear()
            self.instruction.penup()
            self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
            self.instruction.write(next_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

        # Perform a TurtleScreen update. To be used when tracer is turned off.
        self.window.update()

        if done:
            conclusion = "Game Over! "
            if _score_black == _score_white:  # Tie
                reward += 2
                self.winner = "Tie"
                conclusion += "No winner, ends up a Tie"
            elif _score_black > _score_white:
                self.winner = "Black"
                reward += 10 if player == black_player else -10
                conclusion += "Winner is Black."
            else:
                self.winner = "White"
                reward += 10 if player == white_player else -10
                conclusion += "Winner is White."

            print(conclusion)

        # return game board as observations
        observation = self._get_obs()
        # return game information
        info = self._get_info()

        # self.prof.disable()

        # additional parameter truncated is always FALSE
        return observation, reward, done, FALSE, info

    # get random action from list of possible actions (use to train agent)
    def get_random_action(self):
        if self.next_possible_actions:
            return random.choice(list(self.next_possible_actions))
        return ()

    def close(self):
        self.outer.clear()
        self.outer.reset()

        self.inner.clear()
        self.inner.reset()

        self.grid.clear()
        self.grid.reset()

        self.token.clear()
        self.token.reset()

        self.cross.clear()
        self.cross.reset()

        self.instruction.clear()
        self.instruction.reset()

        self.score.clear()
        self.score.reset()

        self.window.bye()
