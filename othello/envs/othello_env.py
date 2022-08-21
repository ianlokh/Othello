# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Importing stuff
import turtle, sys
from tkinter import *

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import time

# Making the coordinate arrays
gridpos = [-150, -100, -50, 0, 50, 100, 150]  # 7 lines = 8 grids
black_player = {"id":-1, "colour":"#000000", "label":"Player 1 (Black)", "score":0}
white_player = {"id":1, "colour":"#FFFFFF", "label":"Player 2 (White)", "score":0}
# black_player = [-1, "#000000", "Player 1 (Black)"]
# white_player = [1, "#FFFFFF", "Player 2 (White)"]

'''
Othello game env
'''
class OthelloEnv(gym.Env):

    metadata = {'render_modes':['human']}

    def __init__(self):
        '''
        Initialises the key game variables

        playerposlist - list of possible player positions based on grid size
        game_board - n x n board matrix
        curr_player - current player can be white_player or black_player
        token - Turtle graphics object to draw tokens
        instruction - Turtle graphics object for instruction text
        score - Turtle graphics object for score text
        window - Turtle graphics object to display game window and for GUI events
        '''
        self.playerposlist = None
        self.game_board = None

        # variable for player turn
        self.curr_player = None

        # global instance of token turtle
        self.token = turtle.Turtle()
        self.token.ht()

        # global instance of instruction turtle
        self.instruction = turtle.Turtle()
        self.instruction.ht()

        # global instance of score turtle
        self.score = turtle.Turtle()
        self.score.ht()

        # Window Setup - needs to be here so that we can initialise the window, draw and initialise the game board,
        # capture the mouse clicks using window.mainloop()
        self.window = turtle.Screen()
        self.window.bgcolor("#444444")

        # define action space
        self.action_space = spaces.Discrete(64)  # 8x8 possible positions
        self.STEP_LIMIT = 1000  # safe guard to ensure agent doesn't get stuck in a loop
        self.sleep = 0  # slow the rendering for human

        # reset game environment
        self.reset()

    def reset(self):
        '''
        Resets the game, along with the default snake size and spawning food.
        '''
        # initialise datastructure
        self.playerposlist = self.generate_player_pos_list(gridpos)
        self.game_board = np.zeros((len(self.playerposlist), len(self.playerposlist)))

        # draw the game board
        self.draw_board(gridpos)
        # set the initial pieces
        self.init_board(self.playerposlist)

        # variable for player turn
        self.curr_player = 0

        return self.game_board

    def play(self):
        # each mouse click is to place the token
        self.window.onscreenclick(self.play_token)
        # listen for
        self.window.listen()
        self.window.onkeypress(self.close, "Escape")
        self.window.mainloop()

    @staticmethod
    def alert_popup(title, message, path):
        """Generate a pop-up window for special messages."""
        root = Tk()
        root.title(title)
        w = 400  # popup window width
        h = 200  # popup window height
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - h) / 2
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        m = message
        m += '\n'
        m += path
        w = Label(root, text=m, width=120, height=10)
        w.pack()
        b = Button(root, text="OK", command=root.destroy, width=10)
        b.pack()
        mainloop()

    # We will draws a dot based on the turtle's position as the center of the circle. Because of this, we need a new array
    # called playerposlist. The values in this will be 25 away from each value in gridpos, because our grid boxes are 50x50.
    # So, we'll start at -225, then -175, -125, etc.
    # Note that because the gridpos are actually position of the lines, therefore the number of boxes will be 1 + no of
    # lines thus the playerposlist should be = 1+len(gridpos).  When we populate this array of positions, we need to then
    # make sure that the center (i.e. where gridpos[i] == 0) translates into 2 positions of the playerposlist[i] and
    # playerposlist[i+1]
    @staticmethod
    def generate_player_pos_list(_gridpos):
        lst = [None] * (len(_gridpos) + 1)
        for i in range(len(_gridpos)):
            if _gridpos[i] < 0:
                lst[i] = _gridpos[i] - 25
            elif _gridpos[i] == 0:
                lst[i] = _gridpos[i] - 25
                lst[i + 1] = _gridpos[i] + 25  # to populate the additional position
            elif gridpos[i] > 0:
                lst[i + 1] = _gridpos[i] + 25  # henceforth the indexing on lst would be i+1
        return lst

    @staticmethod
    def draw_board(_gridpos):
        border = 10

        grid_pos_x = max(_gridpos) + 50 + border
        grid_pos_y = max(_gridpos) + 50 + border
        grid_length = grid_pos_x + grid_pos_x

        # Set this int to a number between 0 and 10, inclusive, to change the speed. Usually, lower is slower, except in the
        # case of 0, which is the fastest possible.
        speed = 0
        if speed < 0 or speed > 10:
            raise Exception("Speed out of range! Please input a value between 0 and 10 (inclusive)")

        # Initializing Turtles
        outer = turtle.Turtle()
        inner = turtle.Turtle()
        grid = turtle.Turtle()

        # Hiding the turtles
        outer.ht()
        inner.ht()
        grid.ht()

        # Making the outer border of the game
        outer.speed(speed)
        outer.color("#000000", "#FFFFFF")
        outer.up()
        outer.goto(grid_pos_x, grid_pos_y)
        outer.down()
        outer.begin_fill()
        for z in range(4):
            outer.rt(90)
            outer.fd(grid_length)
        outer.end_fill()

        # Making the inner border of the game
        inner.speed(speed)
        inner.color("#000000", "#358856")
        inner.up()
        inner.goto(grid_pos_x - border, grid_pos_y - border)
        inner.down()
        inner.begin_fill()
        for z in range(4):
            inner.rt(90)
            inner.fd(grid_length - (border * 2))
        inner.end_fill()

        # Making the grid
        grid.speed(speed)
        grid.color("#000000")
        for p in range(len(gridpos)):
            grid.up()
            grid.goto(-grid_pos_x + border, gridpos[p])
            grid.down()
            grid.fd(grid_length - (border * 2))
            grid.lt(90)
            grid.up()
            grid.goto(gridpos[p], -grid_pos_y + border)
            grid.down()
            grid.fd(grid_length - (border * 2))
            grid.rt(90)

    def draw_token(self, x_ind, y_ind, colour, _poslist):
        self.token.speed(0)
        self.token.up()
        self.token.goto(_poslist[x_ind], _poslist[y_ind])
        self.token.dot(40, colour)

    def init_board(self, _poslist):
        # set the game_board matrix
        self.game_board[3, 3] = black_player['id']
        self.game_board[4, 4] = black_player['id']
        self.game_board[3, 4] = white_player['id']
        self.game_board[4, 3] = white_player['id']

        self.draw_token(3, 3, black_player['colour'], _poslist)
        self.draw_token(4, 4, black_player['colour'], _poslist)
        self.draw_token(3, 4, white_player['colour'], _poslist)
        self.draw_token(4, 3, white_player['colour'], _poslist)

        # write for next player
        self.instruction.clear()
        self.instruction.penup()
        self.instruction.hideturtle()
        self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
        self.instruction.write(black_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

    def get_player(self):
        if self.curr_player < 0:
            self.curr_player = 1
            return white_player
        elif self.curr_player > 0:
            self.curr_player = -1
            return black_player
        elif self.curr_player == 0:
            self.curr_player = -1
            return black_player

    # Function that returns all adjacent elements
    @staticmethod
    def get_adjacent(arr, i, j):
        def is_valid_pos(i, j, n, m):
            if i < 0 or j < 0 or i > n - 1 or j > m - 1:
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

    def add_to_board(self, x_ind, y_ind, player):
        # place the player on the board
        self.game_board[x_ind, y_ind] = player['id']

        # draw the player's token on the screen
        self.draw_token(x_ind, y_ind, player['colour'], self.playerposlist)

        def eval_cell(x, y, _direction, _player, _flip_seq, _flip_tokens):
            try:
                cell_value = self.game_board[x, y]

                # if the cell is a 0 or out of bounds then end the recursion and return the current flip state and token
                # list as what is recorded thus far
                if (cell_value == 0) or (y < 0 or y >= 8) or (x < 0 or x >= 8):
                    return _flip_tokens, _flip_seq

                # if the cell is not the player's cell then mark for flipping
                if player['id'] != cell_value:
                    _flip_seq.append([x, y])
                    _flip_tokens = False
                elif player['id'] == cell_value:  # if the cell is the player's token then end the recursion
                    _flip_tokens = True
                    return _flip_tokens, _flip_seq

                # recursion if there are still more cells to evaluate. The evaluation and input to x and y depends on the
                # direction
                if _direction == 0 and cell_value != 0 and (y < self.game_board.shape[1]):
                    _flip_tokens, _flip_seq = eval_cell(x, y + 1, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 1 and cell_value != 0 and (y < self.game_board.shape[1] and x < self.game_board.shape[0]):
                    _flip_tokens, _flip_seq = eval_cell(x + 1, y + 1, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 2 and cell_value != 0 and (x < self.game_board.shape[0]):
                    _flip_tokens, _flip_seq = eval_cell(x + 1, y, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 3 and cell_value != 0 and (y >= 0 and x < self.game_board.shape[0]):
                    _flip_tokens, _flip_seq = eval_cell(x + 1, y - 1, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 4 and cell_value != 0 and (y >= 0):
                    _flip_tokens, _flip_seq = eval_cell(x, y - 1, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 5 and cell_value != 0 and (y >= 0 and x >= 0):
                    _flip_tokens, _flip_seq = eval_cell(x - 1, y - 1, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 6 and cell_value != 0 and (x >= 0):
                    _flip_tokens, _flip_seq = eval_cell(x - 1, y, _direction, _player, flip_seq, _flip_tokens)

                if _direction == 7 and cell_value != 0 and (y < self.game_board.shape[1] and x >= 0):
                    _flip_tokens, _flip_seq = eval_cell(x - 1, y + 1, _direction, _player, flip_seq, _flip_tokens)

                # return at the end of the recursion
                return _flip_tokens, _flip_seq
            except (IndexError, ValueError):
                return False, []

        # validate the play and identify any captured positions
        for direction in range(8):
            flip_seq = []
            flip_tokens = False

            if direction == 0:
                flip_tokens, flip_seq = eval_cell(x_ind, y_ind + 1, direction, player, flip_seq, flip_tokens)

            if direction == 1:
                flip_tokens, flip_seq = eval_cell(x_ind + 1, y_ind + 1, direction, player, flip_seq, flip_tokens)

            if direction == 2:
                flip_tokens, flip_seq = eval_cell(x_ind + 1, y_ind, direction, player, flip_seq, flip_tokens)

            if direction == 3:
                flip_tokens, flip_seq = eval_cell(x_ind + 1, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 4:
                flip_tokens, flip_seq = eval_cell(x_ind, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 5:
                flip_tokens, flip_seq = eval_cell(x_ind - 1, y_ind - 1, direction, player, flip_seq, flip_tokens)

            if direction == 6:
                flip_tokens, flip_seq = eval_cell(x_ind - 1, y_ind, direction, player, flip_seq, flip_tokens)

            if direction == 7:
                flip_tokens, flip_seq = eval_cell(x_ind - 1, y_ind + 1, direction, player, flip_seq, flip_tokens)

            # if there is a valid capture and the list of captured positions is > 0
            if flip_tokens and len(flip_seq) > 0:
                print(direction, flip_seq)
                # flip all captured positions
                for i in range(len(flip_seq)):
                    self.game_board[flip_seq[i][0], flip_seq[i][1]] = player['id']
                    self.draw_token(flip_seq[i][0], flip_seq[i][1], player['colour'], self.playerposlist)
                print(self.game_board)

    # place the token based on the mouse click position x, y this function will then execute all the logic of the game
    # change from play_token to step
    def step(self, action):




        # get board index from mouse click x, y pos
        def get_board_index(_x_pos, _y_pos):
            # find the closest index for x, y coordinate
            x_index = 0
            curr_x_diff = 50  # set to 50 because it is the max distance from the mouse pos to the grid
            y_index = 0
            curr_y_diff = 50  # set to 50 because it is the max distance from the mouse pos to the grid
            # find the closest index for x y coordinate
            for i in range(len(self.playerposlist)):
                if curr_x_diff > abs(self.playerposlist[i] - _x_pos):
                    x_index = i
                    curr_x_diff = abs(self.playerposlist[i] - _x_pos)

                if curr_y_diff > abs(self.playerposlist[i] - _y_pos):
                    y_index = i
                    curr_y_diff = abs(self.playerposlist[i] - _y_pos)

            return x_index, y_index

        # action as coordinates

        # get the board index from the mouse position
        x_ind, y_ind = get_board_index(_x_pos, _y_pos)

        # check that this is a valid position
        if not self.check_board_pos(x_ind, y_ind):
            self.alert_popup("Error", "You cannot place a token here", "")
            return

        # get the current player - which is the next player based on the current state of the board
        player = self.get_player()

        # add the token to the board
        self.add_to_board(x_ind, y_ind, player)

        # get the next player
        if self.curr_player == -1:
            next_player = white_player
        else:
            next_player = black_player

        # display white and black score
        _score_white, _score_black = self.calculate_score(self.game_board)
        self.score.clear()
        self.score.hideturtle()

        self.score.penup()
        self.score.goto(0, -(self.window.window_height() / 2) + 700)
        self.score.write(white_player['label'] + " score:" + str(_score_white), align="center", font=("Courier", 24, "bold"))

        self.score.penup()
        self.score.goto(0, -(self.window.window_height() / 2) + 670)
        self.score.write(black_player['label'] + " score:" + str(_score_black), align="center", font=("Courier", 24, "bold"))

        # check if there are still positions to play else end the game
        if (_score_white + _score_black) == (len(gridpos) + 1) * (len(gridpos) + 1):
            self.window.bye()
        else:
            # write instructions for next player
            self.instruction.clear()
            self.instruction.penup()
            self.instruction.hideturtle()
            self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
            self.instruction.write(next_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

        # return game board as observations
        observations = self.game_board

        return observations, reward, done, info

    def exitprogram(self):
        self.window.bye()
        sys.exit()

    def close(self):
        self.score.clear()
        self.instruction.clear()

        close_msg = turtle.Turtle()
        close_msg.speed(0)
        close_msg.penup()
        close_msg.hideturtle()
        close_msg.goto(0, (self.window.window_height() / 2) - 100)
        close_msg.write("Press ESC again to exit", align="center", font=("Courier", 24, "bold"))

        self.window.listen()
        self.window.onkeypress(self.exitprogram, "Escape")
