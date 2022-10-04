# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Importing stuff
import turtle, sys
from tkinter import *

import numpy as np
import time

# Making the coordinate arrays
grid_pos = [-150, -100, -50, 0, 50, 100, 150]  # 7 lines = 8 grids
black_player = {"id": -1, "colour": "#000000", "label": "Player 1 (Black)", "score": 0}
white_player = {"id": 1, "colour": "#FFFFFF", "label": "Player 2 (White)", "score": 0}
directions = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))  # eight directions


'''
Othello game class
'''


class Othello():

    def __init__(self):
        '''
        Initialises the key game variables

        player_pos_list - list of possible player positions based on grid size
        game_board - n x n board matrix
        curr_player - current player can be white_player or black_player
        token - Turtle graphics object to draw tokens
        instruction - Turtle graphics object for instruction text
        score - Turtle graphics object for score text
        window - Turtle graphics object to display game window and for GUI events
        '''
        self.player_pos_list = None
        self.game_board = None

        # variable for player turn
        self.curr_player = None

        # global instance of token turtle
        self.token = turtle.Turtle()
        self.token.ht()

        # global instance of cross turtle
        self.cross = turtle.Turtle()
        self.cross.ht()

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

        # a set of the possible coordinates (x, y) for the next player
        self.next_possible_actions = set()
        # a set of the possible positions for the player in turn
        self.player_valid_pos = set()

        # if True, show a red plus sign in the grid where the player is allowed to put a piece
        self.show_next_possible_actions_hint = True

        # reset game environment
        self.reset()

    def reset(self):
        '''
        Resets the game
        '''
        # initialise datastructure
        self.player_pos_list = self.generate_player_pos_list(grid_pos)
        self.game_board = np.zeros((len(self.player_pos_list), len(self.player_pos_list)))

        # draw the game board
        self.draw_board(grid_pos)
        # set the initial pieces
        self.init_board(self.player_pos_list)

        # variable for player turn - black always starts first
        self.curr_player = None

        print("Game Reset.")

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

    # We will draw a dot based on the turtle's position as the center of the circle. Because of this, we need a new array
    # called player_pos_list. The values in this will be 25 away from each value in grid_pos, because our grid boxes are 50x50.
    # So, we'll start at -225, then -175, -125, etc.
    # Note that because the grid_pos are actually position of the lines, therefore the number of boxes will be 1 + no of
    # lines thus the player_pos_list should be = 1+len(grid_pos).  When we populate this array of positions, we need to then
    # make sure that the center (i.e. where grid_pos[i] == 0) translates into 2 positions of the player_pos_list[i] and
    # player_pos_list[i+1]
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
    def draw_board(_grid_pos):
        border = 10

        grid_pos_x = max(_grid_pos) + 50 + border
        grid_pos_y = max(_grid_pos) + 50 + border
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

    # initialise game_board
    def init_board(self, _pos_list):

        # turn turtle animation on or off and set a delay for update drawings.
        self.window.delay(0)
        self.window.tracer(False)

        # set the game_board matrix
        self.game_board[3, 3] = black_player['id']
        self.game_board[4, 4] = black_player['id']
        self.game_board[3, 4] = white_player['id']
        self.game_board[4, 3] = white_player['id']

        self.draw_token(3, 3, black_player['colour'], _pos_list)
        self.draw_token(4, 4, black_player['colour'], _pos_list)
        self.draw_token(3, 4, white_player['colour'], _pos_list)
        self.draw_token(4, 3, white_player['colour'], _pos_list)

        # write for next player - first player always black
        self.instruction.clear()
        self.instruction.penup()
        self.instruction.hideturtle()
        self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
        self.instruction.write(black_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

        # draw valid positions on board
        self.show_valid_board_pos(black_player)

        # Perform a TurtleScreen update. To be used when tracer is turned off.
        self.window.update()
        self.window.tracer(True)

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
        if self.game_board[x_ind, y_ind] == 0 and adj_sum > 0 and (x_ind, y_ind) in self.player_valid_pos:
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
                        flip_tokens, flip_seq = self.eval_cell(_x + directions[j][0], _y + directions[j][1], j, player, flip_seq, flip_tokens)
                        if flip_tokens and len(flip_seq) > 0:
                            valid_positions.append((_x, _y))
        return set(valid_positions)

    def show_valid_board_pos(self, player):
        # get all valid positions based on the next player
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
        # check that player is a valid player
        assert player in (black_player, white_player), "illegal player input"

        # place the player on the board
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
                # print(direction, flip_seq)
                # flip all captured positions
                for i in range(len(flip_seq)):
                    self.game_board[flip_seq[i][0], flip_seq[i][1]] = player['id']
                    self.draw_token(flip_seq[i][0], flip_seq[i][1], player['colour'], self.player_pos_list)
                # print(self.game_board)

    # place the token based on the mouse click position x, y
    # this function will then execute all the logic of the game
    def play_token(self, _x_pos, _y_pos):

        # get board index from mouse click x, y pos
        def get_board_index(_x_pos, _y_pos):
            # find the closest index for x, y coordinate
            x_index = 0
            curr_x_diff = 50  # set to 50 because it is the max distance from the mouse pos to the grid
            y_index = 0
            curr_y_diff = 50  # set to 50 because it is the max distance from the mouse pos to the grid
            # find the closest index for x y coordinate
            for i in range(len(self.player_pos_list)):
                if curr_x_diff > abs(self.player_pos_list[i] - _x_pos):
                    x_index = i
                    curr_x_diff = abs(self.player_pos_list[i] - _x_pos)

                if curr_y_diff > abs(self.player_pos_list[i] - _y_pos):
                    y_index = i
                    curr_y_diff = abs(self.player_pos_list[i] - _y_pos)

            return x_index, y_index

        # turn turtle animation on or off and set a delay for update drawings.
        self.window.delay(0)
        self.window.tracer(False)

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
        next_player = white_player if player == black_player else black_player

        # get all valid positions based on the next player and draw valid positions on board
        self.show_valid_board_pos(next_player)

        # calculate white and black score
        _score_white, _score_black = self.calculate_score(self.game_board)
        # assign scores to the player objects
        white_player['score'] = _score_white
        black_player['score'] = _score_black

        self.score.clear()
        self.score.hideturtle()

        self.score.penup()
        self.score.goto(0, -(self.window.window_height() / 2) + 700)
        self.score.write(white_player['label'] + " score:" + str(white_player['score']), align="center",
                         font=("Courier", 24, "bold"))

        self.score.penup()
        self.score.goto(0, -(self.window.window_height() / 2) + 670)
        self.score.write(black_player['label'] + " score:" + str(black_player['score']), align="center",
                         font=("Courier", 24, "bold"))

        # check if there are still positions to play else end the game
        # if (_score_white + _score_black) == (len(grid_pos) + 1) * (len(grid_pos) + 1):
        if len(self.get_valid_board_pos(next_player)) == 0:
            self.window.bye()
        else:
            # write instructions for next player
            self.instruction.clear()
            self.instruction.penup()
            self.instruction.hideturtle()
            self.instruction.goto(0, -(self.window.window_height() / 2) + 100)
            self.instruction.write(next_player['label'] + " To Play", align="center", font=("Courier", 24, "bold"))

            # Perform a TurtleScreen update. To be used when tracer is turned off.
            self.window.update()
            self.window.tracer(True)

    def exit_program(self):
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
        self.window.onkeypress(self.exit_program, "Escape")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # initialise game
    game = Othello()
    # play game
    game.play()

    # calculate final score
    final_score_white, final_score_black = game.calculate_score(game.game_board)
    # display winner
    if final_score_white > final_score_black:
        game.alert_popup("Game Completed", "White Player is the Winner with " + str(final_score_white) + " points", "")
    else:
        game.alert_popup("Game Completed", "Black Player is the Winner with " + str(final_score_black) + " points", "")
