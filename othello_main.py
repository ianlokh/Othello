# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Importing stuff
import turtle
from tkinter import *

import numpy as np
import time

# Making the coordinate arrays
# gridpos = [-150, -100, -50, 0, 50, 100, 150] # 7 lines = 8 grids
gridpos = [-150, -100, -50, 0, 50, 100, 150]
black_player = [-1, "#000000", "Player 1 (Black)"]
white_player = [1, "#FFFFFF", "Player 2 (White)"]

# variable for player turn
curr_player = 0

# global instance of token turtle
token = turtle.Turtle()
token.ht()

# global instance of instruction turtle
instruction = turtle.Turtle()
instruction.ht()

# global instance of score turtle
score = turtle.Turtle()
score.ht()


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
# Note that because the gridpos are actually position of the lines, therefore the number of boxes will be 1 + no of lines
# thus the playerposlist should be = 1+len(gridpos).  When we populate this array of positions, we need to then make sure
# that the center (i.e. where gridpos[i] == 0) translates into 2 positions of the playerposlist[i] and playerposlist[i+1]
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


def draw_board(_gridpos):
    border = 10

    grid_pos_x = max(_gridpos) + 50 + border
    grid_pos_y = max(_gridpos) + 50 + border
    grid_length = grid_pos_x + grid_pos_x

    # Set this int to a number between 0 and 10, inclusive, to change the speed. Usually, lower is slower, except in the case of 0, which is the fastest possible.
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


def draw_token(x_ind, y_ind, colour, _poslist):
    token.speed(0)
    token.up()
    token.goto(_poslist[x_ind], _poslist[y_ind])
    token.dot(40, colour)


def init_board(_poslist):
    # set the game_board matrix
    game_board[3, 3] = black_player[0]
    game_board[4, 4] = black_player[0]
    game_board[3, 4] = white_player[0]
    game_board[4, 3] = white_player[0]

    draw_token(3, 3, black_player[1], _poslist)
    draw_token(4, 4, black_player[1], _poslist)
    draw_token(3, 4, white_player[1], _poslist)
    draw_token(4, 3, white_player[1], _poslist)

    # write for next player
    instruction.clear()
    instruction.penup()
    instruction.hideturtle()
    instruction.goto(0, -(window.window_height() / 2) + 100)
    instruction.write(black_player[2] + " To Play", align="center", font=("Courier", 24, "bold"))


# Function that returns all adjacent elements
def getAdjacent(arr, i, j):

    def isValidPos(i, j, n, m):
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
    if isValidPos(i - 1, j - 1, n, m):
        v.append(arr[i - 1][j - 1])
    # left
    if isValidPos(i - 1, j, n, m):
        v.append(arr[i - 1][j])
    # top left
    if isValidPos(i - 1, j + 1, n, m):
        v.append(arr[i - 1][j + 1])
    # top
    if isValidPos(i, j - 1, n, m):
        v.append(arr[i][j - 1])
    # top right
    if isValidPos(i, j + 1, n, m):
        v.append(arr[i][j + 1])
    # right
    if isValidPos(i + 1, j - 1, n, m):
        v.append(arr[i + 1][j - 1])
    # bottom right
    if isValidPos(i + 1, j, n, m):
        v.append(arr[i + 1][j])
    # bottom
    if isValidPos(i + 1, j + 1, n, m):
        v.append(arr[i + 1][j + 1])

    # Returning the vector
    return v


def check_board_pos(x_ind, y_ind):
    # get all the adjacent cells
    adj = getAdjacent(game_board, x_ind, y_ind)
    adj_sum = 0
    for i in range(len(adj)):
        adj_sum += abs(adj[i])

    # position must be either 0 or near an already placed token
    if game_board[x_ind, y_ind] == 0 and adj_sum > 0:
        valid_pos = True
    else:
        valid_pos = False

    return valid_pos


def calculate_score(_game_board):
    # calculate score the score
    score_white = 0
    score_black = 0

    for i in range(len(_game_board)):
        for j in range(len(_game_board)):
            value = _game_board[i, j]
            if value == white_player[0]:
                score_white += 1
            elif value == black_player[0]:
                score_black += 1

    return score_white, score_black



def add_to_board(x_ind, y_ind, player):
    # place the player on the board
    game_board[x_ind, y_ind] = player[0]

    # draw the player's token on the screen
    draw_token(x_ind, y_ind, player[1], playerposlist)

    flip_seq = []

    def eval_cell(x, y, direction, player):
        global flip_seq
        cell_value = game_board[x, y]
        if cell_value == 0:
            flip_seq = []

        if player[0] != cell_value:
            flip_seq.append([x_ind + x, y_ind + y])

        if direction == 0 and y < game_board.shape[1]:
            eval_cell(x, y + 1, direction, player)

        return flip_seq


    # validate the play and identify any captured positions
    for direction in range(8):

        flip_seq = []
        flip_tokens = False

        # if direction = 0, then we are checking up
        # if direction == 0:
        #     x = 0
        #     y = 1
        #     while y_ind + y < game_board.shape[1]:
        #         cell_value = game_board[x_ind + x, y_ind + y]
        #         if cell_value == 0:
        #             break
        #
        #         if player[0] != cell_value:
        #             flip_seq.append([x_ind + x, y_ind + y])
        #             y += 1
        #         else:
        #             flip_tokens = True
        #             break
        flip_seq = eval_cell(0, 1, 0, player)

        if direction == 1:
            x = 1
            y = 1
            while y_ind + y < game_board.shape[1] and x_ind + x < game_board.shape[0]:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x += 1
                    y += 1
                else:
                    flip_tokens = True
                    break

        if direction == 2:
            x = 1
            y = 0
            while x_ind + x < game_board.shape[0]:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x += 1
                else:
                    flip_tokens = True
                    break

        if direction == 3:
            x = 1
            y = -1
            while y_ind + y >= 0 and x_ind + x < game_board.shape[0]:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x += 1
                    y -= 1
                else:
                    flip_tokens = True
                    break

        if direction == 4:
            x = 0
            y = -1
            while y_ind + y >= 0:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    y -= 1
                else:
                    flip_tokens = True
                    break

        if direction == 5:
            x = -1
            y = -1
            while y_ind + y >= 0 and x_ind + x >= 0:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x -= 1
                    y -= 1
                else:
                    flip_tokens = True
                    break

        if direction == 6:
            x = -1
            y = 0
            while x_ind + x >= 0:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x -= 1
                else:
                    flip_tokens = True
                    break

        if direction == 7:
            x = -1
            y = 1
            while y_ind + y < game_board.shape[1] and x_ind + x >= 0:
                cell_value = game_board[x_ind + x, y_ind + y]
                if cell_value == 0:
                    break

                if player[0] != cell_value:
                    flip_seq.append([x_ind + x, y_ind + y])
                    x -= 1
                    y += 1
                else:
                    flip_tokens = True
                    break

        # if there is a valid capture and the list of captured positions is > 0
        if flip_tokens and len(flip_seq) > 0:
            print(direction, flip_seq)
            # flip all captured positions
            for i in range(len(flip_seq)):
                game_board[flip_seq[i][0], flip_seq[i][1]] = player[0]
                draw_token(flip_seq[i][0], flip_seq[i][1], player[1], playerposlist)
            print(game_board)



# place the token based on the mouse click position x, y
# this function will then execute all the logic of the game
def play_token(_x_pos, _y_pos):

    # get board index from mouse click x, y pos
    def get_board_index(_x_pos, _y_pos):
        # find the closest index for x, y coordinate
        x_index = 0
        curr_x_diff = 50  # set to 50 because it is the width of the grid and is also the max distance from the mouse pos to the grid
        y_index = 0
        curr_y_diff = 50  # set to 50 because it is the width of the grid and is also the max distance from the mouse pos to the grid
        # find the closest index for x y coordinate
        for i in range(len(playerposlist)):
            if curr_x_diff > abs(playerposlist[i] - _x_pos):
                x_index = i
                curr_x_diff = abs(playerposlist[i] - _x_pos)

            if curr_y_diff > abs(playerposlist[i] - _y_pos):
                y_index = i
                curr_y_diff = abs(playerposlist[i] - _y_pos)

        return x_index, y_index

    # get the current player - which is the next player based on the current state of the board
    player = get_player()

    # get the board index from the mouse position
    x_ind, y_ind = get_board_index(_x_pos, _y_pos)

    # check that this is a valid position
    if not check_board_pos(x_ind, y_ind):
        alert_popup("Error", "You cannot place a token here", "")
    else:
        add_to_board(x_ind, y_ind, player)


    # get the next player
    if curr_player == -1:
        next_player = white_player
    else:
        next_player = black_player


    # display white and black score
    _score_white, _score_black = calculate_score(game_board)
    score.clear()
    score.hideturtle()
    score.penup()
    score.goto(0, -(window.window_height() / 2) + 700)
    score.write(white_player[2] + " score:" + str(_score_white), align="center", font=("Courier", 24, "bold"))

    score.penup()
    score.goto(0, -(window.window_height() / 2) + 670)
    score.write(black_player[2] + " score:" + str(_score_black), align="center", font=("Courier", 24, "bold"))


    # write instructions for next player
    instruction.clear()
    instruction.penup()
    instruction.hideturtle()
    instruction.goto(0, -(window.window_height() / 2) + 100)
    instruction.write(next_player[2] + " To Play", align="center", font=("Courier", 24, "bold"))


def get_player():
    global curr_player

    if curr_player < 0:
        curr_player = 1
        return white_player

    if curr_player > 0:
        curr_player = -1
        return black_player

    if curr_player == 0:
        curr_player = -1
        return black_player

    # value = np.sum(game_board)
    # if value < 0:
    #     return white_player
    # elif value > 0:
    #     return black_player
    # elif value == 0:
    #     return black_player  # black always takes the first turn


def exitprogram():
    window.bye()


def close():
    close_msg = turtle.Turtle()
    close_msg.speed(0)
    close_msg.penup()
    close_msg.hideturtle()
    close_msg.goto(0, (window.window_height() / 2) - 100)
    close_msg.write("Press ESC again to exit", align="center", font=("Courier", 24, "bold"))
    window.listen()
    window.onkeypress(exitprogram, "Escape")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # initialise datastructure
    playerposlist = generate_player_pos_list(gridpos)
    game_board = np.zeros((len(playerposlist), len(playerposlist)))

    # Window Setup - needs to be here so that we can initialise the window, draw and initialise the game board,
    # capture the mouse clicks using window.mainloop()
    window = turtle.Screen()
    window.bgcolor("#444444")

    # draw the game board
    draw_board(gridpos)
    # set the initial pieces
    init_board(playerposlist)

    # each mouse click is to place the token
    window.onscreenclick(play_token)

    # listen for
    window.listen()
    window.onkeypress(close, "Escape")
    window.mainloop()

    # calculate score
    final_score_white, final_score_black = calculate_score(game_board)
    # display winner
    if final_score_white > final_score_black:
        alert_popup("Game Completed", "White Player is the Winner with " + str(final_score_white) + " points", "")
    else:
        alert_popup("Game Completed", "Black Player is the Winner with " + str(final_score_black) + " points", "")

    # print(playerposlist)
    # print(game_board)
    # print(white_player[0], white_player[1])
    # print(np.sum(game_board))
