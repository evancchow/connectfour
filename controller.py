""" For the controller. Add your
    possible controllers here in this file, and you can substitute them
    into board.py. """

from board import *

class Controller:
    """ Abstract class for the controller, representing
        a player. """

    def __init__(self, playerColor, board):
        # Initialize a controller with the (empty) board information,
        # and what player it is, either RED or BLACK.
        assert playerColor == 1 or playerColor == 2 # RED or BLACK

        # Store player, board dimensions.
        # don't store actual board, just pass on each move
        self.playerColor = playerColor # never used
        self.board = board # never used
        self.nrows = board.getNumRows()
        self.ncols = board.getNumCols() # technically don't need, in board

        # Could initialize other stuff here, like the convolutional network.

    def play(self, inputBoard):
        """ Make a move. inputBoard is the current board, you pass
            this to the method every time.
            Override this with specific algorithms you want to make """
        pass

import random

class RandPlayer(Controller):
    """ Example controller for a player who plays randomly. """

    def play(self, inputBoard):
        """ Make a move by just choosing a random available column. """
        avail_cols = inputBoard.availCols()
        if len(avail_cols) == 0:
            return -1

        # Return a random column to move in.
        rand_col = random.choice(avail_cols)
        return rand_col 

class StuckPlayer(Controller):
    """ Another demo controller for a player who only plays 
        in the first column. """

    def play(self, inputBoard):
        """ Put piece in first column, otherwise pass. """
        avail_cols = inputBoard.availCols()
        if 0 not in avail_cols:
            return -1
        else:
            return 0 # first column