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

        # Check available columns to drop piece in.
        # Generally, every controller's specific play() method will need this.
        avail_cols = inputBoard.availCols()
        if len(avail_cols) == 0:
            # A double check to make sure there's available columns.
            # Note: if you want the player to pass a turn, just return -1
            # as below, and in board.py's main method, the player will pass.
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

""" The Neural Network """

from board import *
from controller import *

class MLPPlayer(Controller):

    def play(self, inputBoard):

        # just little shortcut "cls()" for clearing screen
        import os, theano
        import theano.tensor as T
        cls = lambda: os.system("cls")

        # for testing
        inputBoard.randomize()
        def floatX(x): return np.asarray(x, dtype=theano.config.floatX)
        x1 = floatX(inputBoard.board.reshape(-1, 1)) # the input vector
        
        import code; code.interact(local=locals())

        avail_cols = inputBoard.availCols()

""" Neural network classes. """

class HiddenLayer(object):
    # from http://deeplearning.net/tutorial/mlp.html
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input
        if W is None:
            W_values = numpy.asarray( # initial randomly sampled weight mat.
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )