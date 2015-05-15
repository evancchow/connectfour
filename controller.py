""" For the controller. Add your
    possible controllers here in this file, and you can substitute them
    into board.py. """

from board import *
from LogisticRegression import *

import numpy as np
import theano
import theano.tensor as T
import lasagne

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

class HiddenLayer(object):
    """ A hidden layer """
    # from http://deeplearning.net/tutorial/mlp.html
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
            activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray( # initial randomly sampled weight mat.
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:   
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )


def floatX(x): return np.asarray(x, dtype=theano.config.floatX)

class MLPPlayer(Controller):

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

        """ Initialize neural network """

        # input layer, flattened input so dimensions is (numsquares, 1)
        self.l_in = lasagne.layers.InputLayer(board.flatten_scale())

        # hidden layer
        self.l_hidden = lasagne.layers.DenseLayer(self.l_in, num_units=50)

        # output layer. Number of units is number of columns, but note that
        # some columns may be full at some point. So, continue to train each
        # of the output layer units, but when making a move, if the best
        # (smallest probability of other player winning) column is full,
        # just choose the next best column.
        self.l_out = lasagne.layers.DenseLayer(self.l_hidden, 
            num_units=self.ncols, nonlinearity=T.nnet.softmax)

        objective = lasagne.objectives.Objective(self.l_out,
            loss_function =)


    def play(self, inputBoard):
        # Primary method which trains network and returns the move
        self.train(inputBoard)
        return self.makeMove()

    def train(self, inputBoard):

        # just little shortcut "cls()" for clearing screen
        import os
        cls = lambda: os.system("cls")

        # TEST THEANO
        inputBoard.randomize()
        rng = np.random.RandomState(1234)

        # input vector. each element is a feature so of shape (1, -1)
        x1 = floatX(inputBoard.board.reshape(1, -1))

        # first layer


        # layer1 = HiddenLayer(rng, x1, x1.shape[1], 80) # 80 hidden units
        
        # start up new console with variables
        import code; code.interact(local=locals())

        avail_cols = inputBoard.availCols()

    def makeMove():
        return 1


if __name__=="__main__":
    # questions could be different eval functions

    # debugging
    board = Board(6, 7)
    board.randomize()
    test_player = MLPPlayer(1, board)

    for i in range(5):
        test_player.play(board)