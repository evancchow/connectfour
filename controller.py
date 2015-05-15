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

    # The input to the hidden layer is a board position, and the MLP
    # should calculate the probability of winning from that board position.
    # Therefore, for the MLP we feed in all possible inputs (based on
    # all possible moves the player can make). Once we get the winning
    # probabilities for each of these possible moves, we choose the move
    # with the highest probability. Then, we update the MLP based on that
    # move.

    # input is (n_in, 1)
    # weights are (n_out, n_in)
    # bias is (n_out, 1)
    # so you do dot(weights, input)

    # from http://deeplearning.net/tutorial/mlp.html
    def __init__(self, input, n_in, n_out, W=None, b=None,
            activation=T.nnet.sigmoid, rng=None):
        self.input = input
        if rng is None:
            self.rng = np.random.RandomState(1234)
        if W is None:
            W_values = np.asarray( # initial randomly sampled weight mat.
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_in)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            # to get shared values, call "HiddenLayer.W.get_value()"
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:   
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(self.W, input) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # Output shape. Sends out a vector into the next layer!
        self.num_output = n_out

        # The parameters
        self.params = [self.W, self.b]

        """ TD Lambda stuff here """

        # to update after each move, with gradients
        self.lambda_decay = 0.5
        self.eligibility = 0

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

        self.num_hidden_units = 50

        # The first hidden layer. Feed inputs directly into this.
        self.l_hidden = HiddenLayer(board.flattenScaleCurr(), 
            self.nrows * self.ncols, self.num_hidden_units)

        # The output layer, which is a softmax layer. Note that you only need
        # ONE output unit, since given the input (a possible move -->
        # some board position), you want to estimate the probability
        # of winning.
        self.num_output_units = 1
        self.l_output = LogisticRegression(self.l_hidden.output,
            self.l_hidden.num_output, self.num_output_units)

        # The error functions
        self.negative_log_likelihood = self.l_output.negative_log_likelihood
        self.errors = self.l_output.errors

        # Get parameters
        self.params = self.l_hidden.params + self.l_output.params

        import os; cls=lambda: os.system("cls")
        import code; code.interact(local=locals())

    def play(self, inputBoard):
        # Primary method which trains network and returns the move
        self.train(inputBoard)
        return self.makeMove()

    def train(self, inputBoard):
        # Should start training AFTER first move, otherwise
        # will get tons of 0's ... ?

        # Process input vector. each element is a feature so of shape (-1, 1)
        # where rows are features
        x1 = floatX(inputBoard.flattenScaleCurr())

        # Calculate output with forward pass for each possible move.
        # Get the move with the largest probability estimate of winning
        # (and also store that probability estimate which we use to
        # update with backprop)
        avail_cols = inputBoard.availCols()
        
        # Once you have that largest move & probability estimate, you can
        # now update the neural network using TD lambda.
        pass




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