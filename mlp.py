""" Multilayer perceptron for Q-Learning. """

from board import *
from LogisticRegression import *

import numpy as np
import theano
import theano.tensor as T
import lasgane
import code

class MLP(Controller):

    def __init__(self, board):
        
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

        # import os; cls=lambda: os.system("cls")
        # import code; code.interact(local=locals())

        # Just a couple of compiled Theano functions for sigmoid and softmax.
        # This is for calculating forward pass.
        x1 = T.matrix() # some linear combination wx + b
        z1 = T.nnet.sigmoid(x1)
        self.sigmoid = theano.function([x1], z1)
        z2 = T.nnet.softmax(x1) # can probably take out
        self.softmax = theano.function([x1], z2)