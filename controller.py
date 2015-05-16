""" For the controller. Add your
    possible controllers here in this file, and you can substitute them
    into board.py. """

from board import *
from LogisticRegression import *

import numpy as np
import theano
import theano.tensor as T
import lasagne

import code

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

        self.W = W

        lin_output = T.dot(self.W, input)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # Output shape. Sends out a vector into the next layer!
        self.num_output = n_out

        # The parameters
        self.params = [self.W]

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

        # import os; cls=lambda: os.system("cls")
        # import code; code.interact(local=locals())

        # Just a couple of compiled Theano functions for sigmoid and softmax.
        # This is for calculating forward pass.
        x1 = T.matrix() # some linear combination wx + b
        z1 = T.nnet.sigmoid(x1)
        self.sigmoid = theano.function([x1], z1)
        z2 = T.nnet.softmax(x1) # can probably take out
        self.softmax = theano.function([x1], z2)

        """ For the neural network training """
        # self.l_output.currprob = [0, 0]
        # self.l_output.currmove = [0, 0]
        self.backpropsum = 0.0

        self.currprob = [0, 0]
        self.currmove = [0, 0]

        # Create the error function for the MLP
        self.LAMBDA_DECAY = 0.5
        self.y_t1 = T.dscalar('y_t1')
        self.y_t = T.dscalar('y_t')
        self.lag = T.dscalar('lag')
        # self.f_error = (self.LAMBDA_DECAY**self.lag)*(self.y_t1 - self.y_t)
        # self.f_error = abs(self.y_t1 - self.y_t)
        self.f_error = abs(self.p_y_given_x - )

    def getProbEstimate(self, input_vector):
        # Given the flattened, scaled version of the board,
        # run a forward pass to get the probability estimate, i.e.
        # the value function.

        # should use theano functions if more time
        W_hidden = self.l_hidden.W.get_value()
        W_output = self.l_output.W.get_value()

        # run forward pass
        y_hidden = self.sigmoid(np.dot(W_hidden, input_vector))
        y_output = self.sigmoid(np.dot(W_output, y_hidden))

        # import code; code.interact(local=locals())

        return y_output

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
        best_move = -1 # a column
        best_prob = 0
        for col in avail_cols:
            # Drop a piece into a column to get the possible board
            # if you put a piece there. (Save memory,, not create new boards)
            if self.playerColor == 1:
                inputBoard.playRed(col)
            else:
                inputBoard.playBlack(col)

            board.show()

            # Get the flattened board representation
            input_vect = inputBoard.flattenScaleCurr()

            # Now that you're done evaluating the possible board, remove the
            # piece you just put in
            inputBoard.removePiece(col)

            # Run a forward pass with your neural netwoirk to get the
            # probability estimate.
            est_prob = self.getProbEstimate(input_vect)

            # TODO
            print "The probability estimate %s wins is: %.2f" % (
                self.playerColor, est_prob)

            # update best move, probability
            if best_prob < est_prob:
                best_prob = est_prob
                best_move = col

        # Once you have that largest move & probability estimate, you can
        # now update the neural network using TD lambda.

        # Update last move (maybe need for TD Lambda)
        self.currprob = [self.currprob[-1], best_prob]
        self.currmove = [self.currmove[-1], best_move]

        """ Run Backprop """
        self.backprop()

    def backprop(self):
        print "Calculating error ..."
        print self.params
        gparams = [T.grad(self.f_error, param) 
            for param in self.params]
        code.interact(local=locals())
        pass

    def play(self, inputBoard):
        # Primary method which trains network and returns the move
        self.train(inputBoard)
        return self.currmove
        
if __name__=="__main__":
    # questions could be different eval functions

    # debugging
    board = Board(6, 7)
    board.board = np.asarray([ # for debugging
        [0, 2, 2, 0, 2, 0, 0],
        [0, 1, 2, 0, 1, 0, 0],
        [0, 1, 1, 0, 2, 2, 1],
        [0, 1, 2, 0, 1, 2, 1],
        [2, 2, 2, 1, 1, 1, 2],
        [1, 1, 2, 2, 2, 1, 1]])

    print "Current board:"
    board.show()
    red_test = MLPPlayer(1, board)
    black_test = MLPPlayer(2, board)
    
    for i in xrange(2):
        print "---------- round %s ----------" % i
        red_test.play(board)
        black_test.play(board)