""" For the controller. Add your
    possible controllers here in this file, and you can substitute them
    into board.py. """

from board import *
from LogisticRegression import *

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn

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

        """ Parameters for TD-Lambda """
        self.LAMBDA = 0.5
        self.currprob = [0, 0]
        self.currmove = [0, 0]
        self.backpropsum = 0.0

        """ Initialize neural network """

        self.rng = np.random.RandomState(1234)
        num_hidden = 50
        n_in, n_out = self.nrows*self.ncols, num_hidden

        # learning rate
        self.eta = T.scalar()

        # input
        self.u = T.matrix()
        # target (previous estimate!)
        self.t = T.matrix() # may need to be single unit
        # initial hidden state? maybe don't need
        self.h0 = T.vector()
        
        # weights, input -> hidden
        self.W_hidden = theano.shared(np.asarray(
            self.rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_out, n_in)), dtype=theano.config.floatX)*4)

        # weights, hidden -> output
        num_out = 1
        self.W_out = theano.shared(np.asarray(
            self.rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(num_out, n_out)), dtype=theano.config.floatX)*4)

        # Tiny function for forward
        u_in = T.matrix()
        self.y_out = theano.scalar.as_scalar(T.max(1.0 / (1 + T.exp(-1 * T.dot(self.W_out, T.nnet.sigmoid(T.dot(self.W_hidden, u_in))))))) # only 1-element array, can't get theano to cast to scalar. Had [0][0]

        # here, turn y_out into scalar
        self.f_out = theano.function([u_in], self.y_out)
        # v = board.flattenScaleCurr()
        # import code; code.interact(local=locals())

    # Note that rnn_example.py has example of custom error function
    # Feedforward
    def forward(u, W_hidden, W_out):
        h_t = T.nnet.sigmoid(T.dot(u, W_hidden))
        y_t = T.nnet.sigmoid(T.dot(h_t, W_out)) # the estimate
        return h_t, y_t

    def get_gradients(self, input_vector, lambda_decay, best_prob):
        # Recalculate gradients!

        # A cost function. slightly different since abs val is not differentiable
        y_target = T.scalar()
        error = (lambda_decay * (1/2) * (y_target - self.y_out)**2).sum()
        f_error = theano.function([y_target, self.y_out], error)

        # Get gradients
        grad_wh, grad_wo = T.grad(error, [self.W_hidden, self.W_out])
        return grad_wh, grad_wo


    def backprop(self, gradients, best_prob, backpropsum):
        # Backpropagation for TD-Lambda, to train after every move.
        # Note that you have the best prob passed in as a parameter here,
        # so you can use that as part of your error function.

        # # run forward pass. DEPRECATED because already have forward
        # result, updates = theano.scan(self.forward,
        #     sequences=self.u,
        #     outputs_info=[],
        #     non_sequences=[],
            # n_steps=1)

        # updates = [
            # (param, param + )
        # ]

        import code; code.interact(local=locals())

    def getProbEstimate(self, input_vector):
        # Given the flattened, scaled version of the board,
        # run a forward pass to get the probability estimate, i.e.
        # the value function.
        return self.f_out(input_vector)

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

        """ Run Backprop. """
        # Note that you need to have the lagged lambda and the
        # previous best move. Note that you also need to keep a running
        # sum of the past exponentially discounted backpropagations,
        # so that's why way above you have the variable self.backpropsum.

        ### TODO
        ### Work on this first assuming that backprop works,
        ### then go and actually code backprop using Theano.
        ### you'll need the word doc
        ### THIS IS NOT A LOOP. This is for one iteration!!

        # get gradients, given that you have the lambda and the
        # new best prob
        grads = self.get_gradients(x1, self.LAMBDA, best_prob)

        # then get backprop
        self.backprop(grads, best_prob, self.backpropsum)

        # update lambda, backprop sum

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