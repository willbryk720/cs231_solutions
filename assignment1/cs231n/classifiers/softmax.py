from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    loss = 0.0
    scores = X.dot(W)
    scores += np.expand_dims(-1 * np.amax(scores, axis=1), axis=1)

    for i in range(num_train):
        tot_row_score = 0
        for j in range(num_classes):
            tot_row_score += np.exp(scores[i][j])
            if j == y[i]:
                correct_class_score = np.exp(scores[i][j])
        softmax = correct_class_score/tot_row_score
        loss += -1 * np.log(softmax)
        
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += X[i] * (-1 + np.exp(scores[i][j])/tot_row_score)
            else:
                dW[:,j] += X[i] * np.exp(scores[i][j])/tot_row_score
        
    loss = loss / num_train + reg * np.sum(W * W)
    
    dW = (dW / num_train) + reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Loss:
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W)
    scores += np.expand_dims(-1 * np.amax(scores, axis=1), axis=1)
    
    scores_exp = np.exp(scores)
    
    correct_scores = scores_exp[np.arange(num_train), y]
    total_row_scores = np.sum(scores_exp, axis = 1)
        
    loss = np.sum(-1 * np.log(correct_scores / total_row_scores))
    
    loss /= num_train
    
    # Gradient:
    
    multipliers = np.zeros_like(scores)
    # Put in -1s
    multipliers[np.arange(num_train), y] = -1
        
    multipliers += scores_exp / total_row_scores[:, np.newaxis]
    
    for j in range(num_classes):
        dW[:, j] = np.sum(X * (multipliers[:,j])[:, np.newaxis], axis = 0)
        
    dW = dW / num_train + reg * 2 * W
    
    
   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
