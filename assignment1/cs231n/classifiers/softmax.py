import numpy as np
from random import shuffle
# from past.builtins import xrange
from six.moves import xrange

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
  # pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):
    # loss
    scores = X[i].dot(W)
    # c = np.max(scores)
    scores -= scores.max()
    scores_exp = np.exp(scores)
    den_exp = np.sum(scores_exp)
    correct_exp = scores_exp[y[i]]
    loss += -np.log(correct_exp/den_exp)
    
    # grad
    dW[:,y[i]] += (correct_exp/den_exp - 1) * X[i]
    for j in range(num_classes):
        if j == y[i]:
            continue
        dW[:,j] += np.exp(scores[j])/den_exp * X[i]
        
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # pass
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores_max = np.max(scores, axis=1).reshape(num_train,-1)
  scores -= scores_max
  exp_scores = np.exp(scores)
  exp_den = np.sum(exp_scores,axis=1).reshape(num_train,-1)
  correct_index = [np.arange(num_train),y]
  phi = exp_scores/exp_den
  phi_yi = phi[correct_index]
  loss = -np.sum(np.log(phi_yi))
  loss /= num_train
  loss += reg * np.sum(W*W)

  phi[correct_index] -= 1
  dW = X.T.dot(phi)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

