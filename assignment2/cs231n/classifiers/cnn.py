from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # pass
        C, H, W= input_dim
        F = num_filters
        HH = WW = filter_size
        
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool_h = pool_param['pool_height']
        pool_w = pool_param['pool_width']
        pool_s = pool_param['stride']
        
        # after conv:这里看下面loss的代码实现，conv层已经进行了补全。
        # 所以这里对max_pooling的尺寸进行计算
        pout_h = (H - pool_h)//pool_s + 1
        pout_w = (W - pool_w)//pool_s + 1
        
        weights_dims = [(F, C, HH, WW), (F*pout_h*pout_w, hidden_dim), (hidden_dim, num_classes)]
        bias_dims = [F, hidden_dim, num_classes]
        num_layers = 3
        for i in range(num_layers):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(*weights_dims[i])
            self.params['b'+str(i+1)] = np.zeros(bias_dims[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # pass
        '''
        conv - relu - 2x2 max pool - affine - relu - affine - softmax
        '''
        # conv layer
        conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
        # relu layer
        conv_relu, conv_relu_cache = relu_forward(conv_out)
        # max_pooling layer
        max_out, max_cache = max_pool_forward_fast(conv_relu, pool_param)
        # affine_relu layer
        affine_relu_out, affine_relu_cache = affine_relu_forward(max_out, W2, b2)
        # out_layer
        scores, fc_cache = affine_forward(affine_relu_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # pass
        loss, dscores = softmax_loss(scores, y)
        
        daffine_relu_out, grads['W3'], grads['b3'] = affine_backward(dscores, fc_cache)
        dmax_out, grads['W2'], grads['b2'] = affine_relu_backward(daffine_relu_out, affine_relu_cache)
        dconv_relu = max_pool_backward_fast(dmax_out, max_cache)
        drelu = relu_backward(dconv_relu, conv_relu_cache)
        dX, grads['W1'], grads['b1'] = conv_backward_fast(drelu, conv_cache)
        
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
