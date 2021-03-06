import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, size = (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size = (hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        out1, cache1 = affine_forward(X, W1, b1)
        out_relu, cache_relu = relu_forward(out1)
        out2, cache2 = affine_forward(out_relu, W2, b2)
        scores = out2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        dhidden, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        d_relu = relu_backward(dhidden, cache_relu)
        dX, grads['W1'], grads['b1'] = affine_backward(d_relu, cache1)
        
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dims = hidden_dims
        dims.insert(0, input_dim)
        dims.append(num_classes)
        W_params, b_params, gamma_params, beta_params = [],[],[],[]
        for i in range(self.num_layers):
            W_params.append('W' + str(i+1))
            b_params.append('b' + str(i+1))
            if self.use_batchnorm:
                gamma_params.append('gamma' + str(i+1))
                beta_params.append('beta'+ str(i+1))
        if self.use_batchnorm:
                gamma_params.remove('gamma' + str(self.num_layers))
                beta_params.remove('beta' + str(self.num_layers))
        for i in range(self.num_layers):    
            self.params[W_params[i]] = np.random.normal(0, weight_scale, size = (dims[i] , dims[i+1]))
            self.params[b_params[i]] = np.zeros(dims[i+1])
            if self.use_batchnorm and i < self.num_layers-1 :
                self.params[gamma_params[i]] = np.ones((dims[i+1]))
                self.params[beta_params[i]] = np.zeros((dims[i+1]))
        

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches= {}
        W1 = self.params['W1']
        b1 = self.params['b1']
        if self.use_batchnorm:
            gamma1 = self.params['gamma1']
            beta1 = self.params['beta1']
        out, cache1 = affine_forward(X, W1, b1)
        if self.use_batchnorm:
            out, cache_batch1 = batchnorm_forward(out, gamma1, beta1, self.bn_params[0])
            caches['cache_batch1'] = cache_batch1
        out, cache_relu1 = relu_forward(out)
        caches['cache1'], caches['cache_relu1'] = cache1, cache_relu1
        if self.use_dropout:
            out, cache_drop1 = dropout_forward(out, self.dropout_param)
            caches['cache_drop1'] = cache_drop1
            
        for i in range(self.num_layers-1):
            if i == self.num_layers-2:
                W = self.params['W' + str(i+2)]
                b = self.params['b' + str(i+2)]
                out, cache = affine_forward(out, W, b)
                caches['cache' + str(i+2)] = cache
            else:    
                W = self.params['W' + str(i+2)]
                b = self.params['b' + str(i+2)]
                if self.use_batchnorm:
                    gamma = self.params['gamma' + str(i+2)]
                    beta = self.params['beta' + str(i+2)]
                out, cache = affine_forward(out, W, b)
                caches['cache' + str(i+2)] = cache
                if self.use_batchnorm:
                    out, cache_batch = batchnorm_forward(out, gamma, beta, self.bn_params[i+1])
                    caches['cache_batch' + str(i+2)] = cache_batch
                out, cache_relu = relu_forward(out)
                caches['cache_relu' + str(i+2)] = cache_relu
                if self.use_dropout:
                    out, cache_drop = dropout_forward(out, self.dropout_param)
                    caches['cache_drop'+ str(i+2)] = cache_drop
        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        
        num = str(self.num_layers)
        dhidden, grads['W' + num], grads['b'+ num] = affine_backward(dscores, caches['cache'+ num])
       
        
        for i in range(self.num_layers-1):
            num = str(self.num_layers-1-i)
            if self.use_dropout:
                cache_drop = caches['cache_drop' + num]
            cache_relu = caches['cache_relu' + num]
            if self.use_batchnorm:
                cache_batch = caches['cache_batch' + num]
            cache = caches['cache' + num]
            
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, cache_drop)
            d_relu = relu_backward(dhidden, cache_relu)
            if self.use_batchnorm:
                dhidden, grads['gamma' + num], grads['beta' + num] = batchnorm_backward(d_relu, cache_batch)
                dhidden, grads['W'+ num], grads['b'+ num] = affine_backward(dhidden, cache)
            else:
                dhidden, grads['W'+ num], grads['b'+ num] = affine_backward(d_relu, cache)
            
        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            loss += 0.5 * self.reg * (np.sum(W * W))
            grads['W' + str(i+1)] += self.reg * W
        
     
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
