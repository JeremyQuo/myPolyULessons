from builtins import range
from builtins import object
import numpy as np

from layers import *
from optim import *

import os
import  optim
from torch.utils.data import DataLoader
from dataset import trainset, testset, imshow

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be fc - relu - fc - softmax.
    HINT: use the affine_relu_forward(*) module in the layers.py

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=1 * 28 * 28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        Note that you can change these default values according to your computer.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        b1 = np.zeros((hidden_dim))
        W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b2 = np.zeros((num_classes))

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

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
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # dim_size = X[0].shape
        # X = X.reshape(np.prod(dim_size), X.shape[0])
        X=np.array(X)
        # Computes the hidden layer
        h1, cache_h1 = fc_forward(X, W1, b1)

        # Computes the ReLU of hidden layer
        r1, cache_r1 = relu_forward(h1)

        # Computes out layer
        scores, cache_scores = fc_forward(r1, W2, b2)
        scores=np.array(scores)

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
        # self.params[k].
        ############################################################################
        # calculate loss
        loss, softmax_grad = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # calculate gradient
        # dx2, dw2, db2 = ?
        # dx, dw, db = ?
        # Computes the backward of out layer
        dx2, dw2, db2 = fc_backward(softmax_grad, cache_scores)

        # Computes the backward of ReLU
        dh2 = relu_backward(dx2, cache_r1)

        # Computes the backward of hidden layer
        dx, dw, db = fc_backward(dh2, cache_h1)


        # l2 gradient
        grads['W2'] = dw2 + self.reg * np.array(W2)
        grads['b2'] = db2
        grads['W1'] = dw + self.reg * np.array(W1)
        grads['b1'] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


if __name__ == '__main__':
    """build your fc classifier based on TwoLayerNet for mnist. Plot and save your training curves """

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, )
    testloader = DataLoader(testset, batch_size=128, shuffle=True)
    classes = trainset.classes

    model = TwoLayerNet()


    # loop over the dataset multiple times
    num_epoch = 20
    loss_list=[]
    for epoch in range(num_epoch):
        loss_sum = 0.0
        for i, batch in enumerate(trainloader, 0):
            # get the images; batch is a list of [images, labels]
            images, labels = batch
            loss, grads=model.loss(images,labels)
            loss_sum=loss_sum+loss
            for key in grads:
                model.params[key],config=optim.sgd(model.params[key],grads[key])
        loss=loss_sum/(i+1)

        loss_list.append(loss)
    import matplotlib.pyplot as plt
    x = array = np.arange(num_epoch)
    y = loss_list

    my_x_ticks = np.arange(0, num_epoch, 1)
    plt.xticks(my_x_ticks)
    plt.ylabel('loss')
    plt.scatter(x, y, alpha=0.6)
    plt.savefig('./fc_training_curve.png')
    plt.show()


    print('Finished Training')


