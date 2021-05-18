from builtins import object
import numpy as np

from layers import *
from optim import *

from torch.utils.data import DataLoader
from dataset import trainset, testset, imshow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=16, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
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
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        ############################################################################
        C, H, W = input_dim
        F, HH, WW = num_filters, filter_size, filter_size
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['W2'] = weight_scale * np.random.randn(F * H // 2 * W // 2, hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(F)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
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
        z1, cache1 = conv_forward_naive(X, W1, b1, conv_param)
        z2, cache2 = relu_forward(z1)
        z3, cache3 = max_pool_forward_naive(z2, pool_param)
        z4, cache4 = fc_forward(z3, W2, b2)
        z5,cache5=relu_forward(z4)
        z6, cache6 = fc_forward(z5, W3, b3)
        scores = z6
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
        # for self.params[k].             #
        ############################################################################
        loss, softmax_grad = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        loss += 0.5 * self.reg * np.sum(W3 * W3)

        # backpropagation of gradients
        # dout, grads['W3'], grads['b3'] = ?
        # dout, grads['W2'], grads['b2'] = ?
        # dout, grads['W1'], grads['b1'] = ?
        dout, grads['W3'], grads['b3'] = fc_backward(softmax_grad, cache6)
        dout=relu_backward(dout,cache5)
        dout, grads['W2'], grads['b2'] = fc_backward(dout, cache4)
        dout=max_pool_backward_naive(dout,cache3)
        dout = relu_backward(dout, cache2)
        dout, grads['W1'], grads['b1'] = conv_backward_naive(dout, cache1)

        # L2 regularization
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

if __name__ == '__main__':
    """build your cnn classifier based on ThreeLayerConvNet for mnist. Plot and save your training curves """
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, )
    testloader = DataLoader(testset, batch_size=16, shuffle=True)
    classes = trainset.classes

    model = ThreeLayerConvNet()

    # loop over the dataset multiple times
    num_epoch = 5
    loss_list = []
    for epoch in range(num_epoch):
        loss_sum = 0.0
        print(epoch)
        for i, batch in enumerate(trainloader, 0):
            # get the images; batch is a list of [images, labels]
            images, labels = batch
            loss, grads = model.loss(images, labels)
            loss_sum = loss_sum + loss
            for key in grads:
                model.params[key],config=sgd(model.params[key],grads[key],config={"learning_rate":0.05})
        loss = loss_sum / (i + 1)
        loss_list.append(loss)
    import matplotlib.pyplot as plt

    x = array = np.arange(num_epoch)
    y = loss_list
    # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    my_x_ticks = np.arange(0, num_epoch, 1)
    plt.xticks(my_x_ticks)
    plt.ylabel('loss')
    plt.scatter(x, y, alpha=0.6)
    plt.savefig('./cnn_training_curve.png')
    plt.show()
    pass
