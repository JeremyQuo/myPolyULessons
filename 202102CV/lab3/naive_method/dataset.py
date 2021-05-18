import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor


trainset = MNIST(root='../data', train=True,
                   download=False, transform=ToTensor())
testset = MNIST(root='../data', train=False,
                  download=False, transform=ToTensor())


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
