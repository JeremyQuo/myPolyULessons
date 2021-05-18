import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')



class GraphConvolution(Module):
    def __init__(self, input_dimension, output_dimension):
        super(GraphConvolution, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weight = Parameter(torch.FloatTensor(input_dimension, output_dimension))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features,adj):
        # print(features.size())
        if(len(features.size())>1):
            features = torch.mm(adj, features)
        else:
            features=torch.mul(adj,features)
        output = torch.mm(features, self.weight)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dimension) + ' -> ' \
               + str(self.output_dimension) + ')'