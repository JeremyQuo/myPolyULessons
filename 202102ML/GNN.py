import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import tensorflow as tf
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from GraphConvolutionLayer import GraphConvolution
import networkx as nx
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_adj_degree_matrix(input_num):
    # to obtain A
    number=int(math.sqrt(input_num))
    adj = np.zeros((input_num, input_num ), dtype=int)
    for i in range(number):
        for j in range(number):
            temp = np.zeros((number, number), dtype=int)
            for y_delta in [-1, 0, 1]:
                for x_delta in [-1, 0, 1]:
                    if 0 <= i + y_delta < number:
                        if 0 <= j + x_delta < number:
                            temp[i + y_delta][j + x_delta] = 1
            adj[i * number + j] = temp.reshape(number ** 2)
    adj = torch.from_numpy(adj)
    adj = adj.float()
    adj=adj.to(device)
    return adj

def calculate_final_adj(adj):
    # to calculate D^-0.5 * A * D^-0.5 (new adj)
    degree=torch.zeros(adj.size()).to(device)
    for i, row in enumerate(adj):
        degree_num=torch.nonzero(row).size()[0]
        degree[i][i]=1/degree_num
    adj=torch.mm(torch.mm(degree.sqrt(),adj),degree.sqrt())

    adj = adj.float().to(device)
    return adj

# prepare data
def load_data():
    (x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

    # to obtain new feature
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    # enc = OneHotEncoder()
    # Y_train = Y_train.reshape(-1, 1)
    # Y_train = enc.fit_transform(Y_train).toarray()
    # Y_test = Y_test.reshape(-1, 1)
    # Y_test = enc.fit_transform(Y_test).toarray()
    Y_test = torch.from_numpy(Y_test).long()
    Y_train = torch.from_numpy(Y_train).long()
    return x_train, Y_train, x_test, Y_test


class GNN(nn.Module):
    def __init__(self, nfeat=784, num_classes=10):
        super(GNN, self).__init__()
        self.gcn1 = GraphConvolution(nfeat, 256)
        self.diff_gcn1_1 = GraphConvolution(256, 128)
        self.diff_gcn1_2 = GraphConvolution(256, 256)
        self.gcn2 = GraphConvolution(128, 128)
        self.diff_gcn2_1 = GraphConvolution(128, 128)
        self.diff_gcn2_2 = GraphConvolution(128, 64)
        self.gcn3 = GraphConvolution(128, 128)
        self.diff_gcn3_1 = GraphConvolution(128, 128)
        self.diff_gcn3_2 = GraphConvolution(128, 1)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self,x,adj):
        # gcn1
        adj=calculate_final_adj(adj)
        x=self.gcn1(x,adj)

        # Diffpool 1
        z1=self.diff_gcn1_1(x,adj)
        s1=F.softmax(self.diff_gcn1_2(x,adj),dim=1)
        x=torch.mm(s1.t(),z1)
        adj=torch.mm(torch.mm(s1.t(),adj),s1)

        # gcn2
        # adj = calculate_final_adj(adj)
        x = self.gcn2(x,adj)

        # Diffpool 2
        z2 = self.diff_gcn2_1(x, adj)
        s2 = F.softmax(self.diff_gcn2_2(x, adj),dim=1)
        x = torch.mm(s2.t(), z2)
        adj = torch.mm(s2.t(), torch.mm(adj, s2))

        # gcn 3
        # adj = calculate_final_adj(adj)
        x = self.gcn3(x, adj)

        # Diffpool 3
        z3 = self.diff_gcn3_1(x, adj)
        s3 = F.softmax(self.diff_gcn3_2(x, adj),dim=1)
        x = torch.mm(s3.t(), z3)
        adj = torch.mm(s3.t(), torch.mm(adj, s3))

        # fc
        x=self.fc1(x)

        return F.log_softmax(x,dim=1)


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test(x_test, Y_test,model,test_num):
    x_test=x_test.to(device)
    Y_test=Y_test.to(device)
    result=torch.zeros((10000,10))
    adj = create_adj_degree_matrix(x_test.size()[1])
    for i, (feature) in enumerate(x_test):
        output = model(feature,adj)
        # temp = torch.max(output, 1).indices
        # result[i]=temp.data
        result[i]=output
        if i >test_num:
            break
    result=result[:test_num]
    Y_test=Y_test[:test_num]
    print(accuracy(result, Y_test))




setup_seed(0)
x_train, Y_train, x_test, Y_test = load_data()
model = GNN(nfeat=784, num_classes=10)
model.cuda()
num_epochs = 2
optimizer = optim.Adam(model.parameters(), lr=0.01)
x_train=x_train.to(device)
Y_train = Y_train.to(device)
criterion=nn.CrossEntropyLoss()
adj=create_adj_degree_matrix(x_train.size()[1])
for epoch in range(num_epochs):
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    running_loss = 0.0
    for i, (feature, label) in enumerate(zip(x_train, Y_train)):

        model.train()
        # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
        # pytorch中每一轮batch需要设置optimizer.zero_grad

        optimizer.zero_grad()


        output = model(feature,adj)
        loss_train = criterion(output, label.reshape(1))
        loss_train.backward()
        optimizer.step()
        if i>500:
            break



# Check accuracy on test
test(x_test, Y_test,model,20)

